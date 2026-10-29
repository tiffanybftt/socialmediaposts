"""Turn episode/script.json into narration audio plus a timeline.

Outputs (in build/):
  narration.wav   full voice track, silences included
  timeline.json   scene and sentence start/end times, read by the video page
  captions.srt    subtitles for sharing separately
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

voice_dir = Path(sys.argv[1])  # folder holding the .onnx voice files
script = json.loads((ROOT / "episode" / "script.json").read_text())
voice = PiperVoice.load(voice_dir / f"{script['voice']}.onnx")
cfg = SynthesisConfig(length_scale=script["length_scale"])
rate = voice.config.sample_rate


def speak(text: str) -> np.ndarray:
    parts = [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in voice.synthesize(text, cfg)]
    return np.concatenate(parts)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * rate), dtype=np.int16)


track: list[np.ndarray] = []
t = 0.0


def add(chunk: np.ndarray) -> None:
    global t
    track.append(chunk)
    t += len(chunk) / rate


timeline = {"title": script["title"], "subtitle": script["subtitle"], "scenes": []}
for scene in script["scenes"]:
    start = t
    add(silence(script["scene_lead_in"]))
    sentences = []
    for i, s in enumerate(scene["sentences"]):
        audio = speak(s["text"])
        s_start = t
        add(audio)
        sentences.append({"id": s["id"], "caption": s.get("caption", s["text"]), "start": round(s_start, 3), "end": round(t, 3)})
        if i < len(scene["sentences"]) - 1:
            add(silence(script["sentence_gap"]))
    hold_start = t
    add(silence(scene.get("hold_after", 0) + scene.get("tail", script["scene_tail"])))
    timeline["scenes"].append({"id": scene["id"], "phase": scene["phase"], "start": round(start, 3), "end": round(t, 3), "sentences": sentences,
                               "hold": [round(hold_start, 3), round(hold_start + scene.get("hold_after", 0), 3)]})

timeline["duration"] = round(t, 3)
(BUILD / "timeline.json").write_text(json.dumps(timeline, indent=1))
(BUILD / "timeline.js").write_text("window.TIMELINE = " + json.dumps(timeline) + ";\n")  # file:// pages cannot fetch() JSON

pcm = np.concatenate(track)
with wave.open(str(BUILD / "narration.wav"), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(pcm.tobytes())


def stamp(x: float) -> str:
    ms = round(x * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


lines = []
n = 0
for scene in timeline["scenes"]:
    for s in scene["sentences"]:
        n += 1
        lines += [str(n), f"{stamp(s['start'])} --> {stamp(s['end'])}", s["caption"], ""]
(BUILD / "captions.srt").write_text("\n".join(lines))

words = sum(len(s["caption"].split()) for sc in timeline["scenes"] for s in sc["sentences"])
print(f"duration {t:.1f}s ({int(t // 60)}:{t % 60:04.1f})  words {words}  rate {rate}")
for sc in timeline["scenes"]:
    print(f"  {sc['id']:8} {sc['start']:6.1f} -> {sc['end']:6.1f}  ({sc['end'] - sc['start']:.1f}s)")
