"""Assemble the narration from human recordings instead of text-to-speech.

Needs <rec_dir>/take_<id>.wav and <rec_dir>/align.json (see transcribe_takes.py and align_takes.py).
Writes the same three outputs as build_audio.py (build/narration.wav, build/timeline.json + .js, build/captions.srt),
so music, render and mix work unchanged. Each take is level-matched, then placed in its scene with your own pauses.

usage: python build_audio_human.py <rec_dir> [--style upbeat]

--style upbeat trims long pauses (silences over 0.85 s are shortened to about half a second), keeps your words and
breaths untouched, and tells the video page to use its livelier animations.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)
rec = Path(sys.argv[1])
STYLE = "upbeat" if "--style" in sys.argv and sys.argv[sys.argv.index("--style") + 1] == "upbeat" else "calm"

SR = 48000
TARGET_LUFS = -18.0  # every take is brought to this level before the final mix
PAD_BEFORE, PAD_AFTER = 0.30, 0.40  # quiet kept around the first and last word of each take
FADE = 0.04

script = json.loads((ROOT / "episode" / "script.json").read_text())
takes = json.loads((ROOT / "episode" / "takes.json").read_text())
align = json.loads((rec / "align.json").read_text())
sentences = {s["id"]: s for sc in script["scenes"] for s in sc["sentences"]}
scene_of = {s["id"]: sc["id"] for sc in script["scenes"] for s in sc["sentences"]}


def loudness(path: Path) -> float:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                        "highpass=f=70,loudnorm=I=-18:TP=-3:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    return float(re.search(r'"input_i"\s*:\s*"(-?[0-9.]+)"', r.stderr).group(1))


def prepare(take: str) -> tuple[np.ndarray, float, float]:
    raw = rec / f"take_{take}.wav"
    lufs = loudness(raw)
    gain = TARGET_LUFS - lufs
    out = rec / f"clean_{take}.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-af",
                    f"highpass=f=70,volume={gain:.2f}dB,alimiter=limit=0.9:level=0", "-c:a", "pcm_f32le", str(out)], check=True)
    x, sr = sf.read(out, dtype="float32")
    assert sr == SR
    return x, lufs, gain


def compress_silences(seg: np.ndarray, thr_db=-42.0, min_len=0.85, keep_after=0.30, keep_before=0.25):
    """Shorten long pauses. Returns the new audio and a function mapping old times to new times."""
    step = round(0.01 * SR)
    env = 20 * np.log10(np.sqrt((seg[: len(seg) // step * step].reshape(-1, step) ** 2).mean(axis=1)) + 1e-9)
    quiet, cuts, i = env < thr_db, [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * 0.01 >= min_len:
                cuts.append((i * 0.01 + keep_after, j * 0.01 - keep_before))
            i = j
        else:
            i += 1
    cuts = [(a, b) for a, b in cuts if b - a > 0.05]
    pieces, pos, f = [], 0.0, round(0.015 * SR)
    for a, b in cuts + [(len(seg) / SR, len(seg) / SR)]:
        piece = seg[round(pos * SR):round(a * SR)].copy()
        if pieces:
            piece[:f] *= np.linspace(0, 1, f)  # fade in after a cut
        pieces.append(piece)
        pos = b
    for k in range(len(pieces) - 1):  # fade out the end of every piece that is followed by a cut
        pieces[k][-f:] *= np.linspace(1, 0, f)

    def remap(tau: float) -> float:
        return tau - sum(min(max(tau - a, 0.0), b - a) for a, b in cuts)

    return np.concatenate(pieces), remap, sum(b - a for a, b in cuts)


def silence(sec: float) -> np.ndarray:
    return np.zeros(round(sec * SR), dtype=np.float32)


track: list[np.ndarray] = []
t = 0.0


def add(chunk: np.ndarray) -> None:
    global t
    track.append(chunk)
    t += len(chunk) / SR


timeline = {"title": script["title"], "subtitle": script["subtitle"], "voice": "human", "style": STYLE, "scenes": []}
trimmed_total = 0.0
print(f"{'take':5} {'recorded LUFS':>14} {'gain dB':>8}")
for scene in script["scenes"]:
    start = t
    add(silence(max(0.2, script["scene_lead_in"] - PAD_BEFORE)))
    out_sentences = []
    for n, (take, ids) in enumerate((k, v) for k, v in takes.items() if scene_of[v[0]] == scene["id"]):
        x, lufs, gain = prepare(take)
        print(f"{take:5} {lufs:14.1f} {gain:8.1f}")
        spans = align[take]["sentences"]
        a = max(0.0, spans[ids[0]]["start"] - PAD_BEFORE)
        b = min(len(x) / SR, spans[ids[-1]]["end"] + PAD_AFTER)
        seg = x[round(a * SR):round(b * SR)].copy()
        remap = lambda tau: tau
        if STYLE == "upbeat":
            seg, remap, removed = compress_silences(seg)
            trimmed_total += removed
        f = round(FADE * SR)
        seg[:f] *= np.linspace(0, 1, f)
        seg[-f:] *= np.linspace(1, 0, f)
        if n and STYLE != "upbeat":
            add(silence(0.15))  # a little extra air between the concern takes
        seg_start = t
        add(seg)
        for sid in ids:
            out_sentences.append({"id": sid, "caption": sentences[sid].get("caption", sentences[sid]["text"]),
                                  "start": round(seg_start + remap(spans[sid]["start"] - a), 3),
                                  "end": round(seg_start + remap(spans[sid]["end"] - a), 3)})
    hold_start = t
    add(silence(scene.get("hold_after", 0) + max(0.2, scene.get("tail", script["scene_tail"]) - PAD_AFTER)))
    timeline["scenes"].append({"id": scene["id"], "phase": scene["phase"], "start": round(start, 3), "end": round(t, 3),
                               "sentences": out_sentences, "hold": [round(hold_start, 3), round(hold_start + scene.get("hold_after", 0), 3)]})

timeline["duration"] = round(t, 3)
(BUILD / "timeline.json").write_text(json.dumps(timeline, indent=1))
(BUILD / "timeline.js").write_text("window.TIMELINE = " + json.dumps(timeline) + ";\n")

pcm = np.concatenate(track)
print(f"assembled peak {np.abs(pcm).max():.2f}")
sf.write(BUILD / "narration.wav", np.clip(pcm, -1, 1), SR, subtype="PCM_16")


def stamp(x: float) -> str:
    ms = round(x * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


lines, n = [], 0
for scene in timeline["scenes"]:
    for s in scene["sentences"]:
        n += 1
        lines += [str(n), f"{stamp(s['start'])} --> {stamp(s['end'])}", s["caption"], ""]
(BUILD / "captions.srt").write_text("\n".join(lines))

print(f"style {STYLE}" + (f", {trimmed_total:.1f}s of pauses trimmed" if STYLE == "upbeat" else ""))
print(f"duration {t:.1f}s ({int(t // 60)}:{t % 60:04.1f})")
for sc in timeline["scenes"]:
    print(f"  {sc['id']:8} {sc['start']:6.1f} -> {sc['end']:6.1f}  ({sc['end'] - sc['start']:.1f}s)")
