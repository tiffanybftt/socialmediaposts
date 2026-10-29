"""Turn recorded takes into clean WAVs plus word-level speech-to-text timings.

usage: python transcribe_takes.py <folder with *Take_<id>.m4a> <rec_dir> [model_dir]

For every file named like "...Take_6a.m4a" this writes <rec_dir>/take_6a.wav (48 kHz mono, as recorded)
and, for all takes together, <rec_dir>/asr.json. Everything runs locally; no audio leaves this machine.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

src, rec = Path(sys.argv[1]), Path(sys.argv[2])
model_dir = sys.argv[3] if len(sys.argv) > 3 else str(rec.parent / "whisper")
rec.mkdir(parents=True, exist_ok=True)

files = {}
for f in sorted(src.glob("*Take_*.m4a")):
    files[re.search(r"Take_([0-9a-z]+)\.m4a$", f.name).group(1)] = f  # a later upload of the same take wins

model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4, download_root=model_dir)
prompt = "Claude, Anthropic, API, ID, Gmail, Drive. Using Claude with confidence: privacy, security, accuracy, authenticity."
out = {}
for take, f in files.items():
    wav = rec / f"take_{take}.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(f), "-ac", "1", "-ar", "48000", "-c:a", "pcm_f32le", str(wav)], check=True)
    asr_wav = rec / f"asr_{take}.wav"  # quieter 16 kHz copy so clipped takes transcribe cleanly
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-ar", "16000", "-af", "volume=-12dB", str(asr_wav)], check=True)
    audio, sr = sf.read(asr_wav, dtype="float32")
    segs, _ = model.transcribe(audio, language="en", word_timestamps=True, beam_size=5, initial_prompt=prompt, condition_on_previous_text=False)
    words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)} for sg in segs for w in sg.words]
    out[take] = {"dur": round(len(audio) / sr, 2), "words": words}
    print(f"take {take}: {len(audio) / sr:.1f}s, {len(words)} words", flush=True)
(rec / "asr.json").write_text(json.dumps(out))
