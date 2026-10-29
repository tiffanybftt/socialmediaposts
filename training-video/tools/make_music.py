"""Synthesize a soft, original background bed (pad + gentle plucks) so there is nothing to license.

usage: python make_music.py <duration_seconds>   ->  build/music.wav (44.1 kHz stereo)
"""
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100
BPM = 76
BEAT = 60 / BPM
BAR = 4 * BEAT
duration = float(sys.argv[1])
n = int(duration * SR)
rng = np.random.default_rng(7)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# D major feel: D - A - Bm - G, then a lifted G - A - D - D for the last bars
D, A, B, G = 50, 57, 59, 55
CHORDS = {  # root midi, chord tones (semitones above root)
    "D": (D, (0, 4, 7, 12)), "A": (A - 12, (0, 4, 7, 12)), "Bm": (B - 12, (0, 3, 7, 12)), "G": (G - 12, (0, 4, 7, 12)),
}
loop = ["D", "A", "Bm", "G"]
bars = int(np.ceil(duration / BAR)) + 1
lift_from = int((duration - 5 * BAR) / BAR)
seq = [loop[i % 4] if i < lift_from else ["G", "A", "D", "D", "D"][min(i - lift_from, 4)] for i in range(bars)]

out = np.zeros((2, n))


def add(sig, start, pan=0.0):
    i = int(start * SR)
    if i >= n:
        return
    sig = sig[: n - i]
    out[0, i : i + len(sig)] += sig * (1 - pan) / 2 * 2 ** 0.5 / 1.4
    out[1, i : i + len(sig)] += sig * (1 + pan) / 2 * 2 ** 0.5 / 1.4


def env(length, attack, release):
    e = np.ones(length)
    a, r = int(attack * SR), int(release * SR)
    e[:a] = np.linspace(0, 1, a) ** 2
    e[-r:] = np.linspace(1, 0, r) ** 2
    return e


for b, name in enumerate(seq):
    root, tones = CHORDS[name]
    t0 = b * BAR
    L = int((BAR + 1.4) * SR)
    tt = np.arange(L) / SR
    # pad: detuned sine pairs, one voice per chord tone
    pad = np.zeros(L)
    for k, semi in enumerate(tones):
        f = hz(root + 12 + semi)
        for det in (-0.004, 0.004):
            pad += np.sin(2 * np.pi * f * (1 + det) * tt + k) * 0.11
    add(pad * env(L, 1.0, 1.4), t0)
    # bass: root, soft
    bl = int((BAR + 0.4) * SR)
    bt = np.arange(bl) / SR
    add(np.sin(2 * np.pi * hz(root) * bt) * 0.20 * env(bl, 0.05, 0.6), t0)
    # plucks: 8th-note arpeggio, up and back down
    pattern = [0, 1, 2, 3, 2, 1, 2, 1]
    for step, idx in enumerate(pattern):
        f = hz(root + 24 + tones[idx % len(tones)])
        pl = int(1.1 * SR)
        pt = np.arange(pl) / SR
        pluck = (np.sin(2 * np.pi * f * pt) + 0.35 * np.sin(2 * np.pi * 2 * f * pt)) * np.exp(-pt * 4.2) * 0.075
        add(pluck, t0 + step * BEAT / 2, pan=0.5 if step % 2 else -0.5)

# gentle reverb: decaying noise impulse response, FFT convolution
ir_len = int(1.6 * SR)
tail = np.exp(-np.arange(ir_len) / SR * 3.2)
wet = np.zeros_like(out)
for ch in range(2):
    ir = rng.standard_normal(ir_len) * tail
    size = 1 << int(np.ceil(np.log2(n + ir_len)))
    conv = np.fft.irfft(np.fft.rfft(out[ch], size) * np.fft.rfft(ir, size), size)[:n]
    wet[ch] = conv / np.abs(conv).max() * np.abs(out[ch]).max()
mix = out * 0.7 + wet * 0.55

# fade in/out and normalise
fade_in, fade_out = int(3 * SR), int(4.5 * SR)
mix[:, :fade_in] *= np.linspace(0, 1, fade_in)
mix[:, -fade_out:] *= np.linspace(1, 0, fade_out)
mix = mix / np.abs(mix).max() * 0.85

path = Path(__file__).resolve().parent.parent / "build" / "music.wav"
with wave.open(str(path), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix.T * 32767).astype(np.int16).tobytes())
print("wrote", path, f"{duration:.1f}s")
