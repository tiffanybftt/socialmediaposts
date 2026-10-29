"""Synthesize an original background bed so there is nothing to license.

usage: python make_music.py <duration_seconds> [calm|upbeat] [timeline.json]   ->  build/music.wav (44.1 kHz stereo)

calm:   soft pad and gentle plucks at 76 BPM
neosoul: 88 BPM laid-back groove: deep round bass, light drums (snare a touch behind the beat), warm electric piano chords
upbeat: 108 BPM groove (kick, clap, hats, bass, plucks) that follows the video: whooshes at scene changes, a
        riser and crash on the closing logo reveal, then a chord that rings out. Needs the timeline to find them.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100
duration = float(sys.argv[1])
style = sys.argv[2] if len(sys.argv) > 2 else "calm"
timeline = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None
n = int(duration * SR)
rng = np.random.default_rng(7)

D, A, B, G = 50, 57, 59, 55
CHORDS = {  # bass root midi, chord tones (semitones above root)
    "D": (D, (0, 4, 7, 12)), "A": (A - 12, (0, 4, 7, 12)), "Bm": (B - 12, (0, 3, 7, 12)), "G": (G - 12, (0, 4, 7, 12)),
}
LOOP = ["D", "A", "Bm", "G"]


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def env(length, attack, release):
    e = np.ones(length)
    a, r = int(attack * SR), int(release * SR)
    e[:a] = np.linspace(0, 1, a) ** 2
    e[-r:] = np.linspace(1, 0, r) ** 2
    return e


def add(out, sig, start, pan=0.0, gain=1.0):
    i = int(round(start * SR))
    if i < 0:
        sig, i = sig[-i:], 0
    if i >= n or len(sig) == 0:
        return
    sig = sig[: n - i]
    out[0, i : i + len(sig)] += sig * gain * (1 - pan) / 2 * 2 ** 0.5 / 1.4
    out[1, i : i + len(sig)] += sig * gain * (1 + pan) / 2 * 2 ** 0.5 / 1.4


def reverb(out, seconds=1.6, decay=3.2, dark=0.0):
    ir_len = int(seconds * SR)
    tail = np.exp(-np.arange(ir_len) / SR * decay)
    wet = np.zeros_like(out)
    for ch in range(2):
        ir = rng.standard_normal(ir_len) * tail
        if dark:  # one-pole low-pass makes the room warmer
            for i in range(1, ir_len):
                ir[i] = (1 - dark) * ir[i] + dark * ir[i - 1]
        size = 1 << int(np.ceil(np.log2(n + ir_len)))
        conv = np.fft.irfft(np.fft.rfft(out[ch], size) * np.fft.rfft(ir, size), size)[:n]
        wet[ch] = conv / (np.abs(conv).max() + 1e-9) * np.abs(out[ch]).max()
    return wet


def finish(mix, fade_in=3.0, fade_out=4.5):
    fi, fo = int(fade_in * SR), int(fade_out * SR)
    mix[:, :fi] *= np.linspace(0, 1, fi)
    mix[:, -fo:] *= np.linspace(1, 0, fo)
    mix = mix / np.abs(mix).max() * 0.85
    path = Path(__file__).resolve().parent.parent / "build" / "music.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix.T * 32767).astype(np.int16).tobytes())
    print("wrote", path, f"{duration:.1f}s ({style})")


def calm():
    BAR = 4 * 60 / 76
    BEAT = 60 / 76
    bars = int(np.ceil(duration / BAR)) + 1
    lift_from = int((duration - 5 * BAR) / BAR)
    seq = [LOOP[i % 4] if i < lift_from else ["G", "A", "D", "D", "D"][min(i - lift_from, 4)] for i in range(bars)]
    out = np.zeros((2, n))
    for b, name in enumerate(seq):
        root, tones = CHORDS[name]
        t0 = b * BAR
        L = int((BAR + 1.4) * SR)
        tt = np.arange(L) / SR
        pad = np.zeros(L)
        for k, semi in enumerate(tones):
            f = hz(root + 12 + semi)
            for det in (-0.004, 0.004):
                pad += np.sin(2 * np.pi * f * (1 + det) * tt + k) * 0.11
        add(out, pad * env(L, 1.0, 1.4), t0)
        bl = int((BAR + 0.4) * SR)
        bt = np.arange(bl) / SR
        add(out, np.sin(2 * np.pi * hz(root) * bt) * 0.20 * env(bl, 0.05, 0.6), t0)
        for step, idx in enumerate([0, 1, 2, 3, 2, 1, 2, 1]):
            f = hz(root + 24 + tones[idx % len(tones)])
            pt = np.arange(int(1.1 * SR)) / SR
            pluck = (np.sin(2 * np.pi * f * pt) + 0.35 * np.sin(2 * np.pi * 2 * f * pt)) * np.exp(-pt * 4.2) * 0.075
            add(out, pluck, t0 + step * BEAT / 2, pan=0.5 if step % 2 else -0.5)
    finish(out * 0.7 + reverb(out) * 0.55)


def upbeat():
    BPM = 108
    BEAT = 60 / BPM
    EIGHTH = BEAT / 2
    BAR = 4 * BEAT

    scenes = {s["id"]: s for s in timeline["scenes"]} if timeline else {}
    logo_t = next((x["start"] for x in scenes.get("close", {"sentences": []})["sentences"] if x["id"] == "c4"), duration - 8.0)
    close_t = scenes["close"]["start"] if "close" in scenes else duration - 25.0
    scene_starts = [s["start"] for s in timeline["scenes"]][1:] if timeline else []

    # bar lines line up with the logo reveal so the crash lands on a downbeat
    L = int(logo_t // BAR)
    g0 = logo_t - L * BAR
    first_bar, last_bar = -1, int((duration - g0) // BAR) + 1
    stop_drums_at = logo_t + 2 * BAR

    def tone(f, length, decay, harmonics, gain):
        tt = np.arange(int(length * SR)) / SR
        sig = sum(h * np.sin(2 * np.pi * f * (k + 1) * tt) for k, h in enumerate(harmonics))
        return sig * np.exp(-tt / decay) * gain

    def band_noise(seconds, lo, hi):
        length = int(seconds * SR)
        spec = np.fft.rfft(rng.standard_normal(length))
        f = np.fft.rfftfreq(length, 1 / SR)
        spec[(f < lo) | (f > hi)] = 0
        x = np.fft.irfft(spec, length)
        return x / np.abs(x).max()

    clap_noise, hat_noise = band_noise(0.3, 1200, 5000), band_noise(0.1, 7000, 16000)
    crash_noise, whoosh_noise = band_noise(3.0, 3000, 16000), band_noise(0.8, 200, 5000)

    def kick():
        tt = np.arange(int(0.35 * SR)) / SR
        phase = 2 * np.pi * np.cumsum(48 + 120 * np.exp(-tt / 0.035)) / SR
        return np.sin(phase) * np.exp(-tt / 0.11) + 0.15 * np.sin(2 * np.pi * 1800 * tt) * np.exp(-tt / 0.004)

    def clap():
        tt = np.arange(int(0.3 * SR)) / SR
        shape = sum((tt >= d) * np.exp(-np.maximum(tt - d, 0) / 0.05) for d in (0, 0.011, 0.022)) / 3
        return clap_noise * shape

    def hat():
        tt = np.arange(int(0.1 * SR)) / SR
        return hat_noise * np.exp(-tt / 0.018)

    def whoosh():
        tt = np.arange(int(0.8 * SR)) / SR
        riser = np.sin(2 * np.pi * np.cumsum(300 + 1700 * (tt / 0.8) ** 2) / SR)
        return (0.5 * riser + 0.6 * whoosh_noise) * np.sin(np.pi * np.clip(tt / 0.8, 0, 1)) ** 2

    dry, wet = np.zeros((2, n)), np.zeros((2, n))  # low end stays dry; everything else gets room
    for i in range(first_bar, last_bar + 1):
        T = g0 + i * BAR
        name = "D" if i >= L + 2 else LOOP[(i - L) % 4]
        root, tones = CHORDS[name]
        drums = T >= 2 * BAR and T < stop_drums_at
        drum_gain = 1.0 if T >= close_t - BAR else 0.8

        Lp = int((BAR + 0.7) * SR)
        tt = np.arange(Lp) / SR
        pad = sum(np.sin(2 * np.pi * hz(root + 12 + s) * (1 + d) * tt + k) * 0.05 for k, s in enumerate(tones) for d in (-0.004, 0.004))
        add(wet, pad * env(Lp, 0.25, 0.5), T)

        for step, idx in enumerate([0, 1, 2, 3, 2, 1, 2, 1]):  # bright plucks on the 8ths
            add(wet, tone(hz(root + 24 + tones[idx % len(tones)]), 0.6, 0.16, (1, 0.5, 0.25), 0.07), T + step * EIGHTH, pan=0.5 if step % 2 else -0.5)

        for step, semis in zip([0, 3, 4, 6], [0, 0, 7, 0]):  # syncopated bass
            b = tone(hz(root + semis), 0.45, 0.2, (1, 0.35), 0.30)
            b[: int(0.006 * SR)] *= np.linspace(0, 1, int(0.006 * SR))
            add(dry, b, T + step * EIGHTH)

        if drums:
            for step in (0, 4):
                add(dry, kick(), T + step * EIGHTH, gain=0.55 * drum_gain)
            for step in (2, 6):
                add(wet, clap(), T + step * EIGHTH, gain=0.30 * drum_gain)
            for step in range(8):
                add(wet, hat(), T + step * EIGHTH, pan=0.3, gain=(0.13 if step % 2 else 0.07) * drum_gain)

    for s in scene_starts:  # a whoosh just ahead of every scene change
        add(wet, whoosh(), s - 0.45, gain=0.16)

    # closing logo reveal: riser into a crash, then a rising chime
    sw = int(1.4 * SR)
    swell = band_noise(1.4, 2500, 15000) * (np.arange(sw) / sw) ** 2.2
    add(wet, swell, logo_t - 1.4, gain=0.22)
    add(wet, crash_noise[: int(2.5 * SR)] * np.exp(-np.arange(int(2.5 * SR)) / SR / 0.7), logo_t, gain=0.30)
    for k, semi in enumerate((74, 78, 81, 86)):
        add(wet, tone(hz(semi), 1.4, 0.8, (1, 0.3), 0.10), logo_t + 0.10 * k, pan=-0.4 + 0.27 * k)

    mix = dry + wet * 0.75 + reverb(wet, 1.4, 3.6) * 0.6
    mix = np.tanh(mix * 1.15) / np.tanh(1.15)  # gentle soft clip keeps the drum peaks in check
    lows = np.abs(np.fft.rfft(mix[0]))[: int(150 * n / SR)].sum() / (np.abs(np.fft.rfft(mix[0])).sum() + 1e-9)
    print(f"bpm {BPM}, bar lines aligned to logo reveal at {logo_t:.1f}s, drums stop at {stop_drums_at:.1f}s, low-end share {lows:.2f}")
    finish(mix, fade_in=1.0, fade_out=3.5)


def neosoul():
    """Laid-back neo-soul: round bass, light drums behind the beat, warm electric piano with lush chords."""
    BPM = 88
    BEAT = 60 / BPM
    SIX = BEAT / 4  # one sixteenth note
    BAR = 4 * BEAT

    scenes = {sc["id"]: sc for sc in timeline["scenes"]} if timeline else {}
    logo_t = next((x["start"] for x in scenes.get("close", {"sentences": []})["sentences"] if x["id"] == "c4"), duration - 8.0)
    scene_starts = [sc["start"] for sc in timeline["scenes"]][1:] if timeline else []

    L = int(logo_t // BAR)  # bar lines line up with the closing logo reveal
    g0 = logo_t - L * BAR
    kL = 4 * L
    tb = lambda k: g0 + k * BEAT

    # bass root (midi), comping notes (a 3rd, 5th, 7th and 9th voicing), bass 5th
    CH = {
        "Dmaj9": (38, (66, 69, 73, 76)), "Bm9": (35, (62, 66, 69, 73)), "Em9": (40, (67, 71, 74, 78)), "A13": (33, (61, 67, 71, 76)),
        "Gmaj9": (43, (59, 62, 66, 69)), "F#m7": (42, (57, 61, 64, 68)),
    }
    FORM = ["Dmaj9", "Bm9", "Em9", "A13", "Gmaj9", "F#m7", "Em9", "A13"]  # I vi ii V, IV iii ii V
    # ending, in beats after the logo bar starts: the last chord lands as late as it can while still ringing for ~3 s
    remaining = duration - logo_t
    FINAL_B = next((b for b in (8, 6, 4) if remaining - b * BEAT >= 2.9), 4)
    END = {8: [(0, 4, "Dmaj9"), (4, 2, "Em9"), (6, 2, "A13")], 6: [(0, 4, "Dmaj9"), (4, 2, "A13")], 4: [(0, 2, "Em9"), (2, 2, "A13")]}[FINAL_B]
    END = END + [(FINAL_B, None, "Dmaj9")]

    def band_noise(seconds, lo, hi):
        length = int(seconds * SR)
        spec = np.fft.rfft(rng.standard_normal(length))
        f = np.fft.rfftfreq(length, 1 / SR)
        spec[(f < lo) | (f > hi)] = 0
        x = np.fft.irfft(spec, length)
        return x / np.abs(x).max()

    snare_noise, hat_noise, crash_noise = band_noise(0.3, 1500, 6500), band_noise(0.3, 6500, 14000), band_noise(3.0, 3000, 16000)

    def keys(midi, seconds, gain):
        """Warm electric piano: a soft FM tine, two slightly detuned voices, a gentle tremolo."""
        tt = np.arange(int(seconds * SR)) / SR
        sig = 0
        for det in (-0.0035, 0.0035):
            f = hz(midi) * (1 + det)
            index = 1.4 * np.exp(-tt / 0.14) + 0.18
            sig = sig + np.sin(2 * np.pi * f * tt + index * np.sin(2 * np.pi * f * tt))
        sig = sig * np.exp(-tt / 1.1) * (1 + 0.1 * np.sin(2 * np.pi * 4.6 * tt))
        a, r = int(0.006 * SR), min(int(0.1 * SR), len(sig) // 2)
        sig[:a] *= np.linspace(0, 1, a)
        sig[-r:] *= np.linspace(1, 0, r)
        return sig * gain

    def bass(midi, seconds, gain):
        """Round electric bass: strong low harmonics so it still reads on laptop speakers."""
        f = hz(midi)
        tt = np.arange(int(seconds * SR)) / SR
        sig = (np.sin(2 * np.pi * f * tt) + 0.8 * np.sin(4 * np.pi * f * tt) + 0.45 * np.sin(6 * np.pi * f * tt)) * np.exp(-tt / 0.7)
        a, r = int(0.012 * SR), min(int(0.08 * SR), len(sig) // 2)
        sig[:a] *= np.linspace(0, 1, a)
        sig[-r:] *= np.linspace(1, 0, r)
        return sig * gain

    def kick():
        tt = np.arange(int(0.3 * SR)) / SR
        return np.sin(2 * np.pi * np.cumsum(50 + 75 * np.exp(-tt / 0.03)) / SR) * np.exp(-tt / 0.13)

    def snare(vel):
        tt = np.arange(int(0.25 * SR)) / SR
        body = np.sin(2 * np.pi * 190 * tt) * np.exp(-tt / 0.045)
        return (body * 0.7 + snare_noise[: len(tt)] * np.exp(-tt / 0.06) * 0.55 + np.sin(2 * np.pi * 1800 * tt) * np.exp(-tt / 0.008) * 0.3) * vel

    def hat(open_=False):
        tt = np.arange(int(0.3 * SR)) / SR
        return hat_noise[: len(tt)] * np.exp(-tt / (0.12 if open_ else 0.02))

    jitter = lambda: float(rng.uniform(-0.006, 0.006))  # small, deterministic human timing

    # groove for each bar: (sixteenth position, length in sixteenths) for the bass; kick positions; keys hits
    BASS = [[(0, 6), (6, 2), (10, 3), (14, 2)], [(0, 5), (7, 2), (11, 2), (14, 2)]]
    KICKS = [(0, 6, 10), (0, 7, 11)]
    KEYS = [[(0, 6), (7, 3), (12, 4)], [(2, 5), (8, 4), (14, 2)]]

    segs = [(k, 4, FORM[((k // 4) - L) % 8], ((k // 4) - L) % 8) for k in range(-4, kL, 4)]
    segs += [(kL + b, ln, ch, None) for b, ln, ch in END]

    dry, wet = np.zeros((2, n)), np.zeros((2, n))
    for i, (k, ln, name, idx) in enumerate(segs):
        root, voicing = CH[name]
        nxt = CH[segs[i + 1][2]][0] if i + 1 < len(segs) else root
        approach = nxt + (2 if root > nxt else -1)
        if ln is None:  # final chord: it just rings out
            add(dry, bass(root, 3.5, 0.17), tb(k))
            for j, m in enumerate(voicing):
                add(wet, keys(m, 4.6, 0.1), tb(k) + 0.014 * j)
            add(dry, kick(), tb(k), gain=0.3)
            add(wet, crash_noise[: int(2.2 * SR)] * np.exp(-np.arange(int(2.2 * SR)) / SR / 0.8), tb(k), gain=0.12)
            for j, m in enumerate((78, 81, 85, 88)):
                add(wet, keys(m, 2.4, 0.05), tb(k) + 0.11 * j, pan=-0.4 + 0.27 * j)
            continue
        pat = (idx if idx is not None else 0) % 2
        lowest = tb(k)
        if ln == 4:
            for pos, length in BASS[pat]:
                m = approach if pos == 14 else (root + 7 if pos == 10 else (root + 12 if pos == 7 else root))
                add(dry, bass(m, length * SIX * 0.95, 0.17), lowest + pos * SIX + jitter())
            for pos, length in KEYS[pat]:
                for j, m in enumerate(voicing):
                    add(wet, keys(m, length * SIX, 0.1), lowest + pos * SIX + 0.012 * j + jitter())
        else:  # two-beat ending chords
            add(dry, bass(root, 1.4 * BEAT, 0.17), lowest)
            add(dry, bass(approach, 0.5 * BEAT, 0.16), lowest + 1.5 * BEAT)
            for j, m in enumerate(voicing):
                add(wet, keys(m, 1.8 * BEAT, 0.1), lowest + 0.012 * j)
        if k >= 8 and k < kL + FINAL_B and ln == 4:  # light drums: soft kick, rim on 2 and 4 a touch late, ghost notes, loose hats
            for pos in KICKS[pat]:
                add(dry, kick(), lowest + pos * SIX + jitter(), gain=0.3)
            for pos, vel in ((4, 0.3), (12, 0.3), (9, 0.07), (15, 0.08)):
                add(wet, snare(vel), lowest + pos * SIX + (0.022 if vel > 0.2 else 0.012) + jitter(), pan=0.1)
            for pos in range(16):
                swing = 0.12 * SIX if pos % 2 else 0.0  # the off-sixteenths lean back a little
                gain = (0.07 if pos % 4 == 0 else 0.045 if pos % 2 == 0 else 0.025)
                add(wet, hat(open_=(pos == 14 and pat == 0)), lowest + pos * SIX + swing + jitter(), pan=-0.3, gain=gain)

    for sct in scene_starts:  # a soft keys note as each scene changes
        add(wet, keys(81, 1.6, 0.05), sct - 0.3, pan=0.2)

    sw = int(1.2 * SR)  # logo reveal: a soft cymbal swell into a crash
    add(wet, band_noise(1.2, 3000, 14000) * (np.arange(sw) / sw) ** 2.2, logo_t - 1.2, gain=0.1)
    add(wet, crash_noise[: int(2.0 * SR)] * np.exp(-np.arange(int(2.0 * SR)) / SR / 0.7), logo_t, gain=0.12)

    mix = dry + wet * 0.8 + reverb(wet, 1.8, 3.0, dark=0.6) * 0.5
    spec = np.fft.rfft(mix, axis=1)  # warm it up: roll off the top end
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= (1 / (1 + (f / 9000) ** 4))[None, :]
    mix = np.fft.irfft(spec, n, axis=1)
    print(f"neo-soul {BPM} BPM, bar lines aligned to the logo reveal at {logo_t:.1f}s, final chord at {tb(kL + FINAL_B):.1f}s")
    finish(mix, fade_in=1.5, fade_out=2.5)


{"calm": calm, "upbeat": upbeat, "neosoul": neosoul}[style]()
