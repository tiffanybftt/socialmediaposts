"""Synthesize an original background bed so there is nothing to license.

usage: python make_music.py <duration_seconds> [calm|upbeat|neosoul] [timeline.json] [soundfont.sf2]   ->  build/music.wav (44.1 kHz stereo)

calm:   soft pad and gentle plucks at 76 BPM
neosoul: 88 BPM laid-back groove with real instruments: sampled grand piano holding soft chords, a sampled drum kit,
        and a round bass that drives the melody (needs tinysoundfont and a General MIDI SoundFont)
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
timeline = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 and sys.argv[3] not in ("", "-") else None
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
    """Laid-back neo-soul with real instruments: a sampled grand piano holding soft chords, a sampled drum kit,
    and a round synth bass that drives the groove and carries the melody.

    Needs `tinysoundfont` and a General MIDI SoundFont (e.g. FluidR3_GM.sf2, MIT licensed). Pass its path as the
    4th argument or set the SOUNDFONT environment variable.
    """
    import os

    import tinysoundfont

    sf_path = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("SOUNDFONT", "")
    if not Path(sf_path).is_file():
        raise SystemExit("neosoul needs a General MIDI SoundFont: pass its path as the 4th argument or set SOUNDFONT")

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

    # bass root (midi), piano voicing (soft, no root: the bass has it), chord quality
    CH = {
        "Dmaj9": (38, (57, 61, 64, 66), "maj"), "Bm9": (35, (57, 61, 62, 66), "min"), "Em9": (40, (55, 59, 62, 66), "min"),
        "A13": (33, (55, 59, 61, 66), "dom"), "Gmaj9": (43, (59, 62, 66, 69), "maj"), "F#m7": (42, (57, 61, 64, 68), "min"),
    }
    INTERVAL = {"maj": {"r": 0, "3": 4, "5": 7, "7": 11, "o": 12}, "min": {"r": 0, "3": 3, "5": 7, "7": 10, "o": 12},
                "dom": {"r": 0, "3": 4, "5": 7, "7": 10, "o": 12}}
    FORM = ["Dmaj9", "Bm9", "Em9", "A13", "Gmaj9", "F#m7", "Em9", "A13"]  # I vi ii V, IV iii ii V
    # ending, in beats after the logo bar starts: the last chord lands as late as it can while still ringing for ~3.6 s
    remaining = duration - logo_t
    FINAL_B = next((b for b in (8, 6, 4) if remaining - b * BEAT >= 3.6), 4)
    END = {8: [(0, 4, "Dmaj9"), (4, 2, "Em9"), (6, 2, "A13")], 6: [(0, 4, "Dmaj9"), (4, 2, "A13")], 4: [(0, 2, "Em9"), (2, 2, "A13")]}[FINAL_B]
    END = END + [(FINAL_B, None, "Dmaj9")]

    # bass line, per bar: (sixteenth position, length in sixteenths, chord tone). "ap" leads into the next chord.
    BASS_T = [
        [(0, 3, "r"), (3, 1, "o"), (4, 2, "5"), (6, 1, "3"), (8, 3, "r"), (11, 1, "7"), (12, 2, "5"), (14, 2, "ap")],
        [(0, 2, "r"), (2, 2, "5"), (4, 1, "o"), (5, 1, "7"), (6, 2, "5"), (8, 2, "3"), (10, 2, "r"), (12, 1, "5"), (13, 1, "3"), (14, 2, "ap")],
    ]

    def bass(midi, seconds, gain):
        """Round electric bass: strong low harmonics so it still reads on laptop speakers."""
        f = hz(midi)
        tt = np.arange(int(seconds * SR)) / SR
        sig = (np.sin(2 * np.pi * f * tt) + 0.8 * np.sin(4 * np.pi * f * tt) + 0.45 * np.sin(6 * np.pi * f * tt)) * np.exp(-tt / 0.7)
        a, r = int(0.012 * SR), min(int(0.08 * SR), len(sig) // 2)
        sig[:a] *= np.linspace(0, 1, a)
        sig[-r:] *= np.linspace(1, 0, r)
        return sig * gain

    class Sf:
        """A tiny sequencer for the SoundFont synth: collect timed events, then render them to audio."""

        def __init__(self, setup):
            self.sy = tinysoundfont.Synth(samplerate=SR)
            sfid = self.sy.sfload(sf_path)
            for chan, bank, preset, drums in setup:
                self.sy.program_select(chan, sfid, bank, preset, drums)
            self.ev = []

        def on(self, t, chan, key, vel):
            self.ev.append((t, 2, chan, key, int(np.clip(vel, 1, 127))))

        def off(self, t, chan, key):
            self.ev.append((t, 0, chan, key, 0))

        def pedal(self, t, down):
            self.ev.append((t, 1, 0, 64, 127 if down else 0))

        def render(self):
            out, pos = np.zeros((n, 2), np.float32), 0
            for t, kind, chan, a, b in sorted(self.ev, key=lambda e: (e[0], e[1])):
                idx = min(n, max(pos, int(round(t * SR))))
                if idx > pos:
                    out[pos:idx] = np.frombuffer(self.sy.generate(idx - pos), dtype=np.float32).reshape(-1, 2)
                    pos = idx
                if kind == 2:
                    self.sy.noteon(chan, a, b)
                elif kind == 0:
                    self.sy.noteoff(chan, a)
                else:
                    self.sy.control_change(chan, a, b)
            if pos < n:
                out[pos:] = np.frombuffer(self.sy.generate(n - pos), dtype=np.float32).reshape(-1, 2)
            return out.T.astype(np.float64)

    piano, kit, fx = Sf([(0, 0, 0, False)]), Sf([(9, 128, 8, True)]), Sf([(3, 0, 11, False)])  # grand piano, room kit, vibraphone
    KICK, SNARE, HAT, CRASH = 36, 38, 42, 49
    jitter = lambda: float(rng.uniform(-0.006, 0.006))  # small, deterministic human timing
    vary = lambda base: base + int(rng.integers(-5, 6))

    segs = [(k, 4, FORM[((k // 4) - L) % 8], ((k // 4) - L) % 8) for k in range(-4, kL, 4)]
    segs += [(kL + b, ln, ch, None) for b, ln, ch in END]

    dry = np.zeros((2, n))  # the bass
    for i, (k, ln, name, idx) in enumerate(segs):
        root, voicing, kind = CH[name]
        iv = INTERVAL[kind]
        nxt = CH[segs[i + 1][2]][0] if i + 1 < len(segs) else root
        approach = nxt + (2 if root > nxt else -1)
        t0 = tb(k)
        if ln is None:  # final chord: it just rings out
            add(dry, bass(root, 3.5, 1.3), t0)
            piano.pedal(t0 - 0.03, True)
            for j, m in enumerate(voicing):
                piano.on(t0 + 0.014 * j, 0, m, 82 - 2 * j)
                piano.off(t0 + 4.4, 0, m)
            kit.on(t0, 9, KICK, 76)
            kit.on(t0, 9, CRASH, 64)
            for j, m in enumerate((78, 81, 85, 88)):
                fx.on(t0 + 0.12 * j, 3, m, 64)
                fx.off(t0 + 2.4, 3, m)
            continue

        # bass: the driving, melodic line
        template = BASS_T[(idx if idx is not None else 0) % 2] if ln == 4 else [(0, 6, "r"), (6, 1, "ap")]
        for pos, length, tone in template:
            midi = approach if tone == "ap" else root + iv[tone]
            add(dry, bass(midi, max(length * SIX * 0.95, 0.12), 1.0), t0 + pos * SIX + jitter())

        # piano: one soft chord a bar, held with the pedal
        hold = ln * BEAT - 0.5 * BEAT
        piano.pedal(t0 - 0.03, True)
        for j, m in enumerate(voicing):
            piano.on(t0 + 0.012 * j + max(0.0, jitter()), 0, m, vary(54) - 2 * j)
            piano.off(t0 + hold, 0, m)
        if ln == 4 and (idx if idx is not None else 0) % 2 == 0:  # a quiet re-voicing on beat 3
            for j, m in enumerate(voicing[1:]):
                piano.on(t0 + 2 * BEAT + 0.012 * j, 0, m, vary(38))
                piano.off(t0 + 3.4 * BEAT, 0, m)
        piano.pedal(t0 + ln * BEAT - 0.09, False)

        # drums: solid kick, snare on 2 and 4 a touch behind the beat, ghost notes, steady hats
        if k >= 8 and k < kL + FINAL_B and ln == 4:
            pat = (idx if idx is not None else 0) % 2
            for pos in ((0, 6, 10) if pat == 0 else (0, 7, 11)):
                kit.on(t0 + pos * SIX + jitter(), 9, KICK, vary(96 if pos == 0 else 84))
            fill = idx == 7
            for pos, vel in ((4, 84), (12, 84 if not fill else 0), (9, 30), (15, 30 if not fill else 0)):
                if vel:
                    kit.on(t0 + pos * SIX + (0.018 if vel > 60 else 0.01) + jitter(), 9, SNARE, vary(vel))
            if fill:  # a small build into the next chord
                for pos, vel in ((12, 56), (13, 66), (14, 78), (15, 92)):
                    kit.on(t0 + pos * SIX + jitter(), 9, SNARE, vary(vel))
            for pos in range(16):
                vel = 62 if pos % 4 == 2 else 54 if pos % 4 == 0 else 26
                kit.on(t0 + pos * SIX + (0.1 * SIX if pos % 2 else 0.0) + jitter(), 9, HAT, vary(vel))

    for sct in scene_starts:  # a soft vibraphone note as each scene changes
        fx.on(sct - 0.3, 3, 81, 50)
        fx.off(sct + 1.1, 3, 81)

    kit.on(logo_t, 9, CRASH, 88)  # logo reveal: a cymbal swell into a crash

    def noise_band(seconds, lo, hi):
        length = int(seconds * SR)
        spec = np.fft.rfft(rng.standard_normal(length))
        f = np.fft.rfftfreq(length, 1 / SR)
        spec[(f < lo) | (f > hi)] = 0
        x = np.fft.irfft(spec, length)
        return x / np.abs(x).max()

    swell = np.zeros((2, n))
    sw = int(1.2 * SR)
    add(swell, noise_band(1.2, 3000, 14000) * (np.arange(sw) / sw) ** 2.2, logo_t - 1.2, gain=0.2)

    def level(x, target_db):
        """Bring a stem to a target RMS, measured on the steady middle of the track (or all of it if that part is silent)."""
        lo, hi = int(10 * SR), int(min(n - 5 * SR, 45 * SR))
        window = x[:, lo:hi]
        if np.sqrt((window ** 2).mean()) < 1e-6:
            window = x
        rms = np.sqrt((window ** 2).mean())
        return x * 10 ** (target_db / 20) / rms if rms > 1e-9 else x

    bass_t, drums_t, piano_t, fx_t = dry, kit.render(), piano.render(), fx.render()
    mix = (level(bass_t, -24) + level(drums_t, -23) + level(piano_t, -27) + level(fx_t, -36) +
           reverb(level(piano_t, -27) + level(fx_t, -36), 1.8, 3.0, dark=0.4) * 0.35 + swell * 0.04)
    print(f"neo-soul {BPM} BPM, bar lines aligned to the logo reveal at {logo_t:.1f}s, final chord at {tb(kL + FINAL_B):.1f}s")
    finish(mix, fade_in=1.5, fade_out=1.6)


{"calm": calm, "upbeat": upbeat, "neosoul": neosoul}[style]()
