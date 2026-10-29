"""Find where each script sentence starts and ends inside a human recording.

Takes the speech-to-text word timings (rec/asr.json: {take: {dur, words:[{w,s,e}]}}) and aligns the
words of the script sentences for that take against them (Needleman-Wunsch on normalised words).

usage: python align_takes.py <rec_dir>            -> writes <rec_dir>/align.json and prints a report
"""
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
rec = Path(sys.argv[1])
script = json.loads((ROOT / "episode" / "script.json").read_text())
takes = json.loads((ROOT / "episode" / "takes.json").read_text())
asr = json.loads((rec / "asr.json").read_text())
sentences = {s["id"]: s for sc in script["scenes"] for s in sc["sentences"]}


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def tokens(text: str) -> list[str]:
    return [t for t in (norm(w) for w in re.split(r"[\s-]+", text)) if t]  # "spot-check" is two spoken words


def similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def align(script_words, asr_words):
    """Global alignment; returns for each script word the index of its ASR word (or None)."""
    n, m = len(script_words), len(asr_words)
    GAP, MIS = -1.0, -1.0
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * GAP
    for j in range(1, m + 1):
        score[0][j] = j * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = similar(script_words[i - 1], asr_words[j - 1])
            diag = score[i - 1][j - 1] + (2.0 * sim if sim >= 0.75 else MIS)
            score[i][j] = max(diag, score[i - 1][j] + GAP, score[i][j - 1] + GAP)
    i, j, match = n, m, [None] * n
    while i > 0 and j > 0:
        sim = similar(script_words[i - 1], asr_words[j - 1])
        diag = score[i - 1][j - 1] + (2.0 * sim if sim >= 0.75 else MIS)
        if score[i][j] == diag:
            if sim >= 0.75:
                match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif score[i][j] == score[i - 1][j] + GAP:
            i -= 1
        else:
            j -= 1
    return match


def loudness_track(path: Path):
    """10 ms loudness curve of the raw take, plus a speech/silence threshold halfway up from the noise floor."""
    x, sr = sf.read(path, dtype="float32")
    n = sr // 100
    fr = x[: len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    padded = np.pad(db, 2, mode="edge")
    db = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)  # 50 ms median: ignores flicker
    noise, speech = np.percentile(db, 8), np.percentile(db, 90)
    return db, noise + 0.35 * (speech - noise), noise + 0.55 * (speech - noise)  # onset level, speech level


def onsets(db, thr_on, thr, lo, hi):
    """Times in [lo, hi] where a word starts: 60 ms of quiet, then 50 ms above the onset level that grows into speech."""
    return [i / 100 for i in range(max(6, int(lo * 100)), min(int(hi * 100), len(db) - 30))
            if all(db[i:i + 5] >= thr_on) and max(db[i:i + 30]) >= thr and all(db[i - 6:i] < thr_on)]


def snap_start(db, thr_on, thr, t0, prev_end):
    """Speech-to-text can place a sentence start early (after silence) or late (mid-flow). Find the real onset."""
    lo = max(0.0, t0 - 0.6, (prev_end - 0.1) if prev_end is not None else 0.0)
    back = onsets(db, thr_on, thr, lo, t0 + 0.05)  # voice began at or before the speech-to-text time
    if back:
        return max(0.0, back[0] - 0.02)
    ahead = onsets(db, thr_on, thr, t0 + 0.05, t0 + 1.2)  # still silent at t0: the voice begins a little later
    if ahead and db[int(t0 * 100)] < thr_on:
        return max(0.0, ahead[0] - 0.02)
    return t0


def snap_end(db, thr, t1, next_start):
    hi = t1 + 0.4 if next_start is None else min(t1 + 0.4, next_start - 0.04)
    for i in range(min(int(hi * 100), len(db) - 1), int(max(3, (t1 - 1.0) * 100)), -1):
        if all(db[i - 2:i + 1] >= thr):
            return i / 100 + 0.12
    return t1


result, problems = {}, []
for take, ids in takes.items():
    db, thr_on, thr = loudness_track(rec / f"take_{take}.wav")
    words = asr[take]["words"]
    asr_tok = [norm(w["w"]) for w in words]
    keep = [k for k, t in enumerate(asr_tok) if t]
    asr_tok = [asr_tok[k] for k in keep]
    words = [words[k] for k in keep]
    flat, owner = [], []
    for sid in ids:
        toks = tokens(sentences[sid].get("caption", sentences[sid]["text"]))
        flat += toks
        owner += [sid] * len(toks)
    match = align(flat, asr_tok)
    extra = len(asr_tok) - sum(m is not None for m in match)
    result[take] = {"duration": asr[take]["dur"], "sentences": {}}
    rows = []
    for sid in ids:
        idx = [match[k] for k, o in enumerate(owner) if o == sid and match[k] is not None]
        total = owner.count(sid)
        if not idx:
            problems.append(f"take {take}: sentence {sid} not found in the recording")
            continue
        rows.append((sid, words[min(idx)]["s"], words[max(idx)]["e"], len(idx), total))
        if len(idx) / total < 0.85:
            problems.append(f"take {take}: sentence {sid} only {len(idx)}/{total} words matched: "
                            f"heard '{' '.join(w['w'] for w in words[min(idx):max(idx) + 1])}'")
    starts = [snap_start(db, thr_on, thr, r[1], rows[k - 1][2] if k else None) for k, r in enumerate(rows)]
    for k, (sid, asr_start, asr_end, matched, total) in enumerate(rows):
        start = starts[k]
        end = snap_end(db, thr, asr_end, starts[k + 1] if k + 1 < len(rows) else None)
        if end <= start:
            start, end = asr_start, asr_end
        result[take]["sentences"][sid] = {"start": start, "end": end, "matched": matched, "words": total,
                                          "asr_start": asr_start, "asr_end": asr_end}
        if abs(start - asr_start) > 0.15:
            print(f"  take {take} {sid}: start moved {start - asr_start:+.2f}s to where the voice actually begins")
    matched_idx = {m for m in match if m is not None}
    leftovers = [f"'{w['w']}'@{w['s']:.1f}s" for k, w in enumerate(words) if k not in matched_idx]
    result[take]["extra_words"] = extra
    if extra:
        print(f"  take {take}: words heard that are not in the script: {', '.join(leftovers)}")
    if extra > 3:
        problems.append(f"take {take}: {extra} spoken words are not in the script (repeats, ad-libs or mis-hearings)")

(rec / "align.json").write_text(json.dumps(result, indent=1))
for take, r in result.items():
    spans = "  ".join(f"{sid}:{v['start']:.1f}-{v['end']:.1f}" for sid, v in r["sentences"].items())
    print(f"take {take:3} ({r['duration']:.1f}s)  {spans}")
print("\n" + ("PROBLEMS:\n  " + "\n  ".join(problems) if problems else "No problems: every sentence found, nothing extra."))
