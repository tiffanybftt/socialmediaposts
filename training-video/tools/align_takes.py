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


result, problems = {}, []
for take, ids in takes.items():
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
    for sid in ids:
        idx = [match[k] for k, o in enumerate(owner) if o == sid and match[k] is not None]
        total = owner.count(sid)
        if not idx:
            problems.append(f"take {take}: sentence {sid} not found in the recording")
            continue
        start, end = words[min(idx)]["s"], words[max(idx)]["e"]
        result[take]["sentences"][sid] = {"start": start, "end": end, "matched": len(idx), "words": total}
        if len(idx) / total < 0.85:
            problems.append(f"take {take}: sentence {sid} only {len(idx)}/{total} words matched: "
                            f"heard '{' '.join(w['w'] for w in words[min(idx):max(idx) + 1])}'")
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
