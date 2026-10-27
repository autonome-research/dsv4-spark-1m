#!/usr/bin/env python3
"""GEMV patch accuracy check (optimisation log E07.8 follow-up, 2026-10-24).
The patch (patches/dsv4-attention.py, DSV4_PATCH=1) changes the decode-time compressor GEMV,
so this measures decode output, not prefill. Fixed prompts, greedy (temperature 0, seed 0),
thinking off, top-5 logprobs per generated token.

  run:      gemv-acc.py run <label> [repeats]   -> data/GEMV/<label>-<ts>.json
  compare:  gemv-acc.py compare <a.json> <b.json>

Scores: (1) exact-answer accuracy on 40 arithmetic items and 12 fact items; (2) for each
prompt, whether the greedy token sequences are identical, the first divergence index, and
the mean |logprob difference| of the shared prefix. Run `run` twice on one boot first
(repeats=2): that gives the same-boot noise floor. Cross-boot divergence also happens with an
unchanged config (E12b), so token identity alone cannot prove a difference; accuracy and
logprob deltas against that floor are the evidence."""
import json, random, re, sys, urllib.request
from datetime import datetime
from pathlib import Path

BASE, MODEL = "http://127.0.0.1:8888", "deepseek-v4-flash-0731"
OUT = Path("/srv/dsv4/run/data/GEMV")

def items():
    rng = random.Random(20261024)
    out = []
    for i in range(40):
        a, b = rng.randrange(100, 1000), rng.randrange(11, 100)
        if i % 2:
            out.append((f"What is {a} * {b}? Reply with only the number.", str(a * b)))
        else:
            c = rng.randrange(1000, 10000)
            out.append((f"What is {a} + {b} * {c}? Reply with only the number.", str(a + b * c)))
    facts = [("What is the capital of Australia?", "Canberra"), ("What is the chemical symbol for gold?", "Au"),
             ("How many sides does a hexagon have? Reply with only the number.", "6"),
             ("Who wrote 'Pride and Prejudice'?", "Austen"), ("What is the largest planet in the Solar System?", "Jupiter"),
             ("In which year did the Apollo 11 Moon landing happen? Reply with only the year.", "1969"),
             ("What is the boiling point of water at sea level in Celsius? Reply with only the number.", "100"),
             ("What is the square root of 144? Reply with only the number.", "12"),
             ("Which element has atomic number 1?", "Hydrogen"), ("What is the capital of Canada?", "Ottawa"),
             ("How many minutes are in a day? Reply with only the number.", "1440"),
             ("What language has the most native speakers in the world?", "Mandarin")]
    out += facts
    out.append(("Write a short Python function that returns the n-th Fibonacci number.", None))
    out.append(("Explain in three sentences why the sky is blue.", None))
    return out

def ask(prompt):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 128,
               "temperature": 0.0, "seed": 0, "logprobs": True, "top_logprobs": 5,
               "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        c = json.load(r)["choices"][0]
    lp = [[t["token"], t["logprob"]] for t in (c.get("logprobs") or {}).get("content") or []]
    return c["message"]["content"] or "", lp

def correct(answer, expected):
    if expected is None: return None
    if expected.isdigit():
        return expected in re.findall(r"\d+", answer.replace(",", ""))
    return expected.lower() in answer.lower()

def run(label, repeats):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"label": label, "started": datetime.now().isoformat(timespec="seconds"), "passes": []}
    for rep in range(repeats):
        rows = []
        for prompt, exp in items():
            text, lp = ask(prompt)
            rows.append({"prompt": prompt, "expected": exp, "text": text, "correct": correct(text, exp), "logprobs": lp})
        scored = [r for r in rows if r["correct"] is not None]
        acc = sum(r["correct"] for r in scored) / len(scored)
        print(f"pass {rep + 1}: accuracy {sum(r['correct'] for r in scored)}/{len(scored)} = {acc:.3f}", flush=True)
        res["passes"].append(rows)
    path = OUT / f"{label}-{datetime.now():%Y%m%dT%H%M%S}.json"
    path.write_text(json.dumps(res))
    print(path)

def compare_rows(ra, rb):
    same = div = 0; firsts = []; deltas = []
    for x, y in zip(ra, rb):
        ta, tb = [t for t, _ in x["logprobs"]], [t for t, _ in y["logprobs"]]
        k = next((i for i, (p, q) in enumerate(zip(ta, tb)) if p != q), min(len(ta), len(tb)))
        if ta == tb: same += 1
        else: div += 1; firsts.append(k)
        deltas += [abs(x["logprobs"][i][1] - y["logprobs"][i][1]) for i in range(k)]
    return same, div, firsts, (sum(deltas) / len(deltas) if deltas else 0.0), (max(deltas) if deltas else 0.0)

def summary(name, rows):
    s = [r for r in rows if r["correct"] is not None]
    return f"{name}: accuracy {sum(r['correct'] for r in s)}/{len(s)}"

def compare(fa, fb):
    a, b = json.loads(Path(fa).read_text()), json.loads(Path(fb).read_text())
    for i, p in enumerate(a["passes"]): print(summary(f"A[{a['label']}] pass {i + 1}", p))
    for i, p in enumerate(b["passes"]): print(summary(f"B[{b['label']}] pass {i + 1}", p))
    if len(a["passes"]) > 1:
        s, d, f, m, mx = compare_rows(a["passes"][0], a["passes"][1])
        print(f"noise floor A pass1 vs pass2: identical {s}/{s + d}, first divergences {sorted(f)}, mean |dlogprob| {m:.5f} (max {mx:.4f})")
    s, d, f, m, mx = compare_rows(a["passes"][0], b["passes"][0])
    print(f"A vs B (pass 1): identical {s}/{s + d}, first divergences {sorted(f)}, mean |dlogprob| {m:.5f} (max {mx:.4f})")

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run": run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    elif len(sys.argv) == 4 and sys.argv[1] == "compare": compare(sys.argv[2], sys.argv[3])
    else: sys.exit(__doc__)
