#!/usr/bin/env python3
"""E12b serving check: a deterministic long prompt with a hidden fact, greedy decoding.
Same prompt every run for a given context, so outputs with the split indexer on and
off can be compared token for token. Usage: e12b_client.py <ctx> <label>"""
import json, random, sys, time, urllib.request
BASE, MODEL = "http://127.0.0.1:8888", "deepseek-v4-flash-0731"
def post(path, payload, timeout=600):
    r = urllib.request.Request(BASE+path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
target, label = int(sys.argv[1]), sys.argv[2]
rng = random.Random(target)        # fixed per context length
words = "river stone lantern orbit velvet copper meadow signal harbor ember quartz falcon thicket prism glacier tundra compass saffron basalt cinder".split()
def filler(n):
    return " ".join(rng.choice(words) + (f" ({i})." if i % 17 == 16 else "") for i in range(n))
per_word = post("/tokenize", {"model": MODEL, "prompt": filler(4000)})["count"] / 4000
n_words = int((target - 400) / per_word)
rng = random.Random(target)
a, b = filler(int(n_words * 0.37)), filler(n_words - int(n_words * 0.37))
code = f"{rng.randrange(10**7, 10**8)}"
prompt = (f"Below is a long log of random words. Somewhere in it is one sentence with a code.\n\n{a}\n\n"
          f"The vault code is {code}.\n\n{b}\n\n"
          "Question: what is the vault code? Answer with the number first, then describe in two sentences where in the log it appeared.")
n_prompt = post("/tokenize", {"model": MODEL, "prompt": prompt})["count"]
# The word counter in the filler grows longer with depth, so near 1M the estimate
# overshoots (E12b: 1,058,675 tokens for target 1,036,288, HTTP 400). Shrink only
# when over the target by > 1 %, so the 64K / 256K prompts stay as before.
for _ in range(3):
    if n_prompt <= target * 1.01: break
    n_words = int(n_words * target / n_prompt)
    rng = random.Random(target)
    a, b = filler(int(n_words * 0.37)), filler(n_words - int(n_words * 0.37))
    code = f"{rng.randrange(10**7, 10**8)}"
    prompt = (f"Below is a long log of random words. Somewhere in it is one sentence with a code.\n\n{a}\n\n"
              f"The vault code is {code}.\n\n{b}\n\n"
              "Question: what is the vault code? Answer with the number first, then describe in two sentences where in the log it appeared.")
    n_prompt = post("/tokenize", {"model": MODEL, "prompt": prompt})["count"]
payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64,
           "temperature": 0.0, "seed": 0, "stream": True, "stream_options": {"include_usage": True},
           "chat_template_kwargs": {"thinking": False}, "logprobs": True}
req = urllib.request.Request(BASE+"/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); t_first = t_last = None; text = []; toks = []; usage = None
with urllib.request.urlopen(req, timeout=7200) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d = json.loads(line[5:])
        if d.get("usage"): usage = d["usage"]
        for c in d.get("choices", []):
            piece = c.get("delta", {}).get("content")
            if piece:
                t_last = time.time(); t_first = t_first or t_last; text.append(piece)
            for lp in ((c.get("logprobs") or {}).get("content") or []):
                toks.append([lp["token"], round(lp["logprob"], 4)])
out = "".join(text)
ct = (usage or {}).get("completion_tokens", 0)
print(json.dumps({"label": label, "target": target, "prompt_tokens": (usage or {}).get("prompt_tokens", n_prompt),
                  "ttft_s": round(t_first - t0, 1), "prefill_tok_s": round(n_prompt / (t_first - t0)),
                  "decode_tok_s": round((ct - 1) / (t_last - t_first), 2) if ct > 1 and t_last > t_first else None,
                  "code": code, "code_found": code in out, "text": out, "tokens": toks}))
