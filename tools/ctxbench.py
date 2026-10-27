#!/usr/bin/env python3
"""Measure prefill and decode speed at a target context length (the project scoreboard).

Usage: ctxbench.py <context_tokens> [max_new_tokens]      default 256 new tokens

Builds a filler prompt of about <context_tokens> tokens with a unique random prefix, so
the prefix cache never hits, then streams one completion with ignore_eos, so the output
length is fixed. Sampling uses temperature 0.6 (not greedy), as in every run in the log;
the sampled tokens do not change the per-token work, so speed is unaffected.
  TTFT            time to the first streamed token (prefill time)
  decode tok/s    (completion_tokens - 1) / (t_last - t_first); speculative decoding off
Prints one JSON line with prompt_tokens, ttft_s, prefill_tok_s and decode_tok_s.
Normally run through run-guarded.sh, which adds the memory watchdog."""
import json, random, sys, time, urllib.request, uuid
BASE, MODEL = "http://127.0.0.1:8888", "deepseek-v4-flash-0731"
def post(path, payload, timeout=60):
    r = urllib.request.Request(BASE+path, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
rng = random.Random(1234)
words = "river stone lantern orbit velvet copper meadow signal harbor ember quartz falcon thicket prism glacier tundra compass saffron basalt cinder".split()
def filler(n_words):
    out=[]
    for i in range(n_words):
        out.append(rng.choice(words))
        if i % 17 == 16: out.append(f"({i}).")
    return " ".join(out)
def build(target):
    probe = filler(4000); t = post("/tokenize", {"model":MODEL,"prompt":probe})["count"]
    per_word = t/4000
    body = filler(int((target-300)/per_word))
    tag = uuid.uuid4().hex
    return (f"Session {tag}. Below is a long log of random words.\n\n{body}\n\n"
            "Task: ignore the log above. Write a detailed, multi-paragraph essay about the history of lighthouses.")
target = int(sys.argv[1]); max_new = int(sys.argv[2]) if len(sys.argv)>2 else 256
prompt = build(target)
payload = {"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_new,"temperature":0.6,
           "ignore_eos":True,"stream":True,"stream_options":{"include_usage":True},
           "chat_template_kwargs":{"thinking":False}}
req = urllib.request.Request(BASE+"/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
t0=time.time(); t_first=None; t_last=None; usage=None; chunks=0
with urllib.request.urlopen(req, timeout=7200) as r:
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d=json.loads(line[5:])
        if d.get("usage"): usage=d["usage"]
        for c in d.get("choices",[]):
            delta=c.get("delta",{})
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                now=time.time(); t_first = t_first or now; t_last=now; chunks+=1
ct=usage["completion_tokens"]; pt=usage["prompt_tokens"]
ttft=t_first-t0; dec=(ct-1)/(t_last-t_first) if t_last>t_first else float("nan")
res={"target":target,"prompt_tokens":pt,"ttft_s":round(ttft,1),"prefill_tok_s":round(pt/ttft),"completion_tokens":ct,"decode_tok_s":round(dec,2),"chunks":chunks}
print(json.dumps(res)); sys.stdout.flush()
