"""Generate question/answer candidates from cleaned chunks with Groq and Gemini.

- Both providers use the OpenAI-compatible chat endpoint (plain `requests`).
- Keys and model names come from .env (see .env.example).
- Resumable: finished chunks are recorded by a hash of their content (not their
  position), so you can stop it anywhere, even with Ctrl+C or a crash, and rerun.
- Rotates between providers; a provider that hits its daily limit is dropped
  for the rest of the run. When all providers are exhausted, it saves and exits.

Input:  data/chunks_clean.jsonl
Output: data/candidates_raw.jsonl        (one Q&A per line, with provenance)
        data/generation_progress.jsonl   (one line per finished chunk)

Run:    python dataset/generate/generate_qa.py --limit 50
        python dataset/generate/generate_qa.py --dry-run     (no API calls)
"""
import argparse, collections, hashlib, json, os, pathlib, re, sys, time
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]  # project root
DATA = ROOT / "data"
IN_FILE = DATA / "chunks_clean.jsonl"
OUT_FILE = DATA / "candidates_raw.jsonl"
PROGRESS_FILE = DATA / "generation_progress.jsonl"

DOC_CAP_PER_SOURCE = 80   # doc chunks used per doc source (docs are a minor share)
PAPER_RUN = 9             # order: 9 paper chunks, then 1 doc chunk
MAX_FAILURES = 4          # give up on a chunk after this many failed calls
MAX_SERVER_STRIKES = 6    # drop a provider for the run after this many server errors in a row

PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
        "gap_env": "GROQ_SECONDS_BETWEEN_CALLS", "gap_default": 13,
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL",
        "gap_env": "GEMINI_SECONDS_BETWEEN_CALLS", "gap_default": 7,
    },
}

CUE_TYPES = {
    "mechanism": "mechanism, why_how", "tradeoff": "tradeoff",
    "hyperparam": "hyperparameter", "comparison": "comparison",
    "definition": "concept", "code": "implementation",
    "procedure": "implementation", "numeric": "quantitative",
    "equation": "quantitative",
}

SYSTEM = ("You write training data for a machine learning assistant. "
          "You only use facts stated in the passage you are given.")

PROMPT = """Passage (from "<<TITLE>>", section "<<HEADING>>"):
\"\"\"
<<TEXT>>
\"\"\"

Write <<N>> question-answer pairs about this passage.

Rules:
- Use ONLY facts stated in the passage. Do not add outside knowledge.
- Each question must make sense on its own. Never write "the passage", "the text", "the authors", "above", "this work", "the current work" or "our method". Name the method or concept explicitly.
- Do not ask who did what in earlier work (for example "according to Krizhevsky"). Ask about the concept, method, or result itself.
- Answers must contain only claims stated in the passage. Do not add conclusions, interpretations, or advice the passage does not state.
- Never refer to tables, figures, equations by number, or section numbers (for example "Table 9" or "Section 5.4"), and skip facts that only make sense with them.
- Prefer how/why questions, tradeoffs, hyperparameters, comparisons, and practical use. Avoid plain "What is X?" unless the passage is a definition.
- Suggested task types for this passage: <<TYPES>>.
- Answers: 40 to 180 words, self-contained, written as a direct explanation. Do not mention "the passage".
- evidence: copy 1 to 2 sentences from the passage exactly, character for character (keep any parentheses), that support the answer.
- Keep formulas as they appear in the passage.
- If the passage has broken sentences (missing subjects or citations), little teachable content, or mostly lists other work, return [].

Return ONLY a JSON array, no other text:
[{"question": "...", "answer": "...", "task_type": "...", "evidence": "..."}]
task_type must be one of: concept, mechanism, why_how, comparison, tradeoff, hyperparameter, debugging, scenario, implementation, quantitative, experiment_design."""


def load_env():
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def chunk_key(c):
    """Stable ID from the chunk's content, so progress survives re-chunking."""
    return hashlib.sha1((c.get("doc", "") + "\n" + c["text"]).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path):
    """Read a JSONL file, skipping a truncated line left by a crash."""
    rows = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_done(chunks):
    """Keys of finished chunks. Also understands records written by older versions
    of this script (they only stored chunk_id) by mapping the id to today's chunks."""
    id_to_key = {c["id"]: chunk_key(c) for c in chunks}
    done = set()
    # a stop can leave Q&A written but no progress line, so read both files
    for r in read_jsonl(PROGRESS_FILE) + read_jsonl(OUT_FILE):
        k = r.get("chunk_key") or id_to_key.get(r.get("chunk_id"))
        if k:
            done.add(k)
    return done


def ensure_newline(path):
    """If a crash cut the last line short, start the next write on a fresh line."""
    if path.exists() and path.stat().st_size:
        with open(path, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")


def ordered_chunks():
    """Papers first by score, with a doc chunk after every PAPER_RUN papers."""
    chunks = [json.loads(l) for l in open(IN_FILE, encoding="utf-8")]
    by_score = lambda c: -c.get("score", 0)
    papers = sorted([c for c in chunks if c.get("source_type") == "paper"], key=by_score)
    docs, per_source = [], collections.Counter()
    for c in sorted([c for c in chunks if c.get("source_type") != "paper"], key=by_score):
        if per_source[c["source"]] < DOC_CAP_PER_SOURCE:
            per_source[c["source"]] += 1
            docs.append(c)
    order = []
    while papers or docs:
        order += papers[:PAPER_RUN]
        papers = papers[PAPER_RUN:]
        order += docs[:1]
        docs = docs[1:]
    return order


def build_prompt(chunk):
    types = []
    for cue in chunk.get("cues", []):
        for t in CUE_TYPES.get(cue, "").split(", "):
            if t and t not in types:
                types.append(t)
    n = 3 if chunk.get("score", 0) >= 6 else 2
    return (PROMPT.replace("<<TITLE>>", chunk.get("source_title", ""))
            .replace("<<HEADING>>", chunk.get("heading_path", ""))
            .replace("<<TEXT>>", chunk["text"])
            .replace("<<N>>", str(n))
            .replace("<<TYPES>>", ", ".join(types) or "any"))


def parse_response(text):
    """Return a list of Q&A dicts ([] if the model declined), or None if unparseable."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    s, e = t.find("["), t.rfind("]")
    if s == -1 or e == -1:
        return None
    try:
        arr = json.loads(t[s:e + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    out = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        q, a, tt, ev = (str(it.get(k, "")).replace("\u2010", "-").replace("\u2011", "-").strip()
                        for k in ("question", "answer", "task_type", "evidence"))
        if len(q) < 15 or len(a.split()) < 15 or not ev:
            continue
        out.append({"question": q, "answer": a, "task_type": tt, "evidence": ev})
    return out


def call_api(name, prompt):
    p = PROVIDERS[name]
    headers = {"Authorization": f"Bearer {os.environ[p['key_env']]}",
               "Content-Type": "application/json"}
    # generous limit: thinking models spend part of it on hidden reasoning
    body = {"model": os.environ[p["model_env"]], "temperature": 0.3, "max_tokens": 4000,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}]}
    try:
        r = requests.post(p["url"], headers=headers, json=body, timeout=120)
    except requests.RequestException as e:
        return {"ok": False, "code": 0, "body": str(e), "retry_after": 10}
    if r.status_code == 200:
        try:
            choice = r.json()["choices"][0]
            text = choice["message"]["content"] or ""
            finish = choice.get("finish_reason")
        except Exception:
            return {"ok": False, "code": 200, "body": r.text[:300], "retry_after": 5}
        if not text.strip() or finish == "length":
            # empty or cut off: retry instead of marking the chunk as done
            return {"ok": False, "code": 200, "retry_after": 5,
                    "body": f"empty or cut-off reply (finish_reason={finish})"}
        return {"ok": True, "text": text}
    ra = r.headers.get("retry-after", "")
    return {"ok": False, "code": r.status_code, "body": r.text[:2000],
            "retry_after": float(ra) if re.fullmatch(r"\d+(\.\d+)?", ra) else None}


def is_server_error(res):
    """Overload / network / malformed reply: not the chunk's fault."""
    return res["code"] in (0, 200) or res["code"] >= 500


def handle_failure(name, state, res):
    code, body, now = res["code"], res["body"].lower(), time.time()
    s = state[name]
    if code in (401, 403):
        s["disabled"] = f"auth error ({code}): check the API key"
    elif code == 404 or (code == 400 and "model" in body):
        s["disabled"] = "model not found: check the model name in .env"
    elif code == 429:
        ra = res.get("retry_after")
        m = re.search(r'"retrydelay":\s*"(\d+(?:\.\d+)?)s"', body)  # Gemini's suggested wait
        if not ra and m:
            ra = float(m.group(1))
        daily = any(w in body for w in ("per day", "perday", "daily", "tpd", "rpd"))
        if "limit: 0" in body:
            s["disabled"] = "no free quota for this model: try another model name"
        elif daily or (ra and ra > 300):
            s["disabled"] = "daily limit reached"
        else:
            s["ready_at"] = now + (ra or 30)
    elif is_server_error(res):
        s["strikes"] += 1
        if s["strikes"] >= MAX_SERVER_STRIKES:
            s["disabled"] = (f"server errors (HTTP {code}) keep happening: "
                             "try another model name or run later")
        else:
            s["ready_at"] = now + min(30 * 2 ** (s["strikes"] - 1), 600)
    else:
        s["ready_at"] = now + (res.get("retry_after") or 10)
    short = " ".join(res["body"].split())[:90]
    if s["disabled"]:
        print(f"  [{name}] disabled for this run: {s['disabled']}")
    else:
        wait = max(0, int(s["ready_at"] - now))
        print(f"  [{name}] HTTP {code}, waiting {wait}s before using it again: {short}")


def pick_provider(state):
    active = [n for n, s in state.items() if not s["disabled"]]
    if not active:
        return None
    name = min(active, key=lambda n: state[n]["ready_at"])
    wait = state[name]["ready_at"] - time.time()
    if wait > 0:
        time.sleep(wait)
    return name


def process_chunk(chunk, key, state, out, prog, stats, label):
    """Call providers until the chunk is finished or must be skipped.
    Returns False if every provider is exhausted."""
    prompt, failures = build_prompt(chunk), 0
    while True:
        name = pick_provider(state)
        if name is None:
            return False
        res = call_api(name, prompt)
        state[name]["ready_at"] = time.time() + state[name]["gap"]
        if res["ok"]:
            state[name]["strikes"] = 0
            items = parse_response(res["text"])
            status = "bad_json" if items is None else ("empty" if not items else "ok")
            lines = []
            for j, it in enumerate(items or []):
                rec = {"id": f"{key}-q{j}", **it, "chunk_key": key, "chunk_id": chunk["id"],
                       "source": chunk["source"], "source_type": chunk.get("source_type"),
                       "doc": chunk.get("doc"), "source_url": chunk.get("source_url"),
                       "source_title": chunk.get("source_title"),
                       "heading_path": chunk.get("heading_path"),
                       "license": chunk.get("license"), "tier": chunk.get("tier"),
                       "cues": chunk.get("cues"), "method": "llm", "provider": name,
                       "model": os.environ[PROVIDERS[name]["model_env"]]}
                lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
            # one write per chunk: Q&A first, then the progress line
            if lines:
                out.write("".join(lines))
                out.flush()
            prog.write(json.dumps({"chunk_key": key, "chunk_id": chunk["id"], "status": status,
                                   "provider": name, "n": len(lines)}) + "\n")
            prog.flush()
            stats[status] += 1
            stats["qa"] += len(lines)
            stats["by_" + name] += 1
            print(f"{label} {chunk['id']} {name} -> {status} ({len(lines)})")
            return True
        handle_failure(name, state, res)
        if not is_server_error(res):
            failures += 1
        if failures >= MAX_FAILURES:
            stats["skipped"] += 1
            print(f"{label} {chunk['id']} skipped after {failures} failures (will retry next run)")
            return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="max new chunks this run")
    ap.add_argument("--dry-run", action="store_true", help="show plan and prompt, no API calls")
    args = ap.parse_args(argv)
    load_env()

    if not IN_FILE.exists():
        sys.exit(f"missing {IN_FILE} - run dataset/process/clean_chunks.py first")
    all_chunks = ordered_chunks()
    done = load_done(all_chunks)
    queue = [c for c in all_chunks if chunk_key(c) not in done][:args.limit]

    if args.dry_run:
        print(f"{len(done)} chunks already done; {len(queue)} would run now")
        for c in queue[:5]:
            print(" ", c["id"], c["source"], "score", c.get("score"))
        if queue:
            print("\n--- first prompt ---\n" + build_prompt(queue[0]))
        return

    state = {}
    for name, p in PROVIDERS.items():
        if os.environ.get(p["key_env"]) and os.environ.get(p["model_env"]):
            state[name] = {"ready_at": 0.0, "disabled": None, "strikes": 0,
                           "gap": float(os.environ.get(p["gap_env"], p["gap_default"]))}
    if not state:
        sys.exit("no provider configured: copy .env.example to .env and add a key and model name")
    print("providers:", ", ".join(state), f"| already done (skipped): {len(done)} | chunks to run: {len(queue)}")

    stats = collections.Counter()
    DATA.mkdir(exist_ok=True)
    ensure_newline(OUT_FILE)
    ensure_newline(PROGRESS_FILE)
    try:
        with open(OUT_FILE, "a", encoding="utf-8") as out, \
                open(PROGRESS_FILE, "a", encoding="utf-8") as prog:
            for i, chunk in enumerate(queue, 1):
                label = f"[{i}/{len(queue)}]"
                if not process_chunk(chunk, chunk_key(chunk), state, out, prog, stats, label):
                    print("\nAll providers are exhausted. Progress is saved; rerun later to continue.")
                    break
    except KeyboardInterrupt:
        print("\nStopped. Everything finished so far is saved; rerun to continue.")

    print("\nsummary:", dict(stats))
    for n, s in state.items():
        if s["disabled"]:
            print(f"  {n}: {s['disabled']}")


if __name__ == "__main__":
    main()