"""Clean the chunks before question generation.

- skips low-value paper sections (related work, baselines, setup, ...)
- removes exact duplicates and junk (symbol dumps, repetition, license text)
- rejects literature-review chunks (many "et al." / section references)
- rejects chunks with many broken citation gaps ("studied by BitFit .")
- detects knowledge cues (definition, mechanism, tradeoff, ...) and keeps
  only chunks with enough teachable content

Input:  data/chunks_papers.jsonl, data/chunks_docs.jsonl
Output: data/chunks_clean.jsonl   (adds "cues" and "score")
Run:    python dataset/process/clean_chunks.py
"""
import json, re, hashlib, pathlib, collections

ROOT = pathlib.Path(__file__).resolve().parents[2]  # project root
DATA = ROOT / "data"
INPUTS = ["chunks_papers.jsonl", "chunks_docs.jsonl"]

MIN_SCORE = 2.0         # keep chunks at or above this score
MIN_ALPHA_RATIO = 0.5   # letters / characters; filters tables and symbol dumps
MAX_CITATION_GAPS = 3   # reject paper chunks with more broken citations than this
MAX_LIT_REFS = 2         # reject paper chunks with more "et al." / section refs than this

# paper sections that rarely contain teachable content (papers only)
SKIP_SECTIONS = re.compile(
    r"related works?|prior work|baselines?|experimental (?:setup|settings?)|"
    r"implementation details|broader impacts?|ethic|reproducibility|"
    r"funding|author contributions|acknowledg", re.I)

# removing citations leaves gaps like "studied by BitFit ." or "discussed in ."
GAP_RE = re.compile(r"(?<![\\{_^])\b[A-Za-z]{2,} [.,;]")
SPACE_BEFORE_PUNCT = re.compile(r"(?<=[A-Za-z]) ([.,;])")
# "Chen et al. ...", "(§4.6)": prior-work discussion or references to other parts of the paper
LIT_REF = re.compile(r"\bet al\.|§\s?\d")

# cue name -> (regex, weight)
CUES = {
    "definition": (r"\b(is defined as|refers to|is called|we define|denotes|is the process of|"
                   r"is a (?:technique|method|type|form|class|family|model|algorithm))\b", 2.0),
    "mechanism": (r"\b(works by|results in|leads to|which (?:allows|enables|reduces|improves|prevents)|"
                  r"because|so that|in order to|by (?:recomputing|storing|reducing|applying|computing|adding))\b", 2.0),
    "tradeoff": (r"\b(at the cost of|trade-?offs?|drawbacks?|limitations?|downsides?|comes at)\b", 2.0),
    "hyperparam": (r"\b(learning rate|batch size|lora rank|rank r|rank of \d+|r\s?=\s?\d+|dropout|"
                   r"weight decay|warmup|epochs?|temperature|hyperparameters?|defaults? to)\b", 1.0),
    "comparison": (r"\b(unlike|whereas|compared (?:to|with)|in contrast|versus)\b", 1.0),
    "numeric": (r"\d+(?:\.\d+)?\s?(?:%|x|×|GB|MB|ms|FLOPs?)", 1.0),
    "code": (r"```|\bdef \w+\(|\bimport \w+|\btorch\.\w+", 1.0),
    "procedure": (r"(?m)^\s*(?:\d+[.)]|Step \d+)", 1.0),
    "caveat": (r"\bhowever\b", 0.5),
    "equation": (r"\\(?:frac|sum|mathbf|nabla)|[∑∏∇]|\b[a-zA-Z]_\{?\w", 0.5),
}
CUE_RE = {k: (re.compile(p, re.I), w) for k, (p, w) in CUES.items()}


def norm_hash(text):
    t = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def junk_reason(text):
    if sum(c.isalpha() for c in text) / max(len(text), 1) < MIN_ALPHA_RATIO:
        return "low_alpha_ratio"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 5 and len(set(lines)) / len(lines) < 0.6:
        return "repetitive"
    if re.search(r"all rights reserved|copyright \d{4}|licensed under", text, re.I):
        return "license_boilerplate"
    return None


seen, kept, rejected = set(), [], collections.Counter()
per_source = collections.Counter()
cue_counts = collections.Counter()
total = 0

for name in INPUTS:
    path = DATA / name
    if not path.exists():
        print("missing", name, "- skipping")
        continue
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            total += 1
            rec = json.loads(line)
            text = rec["text"]
            is_paper = rec.get("source_type") == "paper"

            if is_paper and SKIP_SECTIONS.search(rec.get("heading_path", "")):
                rejected["skipped_section"] += 1
                continue
            h = norm_hash(text)
            if h in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(h)
            reason = junk_reason(text)
            if reason:
                rejected[reason] += 1
                continue
            if is_paper:
                if len(GAP_RE.findall(text)) > MAX_CITATION_GAPS:
                    rejected["citation_gaps"] += 1
                    continue
                if len(LIT_REF.findall(text)) > MAX_LIT_REFS:
                    rejected["literature_review"] += 1
                    continue
                text = SPACE_BEFORE_PUNCT.sub(r"\1", text)  # "BitFit ." -> "BitFit."
                rec["text"] = text

            cues = [k for k, (rx, _) in CUE_RE.items() if rx.search(text)]
            score = sum(CUE_RE[k][1] for k in cues)
            if score < MIN_SCORE:
                rejected["low_score"] += 1
                continue
            rec["cues"], rec["score"] = cues, score
            kept.append(rec)
            per_source[rec["source"]] += 1
            cue_counts.update(cues)

with open(DATA / "chunks_clean.jsonl", "w", encoding="utf-8") as fh:
    for r in kept:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"input chunks: {total}")
print(f"kept:         {len(kept)}")
print("rejected:    ", dict(rejected))
print("by source:   ", dict(per_source))
print("cue counts:  ", dict(cue_counts))