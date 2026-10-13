"""Chunk the cloned doc repos by heading.
Input:  data/raw/<repo>/
Output: data/chunks_docs.jsonl
Run after collect_docs.py: python dataset/process/chunk_docs.py
"""
import re, json, pathlib

SOURCES = {  # name -> (github repo, license)
    "transformers": ("huggingface/transformers", "Apache-2.0"),
    "peft": ("huggingface/peft", "Apache-2.0"),
    "trl": ("huggingface/trl", "Apache-2.0"),
    "accelerate": ("huggingface/accelerate", "Apache-2.0"),
    "sklearn": ("scikit-learn/scikit-learn", "BSD-3-Clause"),
    "pt_tutorials": ("pytorch/tutorials", "BSD-3-Clause"),
}
EXT = {".md", ".mdx", ".rst"}
RST_UNDERLINE = re.compile(r"^([=\-~^\"#*+])\1{2,}\s*$")
MIN_WORDS, MAX_WORDS = 60, 450


def rst_to_md_headings(text):
    lines, out, i = text.split("\n"), [], 0
    levels = {}
    while i < len(lines):
        if (i + 1 < len(lines) and lines[i].strip()
                and RST_UNDERLINE.match(lines[i + 1])
                and len(lines[i + 1].strip()) >= len(lines[i].strip())):
            ch = lines[i + 1].strip()[0]
            lvl = levels.setdefault(ch, len(levels) + 1)
            out.append("#" * min(lvl, 4) + " " + lines[i].strip())
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def sections(text):
    """Yield (heading_path, body). Code fences are never split."""
    path, buf, in_code, res = [], [], False, []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            res.append((" > ".join(path), body))
        buf.clear()

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            depth = len(m.group(1))
            path[:] = path[:depth - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return res


def split_long(body):
    cur, n, parts = [], 0, []
    for p in body.split("\n\n"):
        w = len(p.split())
        if n + w > MAX_WORDS and cur:
            parts.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        parts.append("\n\n".join(cur))
    return parts


def clean(text):
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)         # front matter
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)               # comments
    text = re.sub(r"^\s*(import|export) .*$", "", text, flags=re.M)  # mdx
    return text


ROOT = pathlib.Path(__file__).resolve().parents[2]  # project root
DATA = ROOT / "data"
out, n_files = [], 0
for name, (repo, lic) in SOURCES.items():
    root = DATA / "raw" / name
    if not root.exists():
        print("missing", name, "- run collect_docs.py first")
        continue
    for f in root.rglob("*"):
        if f.suffix not in EXT or ".git" in f.parts:
            continue
        n_files += 1
        text = clean(f.read_text(errors="ignore"))
        if f.suffix == ".rst":
            text = rst_to_md_headings(text)
        rel = f.relative_to(root).as_posix()
        url = f"https://github.com/{repo}/blob/HEAD/{rel}"
        for hpath, body in sections(text):
            for part in split_long(body):
                if len(part.split()) < MIN_WORDS:
                    continue
                out.append({"id": f"{name}-{len(out):06d}", "source": name,
                            "source_type": "official_docs", "doc": f"{name}/{rel}",
                            "source_url": url,
                            "source_title": hpath.split(" > ")[0] or f.stem,
                            "heading_path": hpath, "license": lic,
                            "tier": "A", "text": part})

DATA.mkdir(exist_ok=True)
with open(DATA / "chunks_docs.jsonl", "w") as fh:
    for r in out:
        fh.write(json.dumps(r) + "\n")
print(f"{n_files} files -> {len(out)} chunks")