"""Shallow, sparse-clone a few official doc folders into data/raw/.
Small side source (capped at ~10% of the final dataset).
Requires git. Run: python dataset/collect/collect_docs.py
"""
import subprocess, pathlib

SOURCES = {
    "transformers": ("https://github.com/huggingface/transformers", ["docs/source/en"]),
    "peft":         ("https://github.com/huggingface/peft", ["docs/source"]),
    "trl":          ("https://github.com/huggingface/trl", ["docs/source"]),
    "accelerate":   ("https://github.com/huggingface/accelerate", ["docs/source"]),
    "sklearn":      ("https://github.com/scikit-learn/scikit-learn", ["doc/modules"]),
    "pt_tutorials": ("https://github.com/pytorch/tutorials",
                     ["beginner_source", "intermediate_source", "advanced_source"]),
}

ROOT = pathlib.Path(__file__).resolve().parents[2]  # project root
DATA = ROOT / "data"
raw = DATA / "raw"
raw.mkdir(parents=True, exist_ok=True)
for name, (url, paths) in SOURCES.items():
    dest = raw / name
    if dest.exists():
        print("skip", name)
        continue
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                    "--sparse", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *paths], check=True)
    print("done", name)