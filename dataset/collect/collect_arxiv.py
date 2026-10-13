"""Fetch seed arXiv papers via ar5iv HTML, split into section chunks.
Output: data/chunks_papers.jsonl
Run from anywhere: python dataset/collect/collect_arxiv.py
"""
import requests, time, json, re, pathlib
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

HDR = {"User-Agent": "ml-dataset-research/0.1 (mailto:YOUR_EMAIL)"}  # <-- put your email
ROOT = pathlib.Path(__file__).resolve().parents[2]  # project root
DATA = ROOT / "data"
RAW = DATA / "raw" / "arxiv"
RAW.mkdir(parents=True, exist_ok=True)
MIN_W, MAX_W = 60, 450
SKIP = re.compile(r"^(references|bibliography|acknowledg|appendix|author contributions)", re.I)

# Seed list. Check titles in the printout; fix or drop any wrong ID.
IDS = """
1706.03762 2106.09685 2305.14314 2305.18290 1412.6980 1711.05101 2205.14135
2307.08691 2203.02155 2001.08361 2203.15556 1910.02054 1909.08053 1502.03167
1607.06450 1512.03385 1810.04805 2104.09864 2402.03300 1706.02677 1710.03740
2208.07339 2210.17323 2306.00978 2309.06180 1604.06174 1207.0580 2302.13971
2005.14165 1301.3781 1508.07909 2305.11206 2101.03961 2312.00752 1503.02531
2201.11903 2212.08073 2501.12948 1608.03983 1803.03635 1706.04599
""".split()


def metadata(ids):
    r = requests.get("http://export.arxiv.org/api/query",
                     params={"id_list": ",".join(ids), "max_results": len(ids)},
                     headers=HDR, timeout=60)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    meta = {}
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/abs/", 1)[1]
        aid = re.sub(r"v\d+$", "", aid)
        meta[aid] = " ".join(e.find("a:title", ns).text.split())
    return meta


def fetch_html(aid):
    f = RAW / f"{aid}.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    r = requests.get(f"https://ar5iv.labs.arxiv.org/html/{aid}", headers=HDR, timeout=60)
    time.sleep(3)  # be polite
    if r.status_code != 200:
        return None
    f.write_text(r.text, encoding="utf-8")
    return r.text


def parse_sections(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup.select("cite, .ltx_bibliography, .ltx_note, .ltx_tag, "
                         "script, style, figure, table"):
        t.decompose()
    for m in soup.find_all("math"):
        m.replace_with(" " + (m.get("alttext") or "") + " ")
    heading, paras, out = "Abstract", [], []
    for el in soup.select("h2.ltx_title, h3.ltx_title, h4.ltx_title, p.ltx_p"):
        if el.name.startswith("h"):
            if paras:
                out.append((heading, paras))
            heading, paras = " ".join(el.get_text(" ").split()), []
        else:
            txt = " ".join(el.get_text(" ").split())
            if txt:
                paras.append(txt)
    if paras:
        out.append((heading, paras))
    return [(h, p) for h, p in out if not SKIP.match(h)]


def pack(paras):
    """Merge paragraphs into chunks of at most MAX_W words."""
    chunks, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if n + w > MAX_W and cur:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


meta = metadata(IDS)
out, failed = [], []
for aid in IDS:
    html = fetch_html(aid)
    if not html:
        failed.append(aid)
        continue
    title = meta.get(aid, aid)
    print(aid, "->", title)
    for heading, paras in parse_sections(html):
        for c in pack(paras):
            if len(c.split()) < MIN_W:
                continue
            out.append({"id": f"arxiv-{len(out):06d}", "source": "arxiv",
                        "source_type": "paper", "doc": f"arxiv/{aid}",
                        "source_url": f"https://arxiv.org/abs/{aid}",
                        "source_title": title, "heading_path": heading,
                        "license": "arXiv (varies per paper)", "tier": "A", "text": c})

with open(DATA / "chunks_papers.jsonl", "w",encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps(r) + "\n")
print(f"\n{len(IDS) - len(failed)} papers -> {len(out)} chunks; failed: {failed}")