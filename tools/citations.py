"""Refresh data/citations.json from OpenAlex for every paper in data/publications.toml.

Google Scholar has no API, so counts come from OpenAlex (open data, no key). A paper's count
is the sum over its DOI and its arXiv DOI when they resolve to different OpenAlex works,
which mirrors how Scholar merges versions. Runs in the Pages workflow before each build and
on the weekly schedule; on failure the committed snapshot is left untouched.
"""
import datetime
import json
import sys
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBS = ROOT / "data" / "publications.toml"
OUT = ROOT / "data" / "citations.json"


def work(doi):
    url = ("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi.lower(), safe="/.")
           + "?select=id,cited_by_count")
    req = urllib.request.Request(url, headers={"User-Agent": "ccao2679.com citation refresh"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
            return d["id"], int(d.get("cited_by_count") or 0)
    except Exception as e:  # 404 = not indexed yet
        print(f"  {doi}: {e}", file=sys.stderr)
        return None


def main():
    pubs = tomllib.loads(PUBS.read_text())["pub"]
    by_id, ok = {}, 0
    for p in pubs:
        dois = [d for d in (p.get("doi"), f"10.48550/arXiv.{p['arxiv']}" if p.get("arxiv") else None) if d]
        seen, n = set(), 0
        for d in dois:
            w = work(d)
            if w and w[0] not in seen:
                seen.add(w[0]); n += w[1]; ok += 1
        by_id[p["id"]] = n
    if ok == 0:
        sys.exit("OpenAlex unreachable or nothing indexed; keeping the old snapshot")
    counts = sorted(by_id.values(), reverse=True)
    h = sum(1 for i, c in enumerate(counts, 1) if c >= i)
    out = {"source": "OpenAlex", "updated": datetime.date.today().isoformat(),
           "total": sum(counts), "h_index": h, "by_id": by_id}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(pubs)} papers, {out['total']} citations, h-index {h}")


if __name__ == "__main__":
    main()
