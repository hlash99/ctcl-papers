#!/usr/bin/env python3
"""Server-side refresh for the CTCL literature tracker.

Polls Europe PMC (PubMed/MEDLINE + preprints) for the latest cutaneous T-cell
lymphoma papers, merges any not-yet-seen ones into ``data.json`` (newest first),
and writes a short plain-language summary of each new paper.

Summaries use whichever key is set: ``ANTHROPIC_API_KEY`` (Claude, preferred) or
``GROQ_API_KEY`` (Groq's free OpenAI-compatible API serving open-weights Llama
models — ``CTCL_GROQ_MODEL``, default ``llama-3.3-70b-versatile``). Without a key
— or if a call fails — it falls back to a trimmed slice of the abstract so the
page always renders. Existing papers are never re-summarized, so a weekly run
only spends on what's genuinely new.

Resilient by design: a network/API failure leaves the last-good ``data.json``
untouched (no commit) rather than wiping it.

Run locally:  python3 scripts/fetch_papers.py
In CI:        .github/workflows/refresh.yml
"""
import html
import json
import os
import re
import sys
from datetime import datetime, timezone

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data.json")

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Title / keyword / MeSH focused so we get CTCL-*centric* papers, not every
# paper that mentions CTCL once in passing. Covers the disease and its two main
# subtypes (mycosis fungoides, Sézary syndrome).
QUERY = (
    'TITLE:"cutaneous T-cell lymphoma" OR TITLE:"mycosis fungoides" '
    'OR TITLE:"Sezary syndrome" OR TITLE:"Sézary syndrome" OR TITLE:"CTCL" '
    'OR KW:"mycosis fungoides" OR KW:"Sezary syndrome" '
    'OR MESH:"Lymphoma, T-Cell, Cutaneous"'
)
QUERY_HUMAN = "cutaneous T-cell lymphoma · mycosis fungoides · Sézary syndrome (CTCL)"

PAGE_SIZE = 50      # newest candidates pulled per run
KEEP_MAX = 400      # cap on stored papers
SUMMARY_MODEL = os.environ.get("CTCL_SUMMARY_MODEL", "claude-opus-4-8")
# Groq serves open-weights Llama models on a free, no-card OpenAI-compatible API.
GROQ_MODEL = os.environ.get("CTCL_GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
UA = ("ctcl-tracker/1.0 "
      "(+https://github.com/hlash99/ctcl-papers; hassan.lash@gmail.com)")

SUMMARY_SYSTEM = (
    "You summarize biomedical journal abstracts about cutaneous T-cell lymphoma "
    "(CTCL) for an educated layperson who is closely tracking the disease. Write "
    "2-3 plain-language sentences covering what the study examined and its key "
    "finding or takeaway. Briefly gloss any technical term. No preamble, no "
    "markdown, no citations — respond with only the summary text."
)

# Rolling "big picture" briefing across the whole recent corpus — regenerated
# whenever new papers arrive, written for a patient / family member, not a doctor.
OVERVIEW_SYSTEM = (
    "You write a plain-language 'big picture' briefing on recent cutaneous T-cell "
    "lymphoma (CTCL) research for a smart, non-medical reader following the field "
    "closely — a patient or a family member, NOT a doctor. Given a list of recent "
    "papers (titles + short summaries), synthesize the MOST IMPORTANT cross-cutting "
    "themes: emerging treatments, diagnostic advances, shifts in understanding, and "
    "practical cautions. Rules: AT MOST 5 bullets; each a single plain-English "
    "sentence; no jargon (briefly gloss any unavoidable term); group related findings "
    "rather than listing papers one by one; lead with what matters most to a patient. "
    "Respond with ONLY a JSON array of bullet strings, nothing else."
)


def build_overview(complete, papers, k=18):
    """Return <=5 plain-language takeaway bullets synthesized across recent papers."""
    recent = sorted(papers, key=sort_key, reverse=True)[:k]
    lines = []
    for p in recent:
        blurb = p.get("summary") or p.get("abstract") or ""
        lines.append(f"- {p.get('title','')} ({p.get('journal','')}, "
                     f"{(p.get('date') or '')[:7]}): {blurb[:400]}")
    text = complete(OVERVIEW_SYSTEM, "Recent CTCL papers:\n\n" + "\n".join(lines), 800)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        bullets = [str(b).strip() for b in json.loads(text) if str(b).strip()]
    except Exception:
        bullets = [re.sub(r"^[-*•\d.\s]+", "", ln).strip()
                   for ln in text.splitlines() if ln.strip()]
    return bullets[:5]


def clean(s):
    """Unescape entities, drop inline markup tags, collapse whitespace.

    Order matters: unescape first so entity-encoded tags (``&lt;sub&gt;``)
    become real tags that the tag-strip then removes.
    """
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


# Europe PMC structured abstracts glue a leading section label onto the text
# ("AbstractCutaneous…", "Background and purposeRadiotherapy…"). Strip a leading
# "Abstract" so the fallback excerpt reads cleanly; AI summaries are unaffected.
def clean_abstract(s):
    s = clean(s)
    return re.sub(r"^Abstract[:.\s]*(?=[A-Z0-9])", "", s).strip()


def trim(text, words=55):
    parts = (text or "").split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:words]).rstrip(".,;: ") + " …"


def paper_url(r):
    pmid, doi = r.get("pmid"), r.get("doi")
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if doi:
        return f"https://doi.org/{doi}"
    return f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}"


def fetch_candidates():
    params = {
        "query": QUERY,
        "format": "json",
        "pageSize": str(PAGE_SIZE),
        "sort": "P_PDATE_D desc",
        "resultType": "core",
    }
    r = requests.get(EPMC, params=params, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    return r.json().get("resultList", {}).get("result", [])


def make_llm():
    """Pick a summarization backend by whichever key is set, else None.

    Prefers Anthropic (ANTHROPIC_API_KEY + SDK); otherwise Groq's free
    OpenAI-compatible endpoint (GROQ_API_KEY), which serves open-weights Llama
    models. Returns (name, complete) where complete(system, user, max_tokens)
    -> str; None means no key configured (page keeps abstract excerpts).
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()

            def complete(system, user, max_tokens):
                m = client.messages.create(model=SUMMARY_MODEL, max_tokens=max_tokens,
                                           system=system,
                                           messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in m.content if b.type == "text").strip()
            return ("claude", complete)
        except ImportError:
            print("anthropic SDK not installed — trying Groq", file=sys.stderr)

    if os.environ.get("GROQ_API_KEY"):
        key = os.environ["GROQ_API_KEY"]

        def complete(system, user, max_tokens):
            r = requests.post(GROQ_URL, timeout=90,
                              headers={"Authorization": "Bearer " + key,
                                       "Content-Type": "application/json"},
                              json={"model": GROQ_MODEL, "max_tokens": max_tokens,
                                    "temperature": 0.3,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]})
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        return ("groq", complete)

    return None


def record_from(r, summary, summary_by, today):
    date = r.get("firstPublicationDate") or ""
    return {
        "id": f"{r.get('source')}:{r.get('id')}",
        "pmid": r.get("pmid"),
        "doi": r.get("doi"),
        "title": clean(r.get("title")),
        "authors": clean(r.get("authorString")),
        "journal": clean(((r.get("journalInfo") or {}).get("journal") or {}).get("title")),
        "date": date,
        "year": int(date[:4]) if date[:4].isdigit() else None,
        "is_preprint": r.get("source") == "PPR",
        "url": paper_url(r),
        "abstract": clean_abstract(r.get("abstractText")),
        "summary": summary,
        "summary_by": summary_by,   # "claude" | "abstract" | "none"
        "first_seen": today,
    }


def load_data():
    if os.path.exists(DATA):
        with open(DATA) as f:
            return json.load(f)
    return {"papers": []}


def sort_key(p):
    return (p.get("date") or "", p.get("first_seen") or "")


def main():
    data = load_data()
    papers = data.get("papers", [])
    seen_id = {p["id"] for p in papers}
    seen_pmid = {p.get("pmid") for p in papers if p.get("pmid")}
    seen_doi = {(p.get("doi") or "").lower() for p in papers if p.get("doi")}

    try:
        candidates = fetch_candidates()
    except Exception as e:
        print(f"fetch failed ({e.__class__.__name__}: {e}) — keeping last-good",
              file=sys.stderr)
        return 0

    llm = make_llm()
    backend = llm[0] if llm else None
    complete = llm[1] if llm else None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new = []
    for r in candidates:
        rid = f"{r.get('source')}:{r.get('id')}"
        pmid = r.get("pmid")
        doi = (r.get("doi") or "").lower()
        if rid in seen_id or (pmid and pmid in seen_pmid) or (doi and doi in seen_doi):
            continue
        title = clean(r.get("title"))
        abstract = clean_abstract(r.get("abstractText"))
        # Start every paper on the abstract-excerpt fallback; the upgrade pass
        # below promotes it to a real LLM summary when a key is available.
        summary, by = (trim(abstract), "abstract") if abstract else ("", "none")
        new.append(record_from(r, summary, by, today))
        seen_id.add(rid)
        if pmid:
            seen_pmid.add(pmid)
        if doi:
            seen_doi.add(doi)

    papers = sorted(new + papers, key=sort_key, reverse=True)[:KEEP_MAX]

    # Upgrade abstract-excerpt summaries to plain-language LLM summaries when a
    # key (Anthropic or Groq) is set. Self-healing: the first keyed run backfills
    # the whole backlog, and any paper already summarized by an LLM is skipped —
    # so a weekly run only spends on genuinely new (or previously-failed) papers.
    if complete:
        upgraded = 0
        for p in papers:
            if p.get("summary_by") in ("claude", "groq") or not p.get("abstract"):
                continue
            try:
                p["summary"] = complete(SUMMARY_SYSTEM,
                                        f"Title: {p['title']}\n\nAbstract: {p['abstract']}", 400)
                p["summary_by"] = backend
                upgraded += 1
            except Exception as e:
                print(f"summary failed for {p['id']} ({e.__class__.__name__}) — kept abstract",
                      file=sys.stderr)
        print(f"{backend} summaries written this run: {upgraded}")

    # Rolling plain-language overview — regenerate when new papers landed this run
    # (or none stored yet). Needs a key; without one, any prior overview is kept.
    if complete and (new or not data.get("overview")):
        try:
            bullets = build_overview(complete, papers)
            if bullets:
                data["overview"] = {
                    "bullets": bullets,
                    "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "n_papers": len(papers),
                    "by": backend,
                }
                print(f"overview regenerated: {len(bullets)} bullets")
        except Exception as e:
            print(f"overview failed ({e.__class__.__name__}: {e}) — keeping previous",
                  file=sys.stderr)

    data["title"] = "CTCL Literature Tracker"
    data["description"] = (
        "Latest peer-reviewed papers and preprints on cutaneous T-cell lymphoma, "
        "auto-collected and summarized."
    )
    data["query_human"] = QUERY_HUMAN
    data["source"] = "Europe PMC (PubMed / MEDLINE + preprints)"
    data["stats"] = {"total": len(papers), "new_this_run": len(new)}
    data["updated"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    data["papers"] = papers

    with open(DATA, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"candidates={len(candidates)} new={len(new)} total={len(papers)} "
          f"summaries={'claude' if client else 'abstract'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
