#!/usr/bin/env python3
"""Server-side refresh for the CTCL literature tracker.

Reads every cutaneous T-cell lymphoma paper Europe PMC (PubMed/MEDLINE +
preprints) lists for the last ``CTCL_LOOKBACK_DAYS`` (default 120) days, merges
any not-yet-seen ones into ``data.json`` (newest first), tags each with a coarse
type (trial / review / case / letter / research) and a Stanford-affiliation flag,
and writes a short plain-language summary of each new paper. Papers pushed past
the ``data.json`` cap move to ``archive.json`` — nothing is ever dropped.

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
import time
from datetime import datetime, timedelta, timezone

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data.json")

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Title / keyword / MeSH focused so we get CTCL-*centric* papers (the "core"
# tier), plus a second "mention" tier for papers that only name the disease in
# the abstract (a T-cell-lymphoma drug trial with a CTCL cohort, say) — those are
# kept but flagged so they can be told apart.
#
# IMPORTANT: every clause is wrapped in its own parentheses by _or(). Europe PMC
# mis-parses a bare OR-chain once a clause contains a comma (the MeSH heading
# "Lymphoma, T-Cell, Cutaneous"): the old un-parenthesised query silently
# matched ~1 in 8 of the papers it should have (34 vs 277 over May-Nov 2026).
CORE_CLAUSES = [
    'TITLE:"cutaneous T-cell lymphoma"', 'TITLE:"cutaneous T-cell lymphomas"',
    'TITLE:"mycosis fungoides"', 'TITLE:Sezary', 'TITLE:CTCL',
    'TITLE:"lymphomatoid papulosis"', 'TITLE:"primary cutaneous anaplastic"',
    'TITLE:"cutaneous lymphoma"', 'TITLE:"cutaneous lymphomas"',
    'TITLE:mogamulizumab',
    'KW:"mycosis fungoides"', 'KW:"Sezary syndrome"',
    'KW:"cutaneous T-cell lymphoma"', 'KW:CTCL',
    'MESH:"Lymphoma, T-Cell, Cutaneous"', 'MESH:"Mycosis Fungoides"',
    'MESH:"Sezary Syndrome"', 'MESH:"Lymphomatoid Papulosis"',
    'MESH:"Pagetoid Reticulosis"',
]
MENTION_CLAUSES = [
    'ABSTRACT:"cutaneous T-cell lymphoma"', 'ABSTRACT:"mycosis fungoides"',
    'ABSTRACT:Sezary', 'ABSTRACT:CTCL',
]


def _or(clauses):
    return "(" + " OR ".join("(" + c + ")" for c in clauses) + ")"


QUERY = _or(CORE_CLAUSES)
QUERY_MENTION = "(" + _or(MENTION_CLAUSES) + " NOT " + QUERY + ")"
QUERY_HUMAN = "cutaneous T-cell lymphoma · mycosis fungoides · Sézary syndrome (CTCL)"


def _int_env(name, default):
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


# Every run re-reads the WHOLE recent window (by first-publication date) and
# de-dupes against data.json, instead of looking only at a "top 50" page. That
# way a paper can't be missed because it was indexed late, sorted low, or only
# got its MeSH terms months after it appeared.
LOOKBACK_DAYS = _int_env("CTCL_LOOKBACK_DAYS", 120)
PAGE_SIZE = 100     # results per Europe PMC page (cursor-paginated)
MAX_PAGES = 30      # safety stop
KEEP_MAX = _int_env("CTCL_KEEP_MAX", 400)   # papers kept in data.json; older ones move to archive.json
MAX_SUMMARIES = _int_env("CTCL_MAX_SUMMARIES", 150)        # LLM summaries per run (newest first)
SUMMARY_BUDGET_S = _int_env("CTCL_SUMMARY_BUDGET_S", 1500)  # wall-clock cap on the summary pass
ARCHIVE = os.path.join(ROOT, "archive.json")
SUMMARY = os.path.join(ROOT, "summary.json")   # slim copy of data.json for the hub card
SUMMARY_MODEL = os.environ.get("CTCL_SUMMARY_MODEL", "claude-opus-4-8")

# Open-LLM providers for server-side summaries — all free-tier, no-credit-card,
# OpenAI-compatible. Each is used ONLY if its key env is set. Resilience is
# two-layered: within a provider, models are tried in order so a decommissioned
# model (Groq retired llama-3.3-70b in June 2026) auto-falls to the next; across
# providers, a full outage falls to the next provider that has a key. Add a
# second key (OPENROUTER_API_KEY / CEREBRAS_API_KEY) to survive a Groq outage.
OPENLLM_PROVIDERS = [
    {"name": "groq", "key": "GROQ_API_KEY",
     "url": "https://api.groq.com/openai/v1/chat/completions",
     "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
     "headers": {}},
    {"name": "openrouter", "key": "OPENROUTER_API_KEY",
     "url": "https://openrouter.ai/api/v1/chat/completions",
     "models": ["meta-llama/llama-3.3-70b-instruct:free",
                "qwen/qwen-2.5-72b-instruct:free",
                "meta-llama/llama-3.1-8b-instruct:free"],
     "headers": {"HTTP-Referer": "https://hlash99.github.io/ctcl-papers/",
                 "X-Title": "CTCL tracker"}},
    {"name": "cerebras", "key": "CEREBRAS_API_KEY",
     "url": "https://api.cerebras.ai/v1/chat/completions",
     "models": ["llama-3.3-70b", "llama3.1-8b"], "headers": {}},
]


def model_body(model):
    """Reasoning-suppression request fields, keyed by model family (the right
    field differs per model): gpt-oss takes reasoning_effort (NOT reasoning_format);
    Qwen3 takes reasoning_format:hidden; plain Llamas need nothing. strip_reasoning()
    is the backstop for anything that still leaks."""
    m = model.lower()
    if "gpt-oss" in m:
        return {"reasoning_effort": "low"}
    if "qwen3" in m or "qwen-3" in m:
        return {"reasoning_format": "hidden"}
    return {}
LAST_PROVIDER = None   # which provider actually served the most recent call
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


# Cheap hallucination guard (no AI): any drug-like name in generated text must
# literally appear in the source text it was written from. On the Nov 2026
# catch-up run this flagged 0 of 278 paper summaries and 1 overview bullet
# (an invented antibody name).
DRUG_RE = re.compile(
    r"\b[A-Za-z][A-Za-z-]{3,}(?:mab|nib|stat|limus|parib|ciclib|sertib|lisib|degib|otene|"
    r"platin|rubicin|trexate|xetan|leukin|kinra|domide|zomib|vedotin|tecan|citabine)\b", re.I)


def _squash(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def ungrounded(text, source):
    """Drug-like names in ``text`` that never occur in ``source``."""
    src = _squash(source)
    return sorted({m.lower() for m in DRUG_RE.findall(text or "") if _squash(m) not in src})


def overview_pool(papers, k=30):
    """The papers the overview is synthesized from: the k most recent that are
    CTCL-centric human studies (veterinary papers and abstract-only mentions are
    left out so a dog study can't lead a briefing written for patients)."""
    pool = [p for p in papers if not p.get("vet") and p.get("match") != "mention"
            and (p.get("summary") or p.get("abstract"))]
    return sorted(pool, key=sort_key, reverse=True)[:k]


def build_overview(complete, recent):
    """Return <=5 plain-language takeaway bullets synthesized across recent papers."""
    lines = []
    for p in recent:
        blurb = p.get("summary") or p.get("abstract") or ""
        lines.append(f"- {p.get('title','')} ({p.get('journal','')}, "
                     f"{(p.get('date') or '')[:7]}): {blurb[:400]}")
    source = "\n".join(lines)
    bullets = []
    for attempt in range(2):             # one retry if a bullet names a drug no paper mentions
        text = complete(OVERVIEW_SYSTEM, "Recent CTCL papers:\n\n" + source, 1500)
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            bullets = [str(b).strip() for b in json.loads(text) if str(b).strip()]
        except Exception:
            bullets = [re.sub(r"^[-*•\d.\s]+", "", ln).strip()
                       for ln in text.splitlines() if ln.strip()]
        bullets = bullets[:5]
        bad = [b for b in bullets if ungrounded(b, source)]
        if not bad:
            return bullets
        print(f"overview attempt {attempt + 1}: {len(bad)} bullet(s) name a drug not in the "
              f"source papers ({ungrounded(' '.join(bad), source)})", file=sys.stderr)
    return [b for b in bullets if not ungrounded(b, source)]   # drop what still fails


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
    # "<h4>Background</h4>Text" -> "Background: Text" (was glued: "BackgroundText")
    s = re.sub(r"\s*</h4>\s*", ": ", html.unescape(s or ""))
    s = re.sub(r"(?<=[^\s>])\s*<h4>", " <h4>", s)
    s = clean(s).replace(":: ", ": ")
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


def _epmc_get(params):
    last = None
    for attempt in range(4):
        try:
            r = requests.get(EPMC, params=params, headers={"User-Agent": UA}, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"Europe PMC HTTP {r.status_code}")
        except (requests.RequestException, ValueError) as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise last


def _search_all(query):
    """Every result for a query (cursor-paginated). Raises if the pages don't
    add up to the reported hit count, so a truncated read is never trusted."""
    out, cursor, hits = [], "*", 0
    for _ in range(MAX_PAGES):
        j = _epmc_get({"query": query, "format": "json", "pageSize": str(PAGE_SIZE),
                       "resultType": "core", "cursorMark": cursor})
        hits = int(j.get("hitCount") or 0)
        res = j.get("resultList", {}).get("result", [])
        out.extend(res)
        nxt = j.get("nextCursorMark")
        if not res or not nxt or nxt == cursor or len(out) >= hits:
            break
        cursor = nxt
    if len(out) < hits:
        raise RuntimeError(f"incomplete read: {len(out)} of {hits} results")
    return out


def fetch_candidates(today):
    """Return (results, coverage). Each result carries "_match": "core" for a
    title/keyword/MeSH hit, "mention" for an abstract-only hit."""
    start = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (today + timedelta(days=400)).strftime("%Y-%m-%d")   # issue dates run ahead
    window = f" AND FIRST_PDATE:[{start} TO {end}]"
    core = _search_all(QUERY + window)
    mention = _search_all(QUERY_MENTION + window)
    for r in core:
        r["_match"] = "core"
    for r in mention:
        r["_match"] = "mention"
    # MEDLINE records first, then preprints, then PMC-only copies, so the
    # title de-dupe keeps the PubMed version of a paper.
    order = {"MED": 0, "PPR": 1}
    res = sorted(core + mention, key=lambda r: order.get(r.get("source"), 2))
    return res, {"window_days": LOOKBACK_DAYS, "from": start,
                 "core_hits": len(core), "mention_hits": len(mention)}


def fetch_by_ids(pmids):
    """Core records for specific PubMed ids (used once, to add publication
    types / affiliations to papers stored before those fields existed)."""
    out = []
    for i in range(0, len(pmids), 40):
        chunk = pmids[i:i + 40]
        q = "(" + " OR ".join(f"(EXT_ID:{x})" for x in chunk) + ") AND (SRC:MED)"
        out.extend(_search_all(q))
    return out


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", "", clean(t).lower())


VET_RE = re.compile(r"\b(canine|dogs?|puppy|puppies|feline|cats?|equine|horses?|bovine|"
                    r"veterinary|ferrets?|rabbits?)\b", re.I)
KIND_TITLE = {
    "trial": re.compile(r"\bphase\s*(?:[1-4]|i{1,3}v?)(?:\s*[/-]\s*(?:[1-4]|i{1,3}v?))?\s+"
                        r"(?:trial|study|clinical|dose|open|random|multi|single|results)|"
                        r"\brandomi[sz]ed\b|\bclinical trial\b|\bopen-label\b", re.I),
    "case": re.compile(r"\bcase reports?\b|\ba case of\b|\bcase series\b|"
                       r"\ba rare case\b|\bcase study\b", re.I),
    "review": re.compile(r"\breview\b|\bmeta-analysis\b|\bguidelines?\b|"
                         r"\bconsensus\b", re.I),
}


def classify(title, pub_types):
    """Coarse paper type from PubMed publication types, falling back to title
    wording (new papers carry only 'Journal Article' until MEDLINE indexes them)."""
    pts = " | ".join(pub_types or []).lower()
    title = title or ""
    if "clinical trial" in pts or "randomized controlled trial" in pts or KIND_TITLE["trial"].search(title):
        return "trial"
    # "A case report and literature review" is a case report, so cases go first.
    if any(k in pts for k in ("case report", "case-report", "case-study")) or KIND_TITLE["case"].search(title):
        return "case"
    if any(k in pts for k in ("review", "meta-analysis", "guideline")) or KIND_TITLE["review"].search(title):
        return "review"
    if any(k in pts for k in ("letter", "editorial", "comment")):
        return "letter"
    return "research"


def extras_from(r):
    """Additive metadata derived from a Europe PMC core record."""
    affs = [r.get("affiliation") or ""]
    for a in ((r.get("authorList") or {}).get("author") or []):
        for d in ((a.get("authorAffiliationDetailsList") or {}).get("authorAffiliation") or []):
            affs.append(d.get("affiliation") or "")
    pub_types = (r.get("pubTypeList") or {}).get("pubType") or []
    title = clean(r.get("title"))
    journal = clean(((r.get("journalInfo") or {}).get("journal") or {}).get("title"))
    return {
        "pub_types": pub_types,
        "kind": classify(title, pub_types),
        "stanford": any("stanford" in a.lower() for a in affs),
        "vet": bool(VET_RE.search(title) or re.search(r"veterinar", journal, re.I)),
    }


class RateLimited(RuntimeError):
    def __init__(self, msg, retry_after):
        super().__init__(msg)
        self.retry_after = retry_after


def strip_reasoning(text):
    """Remove chain-of-thought that reasoning models (gpt-oss, qwen, deepseek)
    sometimes emit inline in the answer: <think>...</think> blocks and OpenAI
    'harmony' channel markers (…analysis… assistantfinal <answer>). Provider-
    agnostic backstop for when reasoning_format:hidden isn't honored."""
    if not text:
        return text
    t = text
    # unclosed <think> (truncated before the model produced an answer): everything
    # from the tag onward is chain-of-thought that never closed → drop it (leaves
    # "" when the tag is at the very start, so the caller keeps the abstract).
    if re.search(r"<think>", t, flags=re.I) and not re.search(r"</think>", t, flags=re.I):
        t = re.split(r"<think>", t, flags=re.I)[0]
    if re.search(r"</think>", t, flags=re.I):        # keep only what follows the reasoning
        t = re.split(r"</think>", t, flags=re.I)[-1]
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S | re.I)
    t = re.sub(r"</?think>", "", t, flags=re.I)
    m = re.search(r"(?:assistantfinal|<\|channel\|>\s*final[^>]*<\|message\|>)\s*", t, flags=re.I)
    if m:                                            # take text after the final-answer marker
        t = t[m.end():]
    t = re.sub(r"<\|[^|]*\|>", "", t)                # drop any leftover harmony tags
    return t.strip()


def looks_reasoning(s):
    """True if text still carries reasoning-model chain-of-thought markers —
    used to detect and heal summaries stored before strip_reasoning existed."""
    return bool(re.search(r"<\s*/?\s*think|assistantfinal|<\|channel\|>", s or "", re.I))


def _oai_post(url, key, model, system, user, max_tokens, extra, body=None):
    """One OpenAI-compatible chat call. Raises with status+body on non-200 so
    failures (a decommissioned model, auth, rate limit) are diagnosable; a 429
    raises RateLimited carrying the server's suggested wait so the caller can
    back off instead of hammering."""
    h = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    h.update(extra or {})
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.3,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    payload.update(body or {})
    r = requests.post(url, headers=h, timeout=90, json=payload)
    if r.status_code == 429:
        wait = r.headers.get("retry-after")
        try:
            wait = float(wait)
        except (TypeError, ValueError):   # else parse "try again in 4.51s" from the body
            m = re.search(r"in ([\d.]+)s", r.text)
            wait = float(m.group(1)) if m else 5.0
        raise RateLimited(f"{model}@{url.split('/')[2]} -> 429", wait)
    if r.status_code != 200:
        raise RuntimeError(f"{model}@{url.split('/')[2]} -> {r.status_code} {r.text[:160]}")
    return strip_reasoning((r.json()["choices"][0]["message"]["content"] or "").strip())


def make_llm():
    """Return (name, complete) or None. Prefers Anthropic; otherwise builds an
    ordered candidate chain of (provider, model) pairs from every open-LLM
    provider whose key is set, and complete() tries them in turn — sticking with
    the last one that worked — so a dead model or a provider outage self-heals."""
    global LAST_PROVIDER
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()

            def complete(system, user, max_tokens):
                global LAST_PROVIDER
                m = client.messages.create(model=SUMMARY_MODEL, max_tokens=max_tokens,
                                           system=system,
                                           messages=[{"role": "user", "content": user}])
                LAST_PROVIDER = "claude"
                return "".join(b.text for b in m.content if b.type == "text").strip()
            return ("claude", complete)
        except ImportError:
            print("anthropic SDK not installed — trying open-LLM providers", file=sys.stderr)

    cands = []
    for prov in OPENLLM_PROVIDERS:
        key = os.environ.get(prov["key"])
        if not key:
            continue
        models = prov["models"]
        if prov["name"] == "groq" and os.environ.get("CTCL_GROQ_MODEL"):   # optional override, tried first
            ov = os.environ["CTCL_GROQ_MODEL"]
            models = [ov] + [m for m in models if m != ov]
        for m in models:
            cands.append((prov["name"], prov["url"], m, prov.get("headers") or {},
                          model_body(m), key))
    if not cands:
        return None

    state = {"i": 0}   # index of the candidate that last succeeded → tried first next time

    def complete(system, user, max_tokens):
        global LAST_PROVIDER
        n = len(cands)
        for attempt in range(6):          # sweep all candidates; back off on rate limits
            errs, waits = [], []
            for off in range(n):
                idx = (state["i"] + off) % n
                name, url, model, extra, body, key = cands[idx]
                try:
                    out = _oai_post(url, key, model, system, user, max_tokens, extra, body)
                    if out:
                        state["i"], LAST_PROVIDER = idx, name
                        return out
                    errs.append(f"{name}/{model}: empty response")
                except RateLimited as e:
                    waits.append(e.retry_after)
                    errs.append(str(e))
                except Exception as e:
                    errs.append(str(e))
            # every candidate was rate-limited → wait what the server asked (capped) and retry
            if waits and len(waits) == n and attempt < 5:
                time.sleep(min(max(waits), 30) + 0.5)
                continue
            break
        raise RuntimeError("all open-LLM candidates failed — " + " | ".join(errs[:4]))
    return ("openllm", complete)


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
        **extras_from(r),
        "match": r.get("_match") or "core",
    }


def load_data():
    if os.path.exists(DATA):
        with open(DATA) as f:
            return json.load(f)
    return {"papers": []}


def sort_key(p):
    return (p.get("date") or "", p.get("first_seen") or "")


def load_archive():
    if os.path.exists(ARCHIVE):
        with open(ARCHIVE) as f:
            return json.load(f)
    return {"papers": []}


def write_json(path, obj):
    """Write atomically so a crash mid-write can't leave a truncated file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def warn(msg):
    print(f"::warning::{msg}")          # shows on the Actions run summary
    print(msg, file=sys.stderr)


def main():
    data = load_data()
    papers = data.get("papers", [])
    try:
        archive = load_archive()
    except Exception as e:               # never risk the archive on a bad read
        warn(f"archive.json unreadable ({e}) — aborting, nothing written")
        return 0
    archived = archive.get("papers", [])
    archive_before = json.dumps(archived, sort_keys=True)
    known = papers + archived            # de-dupe against everything ever stored
    by_id = {p["id"]: p for p in known}
    seen_pmid = {p.get("pmid") for p in known if p.get("pmid")}
    seen_doi = {(p.get("doi") or "").lower() for p in known if p.get("doi")}
    seen_title = {norm_title(p.get("title")) for p in known}

    now = datetime.now(timezone.utc)
    try:
        candidates, coverage = fetch_candidates(now)
    except Exception as e:
        warn(f"Europe PMC fetch failed ({e.__class__.__name__}: {e}) — kept last-good data.json")
        return 0
    if coverage["core_hits"] < max(5, LOOKBACK_DAYS // 7):
        # ~10 core papers appear per week; far fewer means the query broke again.
        warn(f"only {coverage['core_hits']} core hits in {LOOKBACK_DAYS} days — query may be "
             "mis-parsed; kept last-good data.json")
        return 0

    llm = None if os.environ.get("CTCL_NO_LLM") else make_llm()
    complete = llm[1] if llm else None
    backend_is_open = bool(llm) and llm[0] == "openllm"   # pace only the free open-LLM tiers
    today = now.strftime("%Y-%m-%d")
    backfill = bool(os.environ.get("CTCL_BACKFILL"))
    new = []
    for r in candidates:
        rid = f"{r.get('source')}:{r.get('id')}"
        pmid = r.get("pmid")
        doi = (r.get("doi") or "").lower()
        if rid in by_id:
            # Already tracked: only ADD/refresh metadata (publication types arrive
            # weeks after a paper does), and pick up an abstract that arrived late.
            p = by_id[rid]
            p.update(extras_from(r))
            p.setdefault("match", r["_match"])
            if r["_match"] == "core":
                p["match"] = "core"
            if not p.get("abstract") and r.get("abstractText"):
                p["abstract"] = clean_abstract(r.get("abstractText"))
                if p.get("summary_by") in (None, "", "none"):
                    p["summary"], p["summary_by"] = trim(p["abstract"]), "abstract"
            continue
        if (pmid and pmid in seen_pmid) or (doi and doi in seen_doi):
            continue
        title = clean(r.get("title"))
        nt = norm_title(title)
        # PMC-only copies (and preprints) of a paper we already hold carry no
        # pmid/doi, so match them on title. A PubMed record is never dropped
        # this way — the published version of a preprint must still appear.
        if r.get("source") != "MED" and nt and nt in seen_title:
            continue
        abstract = clean_abstract(r.get("abstractText"))
        # Start every paper on the abstract-excerpt fallback; the upgrade pass
        # below promotes it to a real LLM summary when a key is available.
        summary, by = (trim(abstract), "abstract") if abstract else ("", "none")
        rec = record_from(r, summary, by, today)
        if backfill:
            rec["backfill"] = True       # found by a catch-up run, not "new this week"
        new.append(rec)
        by_id[rid] = rec
        seen_title.add(nt)
        if pmid:
            seen_pmid.add(pmid)
        if doi:
            seen_doi.add(doi)

    # One-time enrichment of papers stored before pub_types/kind/stanford existed
    # and now outside the look-back window. Best effort; retried next run if it fails.
    legacy = [p for p in known if "pub_types" not in p and p.get("pmid")]
    if legacy:
        try:
            for r in fetch_by_ids([p["pmid"] for p in legacy]):
                p = by_id.get(f"{r.get('source')}:{r.get('id')}")
                if p is not None and "pub_types" not in p:
                    p.update(extras_from(r))
        except Exception as e:
            print(f"legacy enrichment skipped ({e})", file=sys.stderr)
    for p in known:                      # anything still unclassified: title wording only
        if "kind" not in p:
            p["kind"] = classify(p.get("title"), p.get("pub_types"))

    allp = sorted(new + papers, key=sort_key, reverse=True)
    papers, overflow = allp[:KEEP_MAX], allp[KEEP_MAX:]

    # Self-heal any stored summary that leaked a reasoning model's chain-of-thought
    # (older runs before strip_reasoning): reset it to the abstract excerpt so the
    # upgrade pass below re-generates it cleanly — and if that fails, it stays a
    # clean excerpt rather than showing raw <think> text.
    healed = 0
    for p in papers:
        if p.get("summary_by") not in (None, "", "abstract", "none") and looks_reasoning(p.get("summary")):
            p["summary"] = trim(p["abstract"]) if p.get("abstract") else ""
            p["summary_by"] = "abstract" if p.get("abstract") else "none"
            healed += 1

    # Upgrade abstract-excerpt summaries to plain-language LLM summaries when a
    # key is set, newest first. Bounded three ways so a catch-up backlog or a
    # provider outage can't run the job for hours: a per-run cap, a wall-clock
    # budget, and a stop after 4 failures in a row. Whatever is left keeps its
    # abstract excerpt and is picked up by the next run.
    upgraded = 0
    pending = 0
    if complete:
        fails = streak = 0
        t0 = time.time()
        for p in papers:
            if p.get("summary_by") not in (None, "", "abstract", "none") or not p.get("abstract"):
                continue
            if upgraded >= MAX_SUMMARIES or streak >= 4 or time.time() - t0 > SUMMARY_BUDGET_S:
                pending += 1
                continue
            try:
                out = complete(SUMMARY_SYSTEM,
                               f"Title: {p['title']}\n\nAbstract: {p['abstract']}", 500)
                bad = ungrounded(out, p["title"] + " " + p["abstract"])
                if bad:
                    print(f"summary for {p['id']} names {bad} not in its abstract — kept excerpt",
                          file=sys.stderr)
                elif len(out) >= 20:               # don't overwrite the excerpt with an empty/truncated reply
                    p["summary"] = out
                    p["summary_by"] = LAST_PROVIDER or "ai"
                    upgraded += 1
                streak = 0
                if backend_is_open:
                    time.sleep(2)   # smooth the open-LLM free-tier tokens-per-minute cap
            except Exception as e:
                fails += 1
                streak += 1
                if fails <= 3:   # surface the real HTTP status/body a few times, not 50×
                    print(f"summary failed for {p['id']}: {e} — kept abstract", file=sys.stderr)
        print(f"{LAST_PROVIDER or 'no-provider'} summaries written this run: {upgraded}"
              + (f" ({healed} healed)" if healed else "")
              + (f" ({fails} failed)" if fails else "")
              + (f" ({pending} left for next run)" if pending else ""))

    # Rolling plain-language overview — regenerate when new papers arrived, any
    # summaries were (re)written this run, the stored overview itself leaked
    # reasoning, or none is stored. Needs a key; without one, the prior one is kept.
    overview_bad = looks_reasoning(" ".join((data.get("overview") or {}).get("bullets", [])))
    if complete and (new or upgraded or overview_bad or not data.get("overview")):
        try:
            pool = overview_pool(papers)
            bullets = build_overview(complete, pool)
            if bullets:
                data["overview"] = {
                    "bullets": bullets,
                    "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "n_papers": len(papers),
                    "n_used": len(pool),          # how many papers the bullets were drawn from
                    "by": LAST_PROVIDER or "ai",
                }
                print(f"overview regenerated: {len(bullets)} bullets")
        except Exception as e:
            print(f"overview failed ({e.__class__.__name__}: {e}) — keeping previous",
                  file=sys.stderr)

    # Papers pushed past KEEP_MAX are moved to archive.json, never dropped.
    # (Rewritten only when its contents actually changed.)
    if overflow:
        have = {p["id"] for p in archived}
        archived = sorted(archived + [p for p in overflow if p["id"] not in have],
                          key=sort_key, reverse=True)
    if json.dumps(archived, sort_keys=True) != archive_before:
        archive["papers"] = archived
        archive["updated"] = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        write_json(ARCHIVE, archive)

    data["title"] = "CTCL Literature Tracker"
    data["description"] = (
        "Latest peer-reviewed papers and preprints on cutaneous T-cell lymphoma, "
        "auto-collected and summarized."
    )
    data["query_human"] = QUERY_HUMAN
    data["source"] = "Europe PMC (PubMed / MEDLINE + preprints)"
    data["stats"] = {"total": len(papers), "new_this_run": len(new),
                     "archived": len(archived), "awaiting_summary": pending}
    data["coverage"] = dict(coverage, checked=today)
    data["updated"] = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    data["papers"] = papers

    write_json(DATA, data)
    # Small sibling of data.json (no abstracts/summaries) for at-a-glance
    # consumers such as the hub's preview card — same top-level shape.
    slim = {k: v for k, v in data.items() if k != "papers"}
    slim["papers"] = [{k: p.get(k) for k in ("title", "journal", "date", "first_seen",
                                             "kind", "stanford", "backfill")
                       if p.get(k) is not None} for p in papers]
    write_json(SUMMARY, slim)

    print(f"window={LOOKBACK_DAYS}d core_hits={coverage['core_hits']} "
          f"mention_hits={coverage['mention_hits']} new={len(new)} total={len(papers)} "
          f"archived={len(archived)} summaries={LAST_PROVIDER or 'abstract'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
