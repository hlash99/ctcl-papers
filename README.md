# ctcl-papers

Auto-updating literature tracker for **CTCL** (cutaneous T-cell lymphoma) — live
at **https://hlash99.github.io/ctcl-papers/** and linked from the
[hlash99 dashboard](https://hlash99.github.io/).

It polls the published scientific literature, summarizes each new paper in plain
language, and keeps an accumulating, newest-first list. Same publish pattern as
`asset-tracker` and `iran-r32-tracker`: a static page (`index.html`) renders a
JSON file (`data.json`) that a scheduled GitHub Action keeps fresh.

## How it works

| Piece | What it does |
|-------|--------------|
| `scripts/fetch_papers.py` | Reads **every** CTCL paper **Europe PMC** (PubMed/MEDLINE + preprints) lists for the last 120 days (cursor-paginated, not just a "top 50"), de-dupes against `data.json` + `archive.json`, tags each paper (trial / review / case / letter / research, Stanford-affiliated, veterinary), summarizes each *new* one, and writes `data.json`. |
| `data.json` | The newest 400 papers. Older ones move to `archive.json` (loaded on demand by the page) — nothing is dropped. |
| `index.html` | Zero-build page (vanilla JS) that renders `data.json` with search + filters. |
| `.github/workflows/refresh.yml` | The "server side": runs **weekly**, commits `data.json` only when there are new papers. |

The search scope is title/keyword/MeSH matching on *cutaneous T-cell lymphoma*,
*mycosis fungoides*, *Sézary syndrome*, and *CTCL* — so results are about the
disease. Papers that only name the disease in the abstract are kept too, flagged
`"match": "mention"`.

Every query clause is individually parenthesised. That matters: Europe PMC
mis-parses a bare `A OR B OR MESH:"Lymphoma, T-Cell, Cutaneous"` chain (the
comma), which made the original query silently return ~1 in 8 of the matching
papers. The script now refuses to write if the window returns implausibly few
hits, and the page shows a warning if the last refresh is more than 10 days old.

Paper types come from PubMed publication types where MEDLINE has assigned them
and from title wording otherwise, so a brand-new trial may show as "research"
until it is indexed.

## Summaries

- **With an `ANTHROPIC_API_KEY`** repo secret, each new paper gets a 2–3 sentence
  plain-language summary from Claude (`claude-opus-4-8` by default).
- **Without a key**, it falls back to a trimmed excerpt of the abstract, so the
  page still works. Existing papers are never re-summarized — a weekly run only
  spends tokens on what's new.
- Override the model with the `CTCL_SUMMARY_MODEL` env var (e.g.
  `claude-haiku-4-5` for lower cost on high volume).

## Cadence

Weekly (Mondays). To poll **daily** instead, change the cron in
`.github/workflows/refresh.yml` to `'17 13 * * *'`.

## Run it locally

```sh
pip install requests anthropic         # 'anthropic' optional — only for AI summaries
export ANTHROPIC_API_KEY=sk-ant-...     # optional
python3 scripts/fetch_papers.py
```
