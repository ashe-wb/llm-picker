# llm-picker

**What LLM your machine can run — and which one is a good fit for a specific job.**

Live at **[llm-picker.dev](https://llm-picker.dev)**.

Leaderboards rank models at full precision on hardware you do not have. This site
starts from your machine instead: it works out which models fit, then which of those
suit the work, then tells you what each one is bad at and what people do about it.

## What it does

**1. Works out what fits.** You give it your memory and your workload. It computes
weights plus the real KV cache at the context that workload needs, against the memory
your GPU is actually allowed to use — on Apple Silicon macOS caps that near 75% of
RAM, so a 32GB Mac has about 24GB, not 32. That cap can be raised and the site says how — while
being clear that Apple documents the setting nowhere, so no figure for how far is
authoritative, this project's included. Verdicts are *comfortable*, *tight*, or
*won't fit*, because "runs at 8K but not at 128K" is a useful answer with a condition
attached.

**2. Ranks what fits.** Scores are per quantization band — Q2–Q3, Q4–Q5, Q6–Q8, FP16 —
because published benchmarks measure full precision and the build you download is
compressed. Unrated dimensions count as average rather than being skipped, so a model
measured on a tenth of a task cannot outrank one measured on all of it.

**3. Says what breaks.** Every model carries its known issues at the band you would
run, with a severity, cited sources, and a workaround — or the honest admission that
there isn't one. **29 of 90 known-issue entries have no known fix**, and say so.

## What's in it

| | |
|---|---|
| Models | 41, from 13 vendors, 3B–397B |
| Scores | 353 cells, each with source, metric, raw value, retrieval date and the precision it was measured at |
| Known issues | 90 entries, citation minimums enforced by CI |
| Glossary | 64 terms — every quantization format, benchmark and piece of jargon the site uses |

**Nothing here has been spot-checked on our own hardware yet.** Every figure is
vendor-reported or aggregated from a public benchmark, and the site says so on every
recommendation rather than implying otherwise.

## How it works

- **All content is JSON in [`/data`](data/)** — models, per-band scores with
  provenance, weakness→mitigation entries with citations. You never touch site code
  to contribute; see [CONTRIBUTING.md](CONTRIBUTING.md).
- Scores are **editorial 0–10 judgments over cited public evidence** (LiveBench,
  Aider, BFCL, RULER, Vectara HHEM, EQ-Bench, vendor cards). Cells with no evidence
  stay blank — blank is honest, a guessed number is not.
- Memory figures are computed, not estimated: KV cache comes from each model's own
  `config.json`, which matters because a hybrid model may cache 6 of its 52 layers.
- `npm run check` enforces the schema and the integrity rules — citation minimums,
  licensing posture, weakness coverage, and that every quantization format the
  vocabulary accepts has a glossary definition. CI and the deploy both run it.
- Updated weekly: an automated pipeline proposes score refreshes as a PR; a human
  reviews and merges.

## Develop

```sh
npm install
npm run dev       # local site
npm run check     # validate data + tests + build
```

Static Astro build, hosted on Cloudflare Pages.

## Prior art

"Hasn't this been built already?" — yes, many times, at one layer.
[PRIOR-ART.md](PRIOR-ART.md) surveys 27 existing calculators and recommenders —
including three unrelated projects that share this one's name, and the places where
this project is *not* the first to do something.

## Licenses

Code: [MIT](LICENSE). Data (`/data`): [CC BY 4.0](DATA_LICENSE).
Benchmark numbers remain the work of their cited sources. We deliberately do not
ingest data from sources whose terms forbid republication (e.g. Artificial
Analysis is cited link-only).

`data/tensor-shapes.json` records tensor names and shapes read from each model's
safetensors header — facts about published files, containing no weights and no
model code. The models themselves are under their own terms (Apache-2.0, MIT,
Gemma, Nemotron and the Llama Community licenses); every model entry links its
own. See [DATA_LICENSE](DATA_LICENSE).

No third-party source code is redistributed: the browser is served only this
project's own JS, CSS and HTML, and every dependency stays on the build machine.
[NOTICE](NOTICE) names the works consulted to establish how models behave —
mlx-lm and transformers, both Apache-2.0 — which is a citation rather than an
obligation, since what was taken is behaviour and not code.
[CITATION.cff](CITATION.cff) is there if you need to cite this.
