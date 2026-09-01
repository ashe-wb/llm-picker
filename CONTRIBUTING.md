# Contributing

All content lives in `/data` as JSON. You never need to touch site code to contribute.

## Ground rules

1. **Every score needs provenance.** A score entry without a source, URL, and
   retrieval date will fail CI.
2. **Every weakness needs citations.** `community-consensus` requires ≥2,
   `anecdotal` ≥1. Link the actual thread/leaderboard, not a vibe.
3. **Record the exact quant.** Measurements carry the precision they were taken
   at (`Q4_K_M`, `bf16`, …). Unknown quant strings fail validation. A legitimate
   new format needs two things: an entry in `QUANT_VOCAB` (`src/lib/schemas.ts`)
   *and* a definition in `data/glossary.json` covering it — the build fails until
   both exist, so the vocabulary can never outrun its own documentation.
4. **Respect source licenses.** Sources marked `republishOk: false` in
   `data/sources.json` (e.g. Artificial Analysis) may be cited by link only —
   never copy their numbers into `rawValue`.
5. **Honesty over completeness.** Missing cells render as "not yet rated" —
   that is better than a guessed score. If a weakness has no real fix, use the
   `none` mitigation.

## Adding or updating data

- New model: add `data/models/<id>.json`, `data/scores/<id>.json`,
  `data/weaknesses/<id>.json` (see existing files for shape).
- **Give the model a `kvBytesPerToken`.** Compute it from the model's own
  `config.json` — `2 x fullAttentionLayers x num_key_value_heads x head_dim x 2` —
  and link that file in `kvSourceUrl`. Only layers that actually hold a KV cache
  count: hybrid models cache a fraction of their layers, and sliding-window layers
  need the separate `kvWindowedBytesPerToken` and `slidingWindow` fields. Without
  this figure the model is never recommended, because the picker refuses to guess
  how much memory it needs.
- `weightsGb` per band is **weights only** — the KV cache is computed, not baked in.
- Run `npm run check` before opening the PR (validate + tests + build); CI runs the
  same gate, and so does the deploy.

## Scope

**Open-weight dense and MoE models up to and including 400B total parameters**, that
run on a machine you can own. There is no cap on how many — the catalogue currently
holds 41. The ceiling was 35B until 2026-08-30, 70B until 2026-08-31.

The 35–70B band really is nearly empty, and raising the ceiling to 70B added one
model rather than a category. 70–400B is the opposite: the field went there in force,
and every model in that range is a sparse MoE. Note what that means for the memory
model — total parameters set what you must hold, active parameters set how fast it
runs, so both belong in the file. `activeParamsB` is required for `architecture:
"moe"` and validate.ts enforces it.

Two rules decide whether a model belongs:

- **Within a family, only the current generation gets a sheet.** Granite 4.2
  replaced 4.0 and 4.1, so those are gone; later Qwen releases cover the ground
  Qwen2.5-Coder 32B once did. A catalogue that keeps every generation becomes an
  archive, and the question here is what to download tonight.
- **Across families, nothing is dropped for being outscored.** A model keeps its
  slot if it is the best answer for some machine or some workload, even when a
  rival beats it overall.

Both rules are stated publicly in methodology §1 — keep that section in sync.

Not in scope: **1-bit and ternary builds.** The bottom band is Q2–Q3. Models that
exist only at those widths are not rated here.
