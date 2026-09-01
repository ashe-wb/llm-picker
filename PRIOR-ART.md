# Prior art

Tools that answer some version of "which local model runs on my machine." This file
exists so the question "hasn't this been built already?" has an honest answer in the
repo rather than in a comment thread.

It has been built already, many times, at one layer. What follows is where that layer
ends and what is left.

Surveyed August 2026: 27 named projects, of which four were checked by reading their
tool page or README directly and are marked **verified**. The rest are described from
their own documentation and may be stale. A capability listed as absent means it was not
found, not that it does not exist.

## Coverage

Ordered from commodity to rare. `·` partial.

| Capability | VRAM calculators | gpu_poor | ApXML | WhichLLM | Tier guides | llm-picker |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Weights fit in memory | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| KV cache from the model's own config | · | ✓ | ✓ | ✓ | | ✓ |
| Architecture-aware KV (hybrid, sliding, linear) | | | ✓ | | | ✓ |
| Tokens/sec from memory bandwidth | | ✓ | ✓ | ✓ | · | ✓ |
| Prompt processing as a separate axis | | · | ✓ | | | ✓ |
| Ranks models rather than filtering them | | | | ✓ | · | ✓ |
| Quality varies by quantization band | | | | · | · | ✓ |
| Apple bandwidth per chip, not per capacity | | | · | · | · | ✓ |
| **Known defects per band, with cited mitigations** | | | | | | ✓ |

## Memory calculators

The saturated layer. These answer one question well and stop.

**[ApXML VRAM Calculator](https://apxml.com/tools/vram-calculator)** — *verified.* The
strongest of these by some margin. Calibrates KV scaling for hybrid attention,
sliding-window and linear attention, and DeepSeek Sparse Attention — broader
architecture coverage than this project has. Also reports TTFT, throughput, concurrent
users, CPU/NVMe offload and rental cost, and covers Apple Silicon. Carries no
quality-loss modelling.

**[gpu_poor](https://github.com/RahulSChand/gpu_poor)** — *verified.* Memory breakdown
plus tokens/sec, and it names whether a configuration is memory-bound or compute-bound,
which is the distinction underneath our generation-versus-prefill split. Hardcoded
configs for roughly the top 3,000 models on Hugging Face. GGML, bitsandbytes int8 and
QLoRA. Quantization is treated as a divisor on size.

**[NyxKrage · LLM Model VRAM Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)**
— the reference implementation most others descend from. Near-identical Spaces exist
under Nymbo, dhlak, ThatsGroes and DavidAU, and a philip6369 collection curates more.

**[ModelFit](https://modelfit.io/calculator/)** — browser-only, no signup, and its
compatibility dataset is published CC BY 4.0. The only other tool here that licenses
its data for reuse.

Also: [Spheron GPU Recommender](https://www.spheron.network/tools/gpu-recommender/)
(cheapest card per precision), [TitaniumMonkey's
calculator](https://github.com/TitaniumMonkey/LLM_Hardware_Calculator) (live Hugging
Face fetch), [smcleod's estimator](https://smcleod.net/vram-estimator/), [Logarithmic
Spirals'
planner](https://logarithmicspirals.com/tools/local-ai-vram-calculator-gpu-planner/),
and Hugging Face's own Accelerate memory estimator.

## Model recommenders

Tools that rank rather than filter. This is the layer this project occupies.

**[WhichLLM](https://github.com/Andyyyy64/whichllm)** — *verified.* The closest thing
to this project, and good. Auto-detects NVIDIA, AMD, Intel, Apple Silicon and CPU.
Merges LiveBench, Artificial Analysis, Aider, Open LLM Leaderboard v2, Chatbot Arena
ELO and a curated vision index, with recency-aware demotion so stale leaderboards stop
propping up superseded models. Applies quantization penalties and grades each score's
provenance across five confidence levels — `direct`, `variant`, `base_model`,
`line_interp`, `self_reported` — discounting the weaker ones. Estimates tokens/sec from
bandwidth, MoE-aware on active versus total parameters. Has task profiles. Its repo
description makes the same argument this site does: rank by benchmarks, "not parameter
count."

**[Local AI Master](https://localaimaster.com/tools)** — a suite rather than one tool:
model recommender, VRAM calculator, GPU picker, quantization calculator, leaderboard.

**[LLM Picker Pro](https://github.com/Joseph-elias/Llm-Picker-Extension)** — *verified.*
Shares this project's name and answers a different question with it. A Chrome extension
that classifies your prompt with Gemini, then recommends a model from a Hugging Face
Open LLM Leaderboard CSV, filtered by weight type, licence, architecture and parameter
count. It does not consider hardware at all — no memory fit, no quantization, no speed —
so it will happily suggest a model you cannot load. Needs a local Flask backend and a
Gemini API key to run.

**[Will It Run AI](https://willitrunai.com/blog/what-llm-can-i-run-locally)** —
calculator plus editorial ranking by VRAM tier. The tier-guide format is the dominant
shape of advice in this space, and the thing methodology §3 replaced with a computed
budget.

## Guides rather than tools

A large body of prose competing for the same readers without being software:
[Hardware Corner](https://www.hardware-corner.net/memory-bandwidth-llm-speed/) on
bandwidth and speed, the r/LocalLLaMA wiki,
[InsiderLLM](https://insiderllm.com/guides/best-local-llms-mac-2026/),
[LLMCheck](https://llmcheck.net/benchmarks), HybridLLM, Presenc AI, and many "best
model for 8/16/24GB" posts.

Several of these publish *measured* tokens/sec on real hardware. This site models speed
from published bandwidth specs and says so on every card. Those measurements are where
calibration evidence should come from — see methodology §4.

## On the name

"LLM Picker" is taken at least three times on GitHub, in three unrelated senses:

- **[Joseph-elias/Llm-Picker-Extension](https://github.com/Joseph-elias/Llm-Picker-Extension)**
  — task-to-model recommender, described above. The only one doing adjacent work.
- **[SimonPurdie/LLMPicker](https://github.com/SimonPurdie/LLMPicker)** — a Firefox
  extension for switching between hosted chat sites. A launcher, not a picker in this
  sense.
- **[rachedblili/LLMPicker](https://github.com/rachedblili/LLMPicker)** — a multi-provider
  chat front-end for OpenAI, Anthropic, Gemini, Bedrock and Groq.

All three had no stars at the time of writing, and none competes with this site on
substance. Recorded here so anyone who arrives via the wrong search knows which project
they have found: this one is about whether a model fits *your machine*, and what breaks
when you shrink it to make it fit.

## Where this project is not different

Stated plainly, because a prior-art file that only lists advantages is marketing.

- **Architecture-aware KV cache.** ApXML covers hybrid, sliding-window, linear *and*
  DeepSeek Sparse Attention. We do not model DSA.
- **Bandwidth-derived tokens/sec.** Four tools above do this. Some published derates sit
  at 60–70% of theoretical against our 45–70%; ours may be too conservative.
- **Refusing to rank by size.** WhichLLM makes the same argument, in its repo
  description.
- **Hardware detection.** WhichLLM reads the machine. We ask the reader to describe it.

## Where it is

- **The weakness → mitigation matrix.** Nothing found records that a named model's
  tool-calling breaks at a named band, cites two sources for it, and gives a fix or
  states that none exists. Searching for such a database returns quantization
  explainers instead. This is the part that cannot be derived from a formula, and it is
  the reason the site exists.
- **Per-band scores as evidence rather than arithmetic.** WhichLLM discounts low-bit
  quants with a penalty applied uniformly. We record a score per model per band with
  provenance. That difference is why the picker could find that FP16 buys nothing over
  Q6–Q8 on 16 of the 17 models scored at both — a uniform multiplier cannot surface
  that, because it assumes the answer.

## Corrections

If a tool here is described wrongly, or one is missing, open an issue. Several entries
are characterised from their own documentation rather than from use, and that is a
weaker standard than this project applies to its own data.
