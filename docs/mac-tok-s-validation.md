# Mac tok/s validation — llama.cpp and MLX

Every published Apple Silicon generation-speed measurement we could attribute to a
named chip, scored against what the site predicts for the same chip, model, quantization
band and context. Collected 2026-09-01. Decode tok/s only; "effective" wall-clock figures
that include prefill are excluded, as is anything whose chip is not stated.

## Result

92 measurements. The site's range brackets **14**; **75 sit above it**; 3 below. The site is
systematically pessimistic on Macs, for both runtimes, typically by 1.3–1.8×.

The cause is the unified-memory per-layer overhead, `perTokenLatency.byPlatform.unified` =
0.21–0.45 ms/layer. Solving each measurement for the overhead it implies at peak bandwidth
(`ms/layer` column) gives:

| runtime | dense | mixture-of-experts |
|---|---|---|
| llama.cpp, 2023 build (discussion #4167, commit 8e672ef) | 0.12–0.22 | — |
| llama.cpp, 2026 builds (M4/M5) | 0.05–0.15 | 0.10–0.31 |
| MLX | 0.02–0.13 | 0.09–0.30 |

The 0.21–0.45 figure was fitted to Llama-3.3-70B on an M2 Ultra including a Q2_K row, and
Q2_K is compute-bound in dequantisation: it implies 0.3 ms/layer while Q4_K_M and F16 on the
same machine imply 0.16–0.21. The fit also assumed peak bandwidth; a two-parameter fit of the
#4167 7B rows gives effective bandwidth at 86–105% of peak with a fixed 4–5 ms/token on
M1–M4 and 0.5–2.6 ms on M5, i.e. the overhead is smaller than assumed and largely a
software-version effect (the M5 rows are the only ones measured with a 2026 llama.cpp).

The overhead is not one number. Mixture-of-experts routing costs 2–3× what a dense layer
costs, and older chips pay more of it (0.27–0.31 ms/layer on an M1 Max / M3 Max, 0.10–0.15
on M4/M5). Proposed: widen the unified range to **0.05–0.30 ms/layer**, which brackets 76 of
92 (llama.cpp 42/48, MLX 34/44). What still falls outside:

- above: dense MLX on M3 Ultra, M4 Max and M4 Pro reaching 0.02–0.04 ms/layer (measured up
  to 10% past the new high end);
- below: a 15K-token prompt on an M1 Pro (attention over a long cache, both runtimes), and one
  LM Studio GGUF run of Qwen3.5 (29 tok/s) from before llama.cpp optimised that architecture.

## Corrections to existing citations

- The site quotes the antekapetanovic M4 Max benchmark as "66–70 tok/s" for llama.cpp and
  "85–108" for MLX. The page gives llama.cpp Q4_K_XL **70.4–72.4**, MLX native **126.4–131.8**,
  MLX HTTP server 84.5–107.6, Ollama 41.8–48.1. The MLX-native ratio is 1.8×, not 1.5×; the
  1.5× was the HTTP-server rows against a misquoted floor.
- MLX weights are smaller than the site's Q4 band: MLX 4-bit is 4.5 bits/weight exactly
  (bits + 32/group_size), Q4_K_M measures 4.82–4.91 on disk. Qwen3-32B is 18.4GB in MLX
  4-bit against the site's 20GB; two dense MLX rows on an M3 Max exceed 100% of peak bandwidth
  for exactly this reason.

## Same-machine MLX vs llama.cpp, matched 4-bit

| chip | model | MLX | llama.cpp | ratio | source |
|---|---|---:|---:|---:|---|
| M4 Max 128GB | Qwen3.5-35B-A3B | 130.2 | 72.4 | 1.80 | antekapetanovic.com |
| M4 Max 128GB | Qwen3-4B | 128.9 | 118.2 | 1.09 | arxiv 2601.19139 |
| M4 Max 128GB | Qwen3-8B | 79.9 | 76.9 | 1.04 | arxiv 2601.19139 |
| M4 Max 128GB | Qwen3-30B-A3B | 107.4 | 89.9 | 1.19 | arxiv 2601.19139 |
| M4 Max 128GB | Llama-3.2-3B | 167.5 | 155.8 | 1.08 | arxiv 2601.19139 |
| M4 Max 64GB | Qwen3-4B | 170 | 87 | 1.95 | runanywhere.ai |
| M4 Pro | Qwen3-30B-A3B | 83.1 | 80.7 | 1.03 | arxiv 2607.00501 |
| M4 Pro | Llama-3.2-3B | 112.1 | 102.4 | 1.09 | arxiv 2607.00501 |
| M3 Max 64GB | Qwen3.5-35B-A3B | 95 | 58.4 | 1.63 | hiesch.eu |
| M3 Max 64GB | Gemma 4 26B-A4B | 65.0 | 63.9 | 1.02 | hiesch.eu |
| M3 Max 64GB | Qwen3-32B | 19 | 17.4 | 1.09 | hiesch.eu |
| M3 Max 64GB | Qwen3.6-27B | 20.3 | 17.3 | 1.17 | hiesch.eu |
| M1 Max 64GB | Qwen3 30B-A3B | 55.5 | 58 | 0.96 | famstack.dev |
| M1 Pro 32GB | gpt-oss-20b (15K prompt) | 33.85 | 37.05 | 0.91 | mlx-lm #858 |
| M5 Max 128GB | Gemma 4 26B-A4B | 125.2 | 90.2 | 1.39 | haxlys/llm-bench |
| M5 Max 128GB | Gemma 4 31B (8-bit) | 14.07 | 13.63 | 1.03 | haxlys/llm-bench |
| M1 Ultra 128GB | Qwen3.5-27B distil | 28.9 | 19.9 | 1.45 | HF ronnie3786 card |
| M5 Pro 48GB | AMALIA-9B | 55–59 | 48 | 1.19 | HF teex-pt card |

MLX's advantage is 1.0–1.2× on most models and 1.4–1.9× on GatedDeltaNet (Qwen3.5/3.6)
and on small dense models, where llama.cpp's fixed overhead is a larger share of the token.
It is not a constant, so the "read the speed as a floor" note is directionally right and
numerically wrong.

## Every measurement

`GB read` is active weights plus KV at the stated context, as the site computes it.
`ms/layer` is the per-layer overhead the measurement implies at peak bandwidth. Proposed =
unified 0.05–0.30 ms/layer.

| chip | model | GB read | ctx | runtime | measured | site lo–hi | | proposed lo–hi | | ms/layer | source |
|---|---|---:|---:|---|---:|---:|:--|---:|:--|---:|---|
| m1-max | llama-3.1-8b | 3.8 | 128 | llama.cpp | 61.19 | 40–59 | ↑ above | 50–84 | ok | 0.188 | #4167 L2-7B Q4_0 |
| m1-max | llama-3.1-8b | 7.2 | 128 | llama.cpp | 40.2 | 30–38 | ↑ above | 35–48 | ok | 0.175 | #4167 L2-7B Q8_0 |
| m1-max | llama-3.1-8b | 13.5 | 128 | llama.cpp | 23.03 | 20–23 | ↑ above | 22–26 | ok | 0.223 | #4167 L2-7B F16 |
| m1-ultra | llama-3.1-8b | 3.8 | 128 | llama.cpp | 83.73 | 51–84 | ok | 68–148 | ok | 0.212 | #4167 L2-7B Q4_0 |
| m2-max | llama-3.1-8b | 3.8 | 128 | llama.cpp | 65.95 | 40–59 | ↑ above | 50–84 | ok | 0.151 | #4167 L2-7B Q4_0 |
| m2-ultra | llama-3.1-8b | 3.8 | 128 | llama.cpp | 94.27 | 51–84 | ↑ above | 68–148 | ok | 0.170 | #4167 L2-7B Q4_0 |
| m2-ultra | llama-3.1-8b | 13.5 | 128 | llama.cpp | 41.02 | 31–40 | ↑ above | 36–51 | ok | 0.195 | #4167 L2-7B F16 |
| m3-max-30c | llama-3.1-8b | 3.8 | 128 | llama.cpp | 56.58 | 36–49 | ↑ above | 43–65 | ok | 0.122 | #4167 L2-7B Q4_0 |
| m3-max-40c | llama-3.1-8b | 3.8 | 128 | llama.cpp | 66.31 | 40–59 | ↑ above | 50–84 | ok | 0.149 | #4167 L2-7B Q4_0 |
| m3-ultra | llama-3.1-8b | 3.8 | 128 | llama.cpp | 92.14 | 51–85 | ↑ above | 68–151 | ok | 0.182 | #4167 L2-7B Q4_0 |
| m4-pro | llama-3.1-8b | 3.8 | 128 | llama.cpp | 50.74 | 34–46 | ↑ above | 40–60 | ok | 0.143 | #4167 L2-7B Q4_0 |
| m4-max-32c | llama-3.1-8b | 3.8 | 128 | llama.cpp | 69.95 | 41–60 | ↑ above | 51–86 | ok | 0.132 | #4167 L2-7B Q4_0 |
| m4-max-40c | llama-3.1-8b | 3.8 | 128 | llama.cpp | 83.06 | 46–70 | ↑ above | 58–109 | ok | 0.140 | #4167 L2-7B Q4_0 |
| m4-max-40c | llama-3.1-8b | 13.5 | 128 | llama.cpp | 31.64 | 24–30 | ↑ above | 28–35 | ok | 0.157 | #4167 L2-7B F16 |
| m5 | llama-3.1-8b | 3.8 | 128 | llama.cpp | 31.88 | 24–30 | ↑ above | 27–35 | ok | 0.137 | #4167 L2-7B Q4_0 |
| m5-pro | llama-3.1-8b | 3.8 | 128 | llama.cpp | 66.33 | 36–50 | ↑ above | 43–66 | ok | 0.051 | #4167 L2-7B Q4_0 |
| m5-max-40c | llama-3.1-8b | 3.8 | 128 | llama.cpp | 119.92 | 47–74 | ↑ above | 61–120 | ok | 0.050 | #4167 L2-7B Q4_0 |
| m5-max-40c | llama-3.1-8b | 13.5 | 128 | llama.cpp | 37.11 | 26–33 | ↑ above | 30–40 | ok | 0.104 | #4167 L2-7B F16 |
| m2-ultra | llama-3.3-70b | 42.7 | 512 | llama.cpp | 13.5 | 11–14 | ok | 12–16 | ok | 0.210 | site note L3.3-70B Q4_K_M |
| m2-ultra | llama-3.3-70b | 141.3 | 512 | llama.cpp | 4.87 | 4.4–4.8 | ↑ above | 5–5 | ok | 0.197 | site note L3.3-70B F16 |
| m4-max-40c | qwen3.6-35b-a3b | 2.0 | 4096 | llama.cpp | 72.4 | 46–82 | ok | 63–171 | ok | 0.249 | antekapetanovic Q4_K_XL |
| m3-max-40c | gemma-4-e2b | 1.5 | 1263 | llama.cpp | 108.3 | 51–89 | ↑ above | 69–177 | ok | 0.152 | hiesch b9020 Q4_K_M |
| m3-max-40c | gemma-4-e4b | 2.9 | 1263 | llama.cpp | 72.9 | 38–61 | ↑ above | 49–103 | ok | 0.144 | hiesch b9020 Q4_K_M |
| m3-max-40c | gemma-4-26b-a4b | 2.3 | 1263 | llama.cpp | 63.9 | 51–80 | ok | 66–129 | ↓ below | 0.313 | hiesch b9020 Q4_K_M |
| m3-max-40c | qwen3.6-35b-a3b | 1.9 | 1263 | llama.cpp | 58.4 | 43–74 | ok | 58–140 | ok | 0.300 | hiesch b9020 Q4_K_M |
| m3-max-40c | qwen3-32b | 20.3 | 1263 | llama.cpp | 17.4 | 12–15 | ↑ above | 14–17 | ↑ above | 0.046 | hiesch b9020 Q4_K_M |
| m3-max-40c | qwen3.6-27b | 17.1 | 1263 | llama.cpp | 17.3 | 13–17 | ↑ above | 15–20 | ok | 0.187 | hiesch b9020 Q4_K_M |
| m3-max-40c | gemma-4-31b | 18.9 | 1263 | llama.cpp | 16.5 | 13–16 | ↑ above | 15–19 | ok | 0.166 | hiesch b9020 Q4_K_M |
| m5-max-40c | gemma-4-26b-a4b | 2.2 | 384 | llama.cpp | 90.16 | 58–99 | ok | 78–188 | ok | 0.242 | haxlys Q4_K_M |
| m5-max-40c | gemma-4-31b | 31.1 | 384 | llama.cpp | 13.63 | 12–15 | ok | 14–17 | ↓ below | 0.316 | haxlys Q8_0 |
| m5-max-40c | gpt-oss-20b | 2.1 | 1152 | llama.cpp | 113.2 | 69–115 | ok | 92–206 | ok | 0.215 | haxlys Q4_K_M |
| m5-max-40c | gpt-oss-120b | 2.8 | 1152 | llama.cpp | 86.65 | 47–80 | ↑ above | 64–149 | ok | 0.184 | haxlys Q4_K_M |
| m5-max-40c | qwen3.6-35b-a3b | 1.9 | 1152 | llama.cpp | 91.39 | 47–85 | ↑ above | 65–187 | ok | 0.190 | haxlys Q4_K_M |
| m5-max-40c | qwen3.5-9b | 9.1 | 1152 | llama.cpp | 48.35 | 33–44 | ↑ above | 39–57 | ok | 0.147 | haxlys Q8_0 |
| m5-max-40c | gemma-4-e4b | 4.5 | 1152 | llama.cpp | 76.05 | 37–60 | ↑ above | 49–101 | ok | 0.127 | haxlys Q8_0 |
| m5-max-40c | qwen3-30b-a3b | 1.9 | 1152 | llama.cpp | 112.1 | 40–74 | ↑ above | 56–172 | ok | 0.115 | haxlys Q4_K_M (coder-30B-A3B) |
| m4-max-40c | qwen3-4b | 2.6 | 512 | llama.cpp | 118.2 | 47–79 | ↑ above | 63–146 | ok | 0.095 | arxiv 2601.19139 |
| m4-max-40c | qwen3-8b | 5.1 | 512 | llama.cpp | 76.9 | 38–57 | ↑ above | 48–85 | ok | 0.084 | arxiv 2601.19139 |
| m4-max-40c | qwen3-30b-a3b | 1.9 | 512 | llama.cpp | 89.9 | 40–73 | ↑ above | 55–164 | ok | 0.154 | arxiv 2601.19139 |
| m4-max-40c | llama-3.2-3b | 2.2 | 512 | llama.cpp | 155.8 | 59–99 | ↑ above | 79–177 | ok | 0.078 | arxiv 2601.19139 |
| m4-max-40c | gemma-3-4b | 2.6 | 512 | llama.cpp | 123.2 | 49–82 | ↑ above | 66–148 | ok | 0.090 | arxiv 2601.19139 |
| m4-pro | qwen3-30b-a3b | 1.9 | 512 | llama.cpp | 80.7 | 34–57 | ↑ above | 46–102 | ok | 0.104 | BaseRT arxiv 2607.00501 |
| m4-pro | llama-3.2-3b | 2.2 | 512 | llama.cpp | 102.4 | 47–70 | ↑ above | 59–101 | ↑ above | 0.046 | BaseRT arxiv 2607.00501 |
| m4-max-40c | qwen3-4b | 2.6 | 512 | llama.cpp | 87 | 47–79 | ↑ above | 63–146 | ok | 0.179 | runanywhere.ai |
| m4-max-40c | llama-3.2-3b | 2.2 | 512 | llama.cpp | 137 | 59–99 | ↑ above | 79–177 | ok | 0.109 | runanywhere.ai |
| m1-pro | gpt-oss-20b | 2.4 | 15000 | llama.cpp | 37.05 | 42–56 | ↓ below | 50–71 | ↓ below | 0.585 | mlx-lm #858 (14.7K prompt) |
| m4-max-40c | qwen3-4b | 2.8 | 2176 | MLX | 134.52 | 46–77 | ↑ above | 61–137 | ok | 0.054 | BENCHMARKS.md q4 |
| m4-max-40c | qwen3-4b | 4.6 | 2176 | MLX | 86.91 | 40–60 | ↑ above | 50–92 | ok | 0.068 | BENCHMARKS.md q8 |
| m4-max-40c | qwen3-4b | 8.3 | 2176 | MLX | 52.47 | 31–42 | ↑ above | 37–55 | ok | 0.076 | BENCHMARKS.md bf16 |
| m4-max-40c | qwen3-30b-a3b | 2.2 | 2176 | MLX | 113.33 | 39–70 | ↑ above | 54–150 | ok | 0.095 | BENCHMARKS.md q4 |
| m4-max-40c | qwen3-30b-a3b | 3.8 | 2176 | MLX | 83.16 | 34–57 | ↑ above | 46–101 | ok | 0.094 | BENCHMARKS.md q8 |
| m3-pro | qwen3-4b | 2.8 | 2176 | MLX | 44.62 | 28–36 | ↑ above | 32–46 | ok | 0.066 | mlx-lm PR #1569 |
| m5-max-40c | gemma-4-26b-a4b | 2.3 | 364 | MLX | 125.16 | 57–96 | ↑ above | 77–180 | ok | 0.131 | haxlys 4bit |
| m5-max-40c | gemma-4-26b-a4b | 4.1 | 364 | MLX | 92.3 | 48–74 | ↑ above | 62–114 | ok | 0.119 | haxlys 8bit |
| m5-max-40c | gemma-4-31b | 33.3 | 364 | MLX | 14.07 | 12–14 | ↑ above | 13–16 | ok | 0.214 | haxlys 8bit |
| m1-pro | gpt-oss-20b | 2.4 | 15000 | MLX | 33.85 | 42–56 | ↓ below | 50–71 | ↓ below | 0.691 | mlx-lm #858 MXFP4 |
| m4-max-40c | qwen3-4b | 2.6 | 512 | MLX | 128.9 | 47–79 | ↑ above | 63–146 | ok | 0.075 | arxiv 2601.19139 mlx-lm |
| m4-max-40c | qwen3-8b | 5.1 | 512 | MLX | 79.9 | 38–57 | ↑ above | 48–85 | ok | 0.071 | arxiv 2601.19139 mlx-lm |
| m4-max-40c | qwen3-30b-a3b | 1.9 | 512 | MLX | 107.4 | 40–73 | ↑ above | 55–164 | ok | 0.117 | arxiv 2601.19139 mlx-lm |
| m4-max-40c | llama-3.2-3b | 2.2 | 512 | MLX | 167.5 | 59–99 | ↑ above | 79–177 | ok | 0.062 | arxiv 2601.19139 mlx-lm |
| m4-max-40c | gemma-3-4b | 2.6 | 512 | MLX | 105.4 | 49–82 | ↑ above | 66–148 | ok | 0.131 | arxiv 2601.19139 mlx-lm |
| m4-max-40c | qwen3.6-35b-a3b | 2.0 | 4096 | MLX | 130.2 | 46–82 | ↑ above | 63–171 | ok | 0.095 | antekapetanovic (Qwen3.5) |
| m3-max-40c | qwen3.6-35b-a3b | 1.9 | 1263 | MLX | 95 | 43–74 | ↑ above | 58–140 | ok | 0.135 | hiesch mlx-lm 0.31.3 (Qwen3.5) |
| m3-max-40c | qwen3.6-35b-a3b | 1.9 | 1263 | MLX | 71.1 | 43–74 | ok | 58–140 | ok | 0.223 | hiesch mlx-lm 0.31.2 (Qwen3.5) |
| m3-max-40c | qwen3-32b | 20.3 | 1263 | MLX | 19 | 12–15 | ↑ above | 14–17 | ↑ above | -0.029 | hiesch mlx-lm |
| m3-max-40c | qwen3.6-27b | 17.1 | 1263 | MLX | 20.3 | 13–17 | ↑ above | 15–20 | ok | 0.053 | hiesch mlx-lm |
| m3-max-40c | gemma-4-e2b | 1.5 | 1263 | MLX | 110.2 | 51–89 | ↑ above | 69–177 | ok | 0.148 | hiesch mlx-lm 0.31.2 |
| m3-max-40c | gemma-4-e4b | 2.9 | 1263 | MLX | 60.1 | 38–61 | ok | 49–103 | ok | 0.214 | hiesch mlx-lm 0.31.2 |
| m3-max-40c | gemma-4-26b-a4b | 2.3 | 1263 | MLX | 65 | 51–80 | ok | 66–129 | ↓ below | 0.304 | hiesch mlx-lm 0.31.2 |
| m1-max | qwen3.6-35b-a3b | 1.9 | 1024 | MLX | 57 | 43–74 | ok | 58–141 | ↓ below | 0.311 | famstack LM Studio (Qwen3.5) |
| m1-max | qwen3.6-35b-a3b | 1.9 | 1024 | llama.cpp | 29 | 43–74 | ↓ below | 58–141 | ↓ below | 0.734 | famstack LM Studio GGUF (Qwen3.5) |
| m1-max | qwen3-30b-a3b | 1.9 | 1024 | MLX | 55.5 | 37–65 | ok | 51–132 | ok | 0.267 | famstack pt2 LM Studio |
| m1-max | qwen3-30b-a3b | 1.9 | 1024 | llama.cpp | 58 | 37–65 | ok | 51–132 | ok | 0.251 | famstack pt2 LM Studio GGUF |
| m5-max-40c | qwen3.6-35b-a3b | 2.0 | 4096 | MLX | 120.4 | 47–85 | ↑ above | 65–184 | ok | 0.122 | ywchiu omlx @4K |
| m5-max-40c | gemma-4-e2b | 1.5 | 4096 | MLX | 204.9 | 55–101 | ↑ above | 76–231 | ok | 0.066 | Incept5 4bit @4k |
| m5-max-40c | gemma-4-e4b | 2.9 | 4096 | MLX | 126.7 | 42–72 | ↑ above | 57–140 | ok | 0.067 | Incept5 4bit @4k |
| m5-max-40c | gemma-4-26b-a4b | 2.4 | 4096 | MLX | 116.5 | 57–96 | ↑ above | 76–176 | ok | 0.147 | Incept5 4bit @4k |
| m5-max-40c | gemma-4-31b | 19.1 | 4096 | MLX | 27.2 | 17–22 | ↑ above | 19–27 | ok | 0.056 | Incept5 4bit @4k |
| m5-max-40c | qwen3.5-122b-a10b | 5.8 | 4096 | MLX | 65.9 | 32–49 | ↑ above | 41–80 | ok | 0.105 | hardware-corner @4K |
| m5-max-40c | gpt-oss-120b | 2.9 | 4096 | MLX | 87.9 | 47–79 | ↑ above | 63–144 | ok | 0.173 | hardware-corner @4K |
| m5-max-40c | qwen3-coder-next-80b-a3b | 3.3 | 4096 | MLX | 79.3 | 37–63 | ↑ above | 50–123 | ok | 0.144 | hardware-corner 8bit @4K |
| m3-ultra | llama-3.1-8b | 5.5 | 1024 | MLX | 119.14 | 46–72 | ↑ above | 59–113 | ↑ above | 0.036 | guruswami Q4 |
| m3-ultra | llama-3.1-8b | 9.5 | 1024 | MLX | 73.85 | 37–52 | ↑ above | 45–71 | ↑ above | 0.033 | guruswami Q8 |
| m3-ultra | llama-3.1-8b | 9.4 | 32768 | MLX | 66.53 | 37–53 | ↑ above | 46–72 | ok | 0.085 | guruswami Q4 @32K |
| m3-ultra | qwen3-32b | 19.4 | 1024 | MLX | 31.46 | 18–26 | ↑ above | 22–35 | ok | 0.098 | guruswami Qwen2.5-32B Q4 |
| m4-pro | qwen3.6-35b-a3b | 1.9 | 2048 | MLX | 78.81 | 39–63 | ↑ above | 51–104 | ok | 0.128 | HN LM Studio |
| m4-pro | qwen3.6-27b | 17.1 | 2048 | MLX | 14.65 | 10–12 | ↑ above | 12–14 | ↑ above | 0.014 | HN LM Studio |
| m4-pro | gemma-4-26b-a4b | 2.3 | 2048 | MLX | 64.93 | 44–64 | ↑ above | 55–93 | ok | 0.206 | HN LM Studio |
| m4-pro | qwen3-30b-a3b | 1.9 | 512 | MLX | 83.1 | 34–57 | ↑ above | 46–102 | ok | 0.096 | BaseRT |
| m4-pro | llama-3.2-3b | 2.2 | 512 | MLX | 112.1 | 47–70 | ↑ above | 59–101 | ↑ above | 0.016 | BaseRT |
| m4-max-40c | qwen3-4b | 2.6 | 512 | MLX | 170 | 47–79 | ↑ above | 63–146 | ↑ above | 0.023 | runanywhere.ai |
| m4-max-40c | llama-3.2-3b | 2.2 | 512 | MLX | 210 | 59–99 | ↑ above | 79–177 | ↑ above | 0.019 | runanywhere.ai |

## Sources

- llama.cpp discussion #4167 — https://github.com/ggml-org/llama.cpp/discussions/4167 (LLaMA-2-7B, `llama-bench -p 512 -n 128`; M1–M4 rows commit 8e672ef, 2023; M5 rows commit c1d0e7a, 2026)
- mlx-lm BENCHMARKS.md — https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md (M4 Max 64GB, mlx 0.29.2, mlx-lm 0.28.2)
- Apple, M5 neural accelerators — https://machinelearning.apple.com/research/exploring-llms-mlx-m5 (ratios only: generation 1.19–1.27× M5/M4)
- mlx-lm PR #1569 (M3 Pro 18GB), issue #858 (M1 Pro 32GB, same-machine A/B)
- antekapetanovic.com — https://antekapetanovic.com/blog/qwen3.5-apple-silicon-benchmark/ (M4 Max 128GB, macOS 26.3, 10 runs ± sd)
- hiesch.eu — https://hiesch.eu/blog/running-local-llms-easter-weekend and https://hiesch.eu/blog/llamacpp-benchmarks-speculative-decoding (M3 Max 64GB, versions per CSV)
- famstack.dev — https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/ and part 2 (M1 Max 64GB 24-core, LM Studio 0.4.5)
- arXiv 2601.19139 (M4 Max 128GB, vllm-mlx authors), arXiv 2607.00501 (BaseRT, M4 Pro)
- runanywhere.ai (M4 Max 64GB)
- haxlys/llm-bench — https://github.com/haxlys/llm-bench (M5 Max 128GB, macOS 26.4, matched quant per runtime)
- guruswami-ai/mlx-benchmarks — https://github.com/guruswami-ai/mlx-benchmarks (M3 Ultra 512GB, MLX 0.30.7, raw CSVs)
- ywchiu/mlx_benchmark_lab (M5 Max 64GB), Incept5/gemma4-benchmark (M5 Max 128GB, MLX 0.31.1)
- hardware-corner.net M5 Max post (second-hand from a forum user; mlx_lm, per-context tables)
- Hacker News 49079803 (Mac mini M4 Pro 64GB, LM Studio 0.4.20)
- Hugging Face cards: ronnie3786 Qwen3.5-27B MLX-4bit (M1 Ultra, 10 runs), teex-pt AMALIA-9B (M5 Pro, llama.cpp b9850)

Not used: reddit (unreachable), anything publishing ranges without a machine, "effective"
tok/s that divide output by wall-clock including prefill, and MLX discussion #3300's
"95–111% of bandwidth" claim, which is taken against a 620GB/s "effective" figure rather
than the M3 Ultra's 819GB/s peak.
