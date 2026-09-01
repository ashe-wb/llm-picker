# Measuring the runtime overhead

The site's third memory term — what the runtime takes before your context — is derived from
what llama.cpp allocates rather than fitted to an observation. Three of its four parts come
from the model's own `config.json`. The fourth, llama.cpp and its backend kernels, is a
published range of 0.5–1.5GB, and that range is why every context figure on the site is a
span rather than a number.

Narrowing it is the highest-value measurement anyone can contribute, and it takes ten minutes.

## What to run

Load any model twice, changing only the context, and once more changing only the micro-batch:

```sh
llama-server -m <model>.gguf -c 4096   -ub 512  --no-warmup
llama-server -m <model>.gguf -c 131072 -ub 512  --no-warmup
llama-server -m <model>.gguf -c 4096   -ub 2048 --no-warmup
```

At startup llama.cpp prints its allocations. Record these lines from each run:

```
llama_kv_cache: ... KV buffer size =  ____ MiB
llama_context:  ... compute buffer size =  ____ MiB
llama_context:  ...     output buffer size =  ____ MiB
```

Also record the process's resident memory once loaded and idle, which is the only way to see
the framework cost that llama.cpp does not print.

**Record `sysctl iogpu.wired_limit_mb` alongside every reading.** A raised cap changes the budget
the measurement is being compared against, and that is exactly what invalidated the last
real-machine figure this project relied on: a ~80K context reading was treated as a 24GB-budget
data point when the machine was running well above 24GB. A measurement without its cap noted
cannot test anything.

## What it tells us

- **Compute buffer, run 1 vs run 3.** Should scale with `-ub`. The model predicts it is
  dominated by `ubatch × vocab_size × 4`; quadrupling the micro-batch should roughly quadruple
  it. If it does not, `activationCopies` or the mechanism itself is wrong.
- **Compute buffer, run 1 vs run 2.** The difference is the attention mask, predicted at
  `context × ubatch × 2` — about 128MB across that jump at `-ub 512`. A much larger difference
  means flash attention was off.
- **Resident memory minus every buffer llama.cpp printed.** That remainder is `frameworkGb`,
  currently assumed to be 0.5–1.5GB. A measured value replaces the range, and every context
  estimate on the site narrows with it.

## Where it goes

`runtimeOverhead` in `data/hardware.json`. Set `frameworkGbLo` and `frameworkGbHi` to the
measured value plus an honest margin, record the machine, the model, the llama.cpp build and
the raw numbers in the `note`, and cite the run.

If the measurement contradicts the derivation rather than narrowing it, that is the more
valuable result. The test `RETRACTED: there is no real-machine check on the overhead term` in
`tests/picker.test.ts` marks the slot a real measurement should take; replace it with a check
that fails when the derivation is contradicted rather than one that absorbs it — fix the model,
not the test.
