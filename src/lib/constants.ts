/**
 * The vocabularies the site is built on, with no zod attached.
 *
 * Split out of schemas.ts for one concrete reason: fit.ts needs exactly one
 * runtime value from it -- BANDS, four strings -- and importing that pulled the
 * whole of schemas.ts into the browser bundle, zod included, because schemas.ts
 * builds its validators at module top level. The client had ~40KB of a
 * validation library it never calls, and shipping it also carried zod's MIT
 * notice obligation for code the browser had no use for.
 *
 * schemas.ts re-exports everything here, so importing from either still works.
 */

/** Editorial scoring unit. Order matters: index = quality/memory rank. */
export const BANDS = ['low', 'mid', 'high', 'full'] as const;

export const BAND_LABELS: Record<Band, string> = {
  low: 'Q2–Q3',
  mid: 'Q4–Q5',
  high: 'Q6–Q8',
  full: 'FP16',
};

/** Plain-language explanation shown alongside the Q-notation label. */
export const BAND_DESCRIPTIONS: Record<Band, string> = {
  low: '2–3 bit',
  mid: '4–5 bit',
  high: '6–8 bit',
  full: 'full precision (fp16/bf16)',
};

/**
 * Validated open vocabulary: every quant/precision a measurement may be
 * recorded at, mapped to its band. Extend by adding entries here — validate.ts
 * rejects unknown quant strings so typos never enter the dataset.
 */
export const QUANT_VOCAB: Record<string, Band> = {
  // ≤3-bit GGUF. 1-bit and ternary formats are deliberately absent: they are not
  // something this site asks anyone to run, so they are not part of the vocabulary.
  IQ2_XXS: 'low', IQ2_XS: 'low', IQ2_S: 'low', IQ2_M: 'low',
  Q2_K: 'low', Q2_K_S: 'low',
  IQ3_XXS: 'low', IQ3_XS: 'low', IQ3_S: 'low', IQ3_M: 'low',
  Q3_K_S: 'low', Q3_K_M: 'low', Q3_K_L: 'low',
  'mlx-3bit': 'low', Q3_K_XL: 'low',
  // 4–5 bit
  Q4_0: 'mid', Q4_1: 'mid', Q4_K_S: 'mid', Q4_K_M: 'mid',
  IQ4_XS: 'mid', IQ4_NL: 'mid',
  Q5_0: 'mid', Q5_1: 'mid', Q5_K_S: 'mid', Q5_K_M: 'mid',
  'mlx-4bit': 'mid', 'awq-4bit': 'mid', 'gptq-4bit': 'mid',
  nvfp4: 'mid', mxfp4: 'mid', fp4: 'mid',
  // 6–8 bit
  Q6_K: 'high', Q6_K_XL: 'high', Q8_0: 'high',
  'mlx-8bit': 'high', fp8: 'high', int8: 'high',
  // full precision
  fp16: 'full', bf16: 'full', fp32: 'full',
};

export const GLOSSARY_CATEGORIES = ['quantization', 'memory', 'tooling', 'benchmark'] as const;

export const GLOSSARY_CATEGORY_LABELS: Record<GlossaryCategory, string> = {
  quantization: 'Quantization formats',
  memory: 'Memory & architecture',
  tooling: 'Runtimes & file formats',
  benchmark: 'Benchmarks',
};

export const GLOSSARY_CATEGORY_BLURBS: Record<GlossaryCategory, string> = {
  quantization:
    'The compression formats model weights are shipped in. This is where most of the alphabet soup lives.',
  memory: 'What actually consumes memory when a model runs, and the architecture words behind it.',
  tooling: 'The programs that run models locally, and the file formats they read.',
  benchmark: 'The tests behind the numbers on every model sheet — what each one measures.',
};

export const DIMENSION_IDS = [
  'math',
  'coding',
  'tool-calling',
  'long-context',
  'factuality',
  'instruction-following',
  'creative-writing',
] as const;

export const PLATFORM_IDS = ['unified', 'unified-amd', 'unified-nvidia', 'discrete', 'cpu'] as const;

/**
 * How fast this platform ingests a prompt, as a class rather than a number.
 * Prompt processing is compute-bound where generation is bandwidth-bound, and
 * the two invert: a DGX Spark chews through a prompt several times faster than
 * a Strix Halo while generating no quicker, and an M3 Ultra reverses that. A
 * tok/s figure here would depend on the backend (Metal / CUDA / ROCm / Vulkan)
 * and is honestly measured for only a handful of machines, so we grade it.
 */
export const PREFILL_CLASSES = ['fast', 'moderate', 'slow'] as const;

export const SEVERITIES = ['mild', 'moderate', 'severe'] as const;

export type Band = (typeof BANDS)[number];
export type GlossaryCategory = (typeof GLOSSARY_CATEGORIES)[number];
export type DimensionId = (typeof DIMENSION_IDS)[number];
export type PlatformId = (typeof PLATFORM_IDS)[number];
export type PrefillClass = (typeof PREFILL_CLASSES)[number];
export type Severity = (typeof SEVERITIES)[number];
