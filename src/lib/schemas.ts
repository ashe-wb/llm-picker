import { z } from 'zod';
import {
  BANDS,
  BAND_LABELS,
  BAND_DESCRIPTIONS,
  QUANT_VOCAB,
  GLOSSARY_CATEGORIES,
  GLOSSARY_CATEGORY_LABELS,
  GLOSSARY_CATEGORY_BLURBS,
  DIMENSION_IDS,
  PLATFORM_IDS,
  PREFILL_CLASSES,
  SEVERITIES,
} from './constants';
import type {
  Band,
  GlossaryCategory,
  DimensionId,
  PlatformId,
  PrefillClass,
  Severity,
} from './constants';
export * from './constants';

/* -------------------------------------------------------------------------- */
/* Bands & quant vocabulary                                                   */
/* -------------------------------------------------------------------------- */

export const BandSchema = z.enum(BANDS);




export const QuantSchema = z
  .string()
  .refine((q) => q in QUANT_VOCAB, {
    message: 'Unknown quant/precision — add it to QUANT_VOCAB in src/lib/schemas.ts if legitimate',
  });

export function bandOfQuant(quant: string): Band | undefined {
  return QUANT_VOCAB[quant];
}

/* -------------------------------------------------------------------------- */
/* Glossary                                                                   */
/* -------------------------------------------------------------------------- */

export const GlossaryCategorySchema = z.enum(GLOSSARY_CATEGORIES);



/**
 * One explained term. A single entry may cover several quant strings: Q4_K_S,
 * Q4_K_M and Q4_K_L are one scheme at three sizes, not three ideas. validate.ts
 * checks that the union of every `quants` list covers QUANT_VOCAB exactly, so a
 * new quant cannot enter the vocabulary without an explanation.
 */
export const GlossaryTermSchema = z.object({
  /** Anchor slug — links point at /glossary#<id>. */
  id: z.string().regex(/^[a-z0-9][a-z0-9._-]*$/),
  term: z.string(),
  /** What the letters stand for, when they stand for something. */
  expands: z.string().optional(),
  category: GlossaryCategorySchema,
  /** Same field name, and same plain voice, as MitigationSchema. */
  newbieExplainer: z.string(),
  /** The one-line "so which do I pick" answer. The most useful field here. */
  rule: z.string().optional(),
  quants: z.array(QuantSchema).default([]),
  /** Extra spellings for matching free-text provenance metric strings. */
  aliases: z.array(z.string()).default([]),
  seeAlso: z.array(z.string()).default([]),
  url: z.string().url().optional(),
});

/* -------------------------------------------------------------------------- */
/* Dimensions                                                                 */
/* -------------------------------------------------------------------------- */

export const DimensionIdSchema = z.enum(DIMENSION_IDS);

export const DimensionSchema = z.object({
  id: DimensionIdSchema,
  label: z.string(),
  newbieBlurb: z.string(),
  quantSensitive: z.boolean(),
});

/* -------------------------------------------------------------------------- */
/* Hardware: platforms, thresholds, presets                                   */
/* -------------------------------------------------------------------------- */

/**
 * How much memory a platform will actually let a model occupy. Encoded as data
 * rather than code because "macOS gives the GPU ~75% of RAM" is a claim about
 * the world, and claims on this site carry citations.
 */
export const CeilingSchema = z.object({
  kind: z.enum(['fraction-of-ram', 'vram-minus', 'ram-only']),
  /** Fraction for `fraction-of-ram`, GB to subtract for `vram-minus`. */
  value: z.number().positive().optional(),
  /**
   * Names the cap in three or four words, for the parenthetical on the card's
   * RUNS AT line — "the macOS GPU cap", "reserved for the display". Written to
   * slot into a sentence, not to stand alone. Required for every kind except
   * `ram-only`, which has no cap to name; validate.ts enforces that.
   */
  capLabel: z.string().optional(),
  overridable: z.boolean().default(false),
  /**
   * How far this site is willing to OFFER the cap raised — not how far it is
   * safe to raise it. Nobody knows that: Apple documents `iogpu.wired_limit_mb`
   * nowhere, so there is no official guidance to appeal to and this project is
   * not an authority on what a stranger's machine survives.
   *
   * The offer stops at RAM minus the platform's OS reserve — the same figure
   * `budgetFor` spends — so the control cannot promise more than the model will
   * use. This block once carried a second, sqrt-scaled reserve of its own; it
   * never bound at any capacity and was removed.
   *
   * `publishedReserveGb` is what someone else, with a source, says to hold back
   * at a given capacity. Kept for comparison, and options beyond it are marked
   * as beyond it, so the reader can see where consensus sits either way.
   */
  raise: z
    .object({
      /** Capacity in GB -> the reserve a cited source recommends at that capacity. */
      publishedReserveGb: z.record(z.string(), z.number().positive()).optional(),
      note: z.string(),
      /** Required by validate.ts: this is a claim about the world. */
      citation: z.string().url().optional(),
    })
    .optional(),
  note: z.string(),
  citation: z.string().url().optional(),
});

/** Memory left for the OS and other apps. Proportional with an absolute floor. */
export const ReserveSchema = z.object({
  minGb: z.number().nonnegative(),
  fractionOfRam: z.number().min(0).max(1),
  note: z.string(),
});

/** Shown when a model fails *only* because of the platform ceiling. */
export const HardwareAdviceSchema = z.object({
  id: z.string(),
  label: z.string(),
  trigger: z.enum(['ceiling-only']),
  newbieExplainer: z.string(),
  command: z.string().optional(),
  caveat: z.string().optional(),
  citation: z.string().url().optional(),
});

export const PlatformIdSchema = z.enum(PLATFORM_IDS);


/**
 * A specific piece of silicon and the bandwidth it reads memory at.
 *
 * Apple silicon needs this because capacity says nothing about bandwidth: a
 * 32GB Mac is anything from an M4 at 120GB/s to an M5 Max at 614, a 5x spread,
 * and generation speed is bounded by that number. It does not track generation
 * either — the M3 Pro reads at 150GB/s where the M1 and M2 Pro both managed
 * 200, a regression Apple shipped deliberately. No formula recovers this, so it
 * is a table.
 *
 * Discrete GPUs need no entry: a preset names exactly one card.
 */
export const ChipSchema = z.object({
  id: z.string(),
  label: z.string(),
  /** Generation, for grouping the dropdown. */
  family: z.string(),
  platform: PlatformIdSchema,
  bandwidthGbs: z.number().positive(),
  /**
   * Bandwidth at a specific capacity, where the chip is not uniform. The M6
   * reads at 153GB/s with 16GB fitted and 170GB/s at 24GB or more — the only
   * chip in the table that does this, and the reason this is a map rather than
   * a second entry pretending to be a different chip.
   */
  bandwidthAtRamGb: z.record(z.string(), z.number().positive()).optional(),
  /**
   * Exactly the capacities Apple sold this chip with, from their tech specs.
   * An enumeration, not a bound: an M1 Max was 32 or 64GB and nothing between,
   * so a range would keep offering the 36 and 48 rungs it never shipped with.
   *
   * The union across machines, because this describes silicon rather than a
   * product — the 32-core M5 Max is 36GB only in a MacBook Pro but reaches
   * 128GB in a Mac Studio, and it is the same chip either way.
   */
  ramOptions: z.array(z.number().positive()).min(1),
  citation: z.string().url().optional(),
  /**
   * Overrides the platform's prompt-processing class, the way a preset can.
   * Needed because on Apple silicon prefill stopped tracking bandwidth with
   * the M5: its GPU cores carry matrix units that a 40-core M5 Max ingests a
   * prompt with at 3.6x the rate of a 40-core M4 Max, and past both Ultras.
   */
  prefillClass: z.enum(PREFILL_CLASSES).optional(),
  /** Shown while a machine on this chip is selected; forwarded to its presets. */
  note: z.string().optional(),
  noteUrl: z.string().url().optional(),
});

/**
 * How fast the CPU reads system RAM, in GB/s. This is not a detail of the CPU:
 * it is the speed of every layer the CPU runs, so on a machine with a graphics
 * card too small for the model it is half of what governs the answer, and on a
 * machine with no card at all it is the whole of it.
 *
 * A class rather than a per-machine measurement, because the figure is decided
 * by generation and channel count and nothing else: `MT/s x 8 bytes x channels`.
 * Two channels of DDR5-5600 read at 89.6GB/s in a laptop and in a tower alike.
 */
export const SystemMemorySchema = z.object({
  id: z.string(),
  label: z.string(),
  bandwidthGbs: z.number().positive(),
  note: z.string(),
  citation: z.string().url().optional(),
});

export const PlatformSchema = z.object({
  id: PlatformIdSchema,
  label: z.string(),
  /**
   * How this platform is offered in the picker, where the question is "what
   * graphics card do you have" rather than "what memory architecture is this".
   * `label` describes the architecture and is what the budget line quotes; this
   * describes the hardware and is what someone picks from. They differ because
   * "CPU only (no GPU acceleration)" read as a claim that the CPU is idle on
   * every other option, which is the misreading this field exists to end.
   */
  pickerLabel: z.string().optional(),
  blurb: z.string(),
  prefillClass: z.enum(PREFILL_CLASSES).optional(),
  ceiling: CeilingSchema,
  reserve: ReserveSchema,
  advice: HardwareAdviceSchema.optional(),
  offload: z.object({ allowed: z.boolean(), note: z.string() }).optional(),
});

export const ThresholdsSchema = z.object({
  tightAtFractionOfBudget: z.number().min(0).max(1),
  tightBelowHeadroomGb: z.number().nonnegative(),
  tightBelowHeadroomFraction: z.number().min(0).max(1),
  /** Minimum share of a model that must stay in fast memory to be offered at all. */
  offloadMinResidentFraction: z.number().min(0).max(1),
  note: z.string(),
});

/** Quick-fill machine. Replaces the old hardware tiers. */
export const PresetSchema = z.object({
  id: z.string(),
  label: z.string(),
  /** Heading the chip is filed under. Purely presentational. */
  group: z.string(),
  /** Optional second level inside that heading, rendered as an optgroup. */
  subgroup: z.string().optional(),
  platform: PlatformIdSchema,
  ramGb: z.number().positive(),
  vramGb: z.number().positive().optional(),
  /**
   * Peak theoretical memory bandwidth, GB/s. This is what separates machines
   * the memory model calls identical: generation reads every active weight once
   * per token, so tokens/sec is bounded by bandwidth divided by bytes read.
   * A published spec, not a measurement. Set on discrete cards, where a preset
   * names exactly one piece of silicon. Apple presets leave it unset and carry
   * a `chipId` instead, because a 32GB Mac spans 120 to 614 GB/s.
   */
  memoryBandwidthGbs: z.number().positive().optional(),
  /** Default chip for this preset, on platforms where the chip varies. */
  chipId: z.string().optional(),
  /**
   * Which `systemMemory` class this machine typically ships with. Sets the
   * speed of any layer the CPU runs, so a Tesla-era card in a DDR4 server and
   * a current card in a DDR5 desktop offload at different speeds even when the
   * VRAM figure is identical.
   */
  systemMemoryId: z.string().optional(),
  /**
   * Overrides the platform's prompt-processing class. The `discrete` platform
   * defaults to CUDA-flavoured `fast`, which ROCm, SYCL/Vulkan and the
   * pre-Pascal Teslas have not earned.
   */
  prefillClass: z.enum(PREFILL_CLASSES).optional(),
  /**
   * A caveat that survives the memory model. This site sizes what FITS, so two
   * cards with the same VRAM get the same answer — fine when they differ only in
   * speed, wrong when one of them cannot run the band being recommended. A Tesla
   * P40 holds a 24GB model exactly like a 3090 and then executes FP16 at 1/64
   * rate. Shown only while that preset is the selected machine.
   */
  note: z.string().optional(),
  noteUrl: z.string().url().optional(),
});

/**
 * Memory a running model needs beyond weights and KV cache: framework and kernel
 * overhead (roughly fixed) plus llama.cpp's compute buffer (scales with model
 * dimensions). Omitting it made every fit verdict about 2GB optimistic.
 */
export const RuntimeOverheadSchema = z.object({
  /**
   * llama.cpp itself, its backend kernels and the serving process — everything
   * that is there before a single token is processed. The one input here that
   * is not derived, so it is a RANGE rather than a number: published figures
   * put it at 0.5-1.5GB and this site does not pretend to know better. Fit
   * verdicts use the high end, so a fit is never promised on optimism.
   */
  frameworkGbLo: z.number().positive(),
  frameworkGbHi: z.number().positive(),
  /** Micro-batch size the estimate assumes. llama.cpp's default is 512. */
  ubatch: z.number().int().positive(),
  /**
   * Peak simultaneously-live copies of the widest activation tensor. Three is
   * the FFN's gate, up and their product, which is where the graph peaks.
   */
  activationCopies: z.number().int().positive(),
  /**
   * Bytes of KQ mask per token of context, per token of micro-batch. Two, for
   * an f16 mask. Assumes flash attention is on, which is llama.cpp's default;
   * without it the attention scores are materialised instead and cost orders
   * of magnitude more. See `flashAttentionOffBytesPerTokenPerHead`.
   */
  kqBytesPerTokenPerUbatch: z.number().positive(),
  /**
   * What one token of context costs per attention head with flash attention
   * OFF, in bytes: the materialised scores tensor, f32. Not part of any
   * estimate — used only to tell the reader what they would pay for turning
   * flash attention off, which at 128K is measured in gigabytes.
   */
  flashAttentionOffBytesPerTokenPerHead: z.number().positive(),
  note: z.string(),
  citation: z.string().url().optional(),
});

/**
 * The fixed cost of producing one token that does NOT scale with weight bytes:
 * kernel launches, attention over the KV cache, sampling, expert routing.
 *
 * This replaced a multiplicative "fraction of peak bandwidth", which was the
 * wrong shape. Utilisation is not a property an engine has — it is what you
 * observe when a fixed overhead is divided by a varying read. That is why
 * llama.cpp's own table shows 4-bit looking less efficient than F16 on the same
 * chip, and why sparse mixture-of-experts models, which read a fraction of
 * themselves per token, looked wildly efficient and were being overpredicted.
 *
 * Charged PER LAYER, because that is what the measurements show. Solving for a
 * flat per-token cost gives 6-14ms on a 32-layer 7B but 25-41ms on an 80-layer
 * 70B; divided by layer count both land near 0.2-0.5ms, which is what a
 * per-layer kernel dispatch would do. A flat figure misses a published 70B
 * measurement; the per-layer one brackets it.
 *
 * Modelled per platform because it is not portable: CUDA launches cheaper than
 * Metal, and solving the same equation against each platform's published
 * measurements gives materially different numbers.
 */
export const PerTokenLatencySchema = z.object({
  /** Per LAYER, not per token: kernel launches are dispatched per layer. */
  loMs: z.number().positive(),
  hiMs: z.number().positive(),
  /** Where a platform's own measurements give a better figure than the default. */
  byPlatform: z
    .record(PlatformIdSchema, z.object({ loMs: z.number().positive(), hiMs: z.number().positive() }))
    .optional(),
  note: z.string(),
  citation: z.string().url().optional(),
});

export const HardwareSchema = z.object({
  platforms: z.array(PlatformSchema).min(1),
  thresholds: ThresholdsSchema,
  runtimeOverhead: RuntimeOverheadSchema,
  ramOptions: z.array(z.number().positive()).min(1),
  vramOptions: z.array(z.number().positive()).min(1),
  presets: z.array(PresetSchema).min(1),
  chips: z.array(ChipSchema).min(1),
  systemMemory: z.array(SystemMemorySchema).min(1),
  perTokenLatency: PerTokenLatencySchema,
});

/** A machine the user described. Not persisted — built from the picker inputs. */
export interface Machine {
  platform: PlatformId;
  ramGb: number;
  vramGb?: number;
  /** User-supplied GPU memory ceiling in GB, when the platform allows one. */
  ceilingOverrideGb?: number;
  /**
   * Which chip, when the platform has a choice. Apple presets cannot carry a
   * bandwidth because capacity does not imply one; this is where the figure
   * actually comes from on a Mac.
   */
  chipId?: string;
  /**
   * How fast this machine's system RAM is, as a `systemMemory` id. Governs the
   * layers the CPU runs, so it is the slow half of any partial-offload figure
   * and the whole of a CPU-only one. Undefined falls back to the platform
   * default rather than to a guess about the reader's motherboard.
   */
  systemMemoryId?: string;
  /**
   * Which preset the reader started from, when they did. The other fields
   * cannot identify a machine on their own — eight presets are `discrete` with
   * 24GB of VRAM, from a Tesla M40 at 288GB/s to an RTX 4090 at 1008 — so
   * without this a P40 would be quoted a 4090's speed. Cleared the moment a
   * control is edited by hand, because it is then no longer that machine.
   */
  presetId?: string;
}

/* -------------------------------------------------------------------------- */
/* Tasks (picker)                                                             */
/* -------------------------------------------------------------------------- */

export const TaskSchema = z.object({
  id: z.string(),
  label: z.string(),
  blurb: z.string().optional(),
  /** Context this workload is assumed to need. Drives the KV half of the fit. */
  assumedContextTokens: z.number().int().positive(),
  requires: z.object({ vision: z.boolean().optional() }).optional(),
  /** Dimension weights; validate.ts enforces they sum to 1.0. */
  weights: z.record(DimensionIdSchema, z.number().min(0).max(1)),
});

/* -------------------------------------------------------------------------- */
/* Mitigation library                                                         */
/* -------------------------------------------------------------------------- */

export const MitigationSchema = z.object({
  id: z.string(),
  label: z.string(),
  newbieExplainer: z.string(),
  howToLinks: z.array(z.object({ label: z.string(), url: z.string().url() })).default([]),
  /**
   * If set, this mitigation only works when the model itself is decent at the
   * named dimension (e.g. code-interpreter needs tool-calling ≥ 4).
   */
  worksVia: DimensionIdSchema.optional(),
  caveat: z.string().optional(),
});

/* -------------------------------------------------------------------------- */
/* Source registry                                                            */
/* -------------------------------------------------------------------------- */

export const SourceSchema = z.object({
  id: z.string(),
  name: z.string(),
  url: z.string().url().optional(),
  dataAccess: z.string().optional(),
  licenseNote: z.string().optional(),
  fetchMethod: z.enum(['api', 'manual', 'link-only']),
  /** false = cite/link only; validate.ts rejects rawValue on such sources. */
  republishOk: z.boolean().default(true),
  dimensions: z.array(DimensionIdSchema).default([]),
});

/* -------------------------------------------------------------------------- */
/* Models                                                                     */
/* -------------------------------------------------------------------------- */

const BandInfoSchema = z.object({
  /**
   * Weights only, at a typical quant for the band. KV cache is NOT included —
   * it depends on context length and is computed from `kvBytesPerToken`.
   */
  weightsGb: z.number().positive(),
  exampleQuants: z.array(QuantSchema).min(1),
});

export const ModelSchema = z.object({
  id: z.string().regex(/^[a-z0-9][a-z0-9.-]*$/),
  name: z.string(),
  family: z.string(),
  vendor: z.string(),
  architecture: z.enum(['dense', 'moe']),
  paramsB: z.number().positive(),
  activeParamsB: z.number().positive().optional(),
  /**
   * `num_hidden_layers` from the model's config.json — the same file
   * `kvSourceUrl` already cites for the KV figure. Present so a partial fit can
   * be stated as the layer count the reader actually types after `-ngl`, rather
   * than as a percentage they then have to convert.
   */
  layers: z.number().int().positive().optional(),
  /**
   * `vocab_size`, from the same config.json. This is the single biggest term in
   * a runtime's compute buffer: the final projection materialises one row per
   * vocabulary entry for every token in the micro-batch, which is half a
   * gigabyte on a 256K-vocabulary model before any activation is counted.
   */
  vocabSize: z.number().int().positive().optional(),
  /**
   * The widest tensor one micro-batch flows through:
   * `max(hidden_size, intermediate_size, moe_intermediate_size * num_experts_per_tok)`.
   * The MoE term is what keeps a sparse model honest — a 397B-A17B is narrow
   * where it counts, and sizing its scratch from total parameters was the bug
   * this field exists to make impossible.
   */
  width: z.number().int().positive().optional(),
  /**
   * `num_attention_heads`. Not part of any estimate: used only to price what
   * running WITHOUT flash attention would cost, which is the difference
   * between a mask and a materialised scores tensor — gigabytes at long
   * context, and a failure mode the site would otherwise never mention.
   */
  attentionHeads: z.number().int().positive().optional(),
  /** `num_key_value_heads`. With attentionHeads this gives the GQA ratio, which
   *  is most of why two models of the same size cache differently. */
  kvHeads: z.number().int().positive().optional(),
  /** Explicit `head_dim`, else hidden_size / num_attention_heads. */
  headDim: z.number().int().positive().optional(),
  /**
   * Per-layer attention type: `F` full, `S` sliding-window, `.` linear or
   * recurrent. One character per layer, so its length equals `layers`.
   *
   * Assembled from whichever of layer_types, hybrid_override_pattern,
   * full_attention_interval or sliding_window_pattern the model publishes; where
   * none of those exist it is written only if the model's own kvBytesPerToken
   * confirms that every layer caches. Absent otherwise — including where the
   * derived pattern CONTRADICTS kvBytesPerToken, since publishing either number
   * would mean picking one without knowing which is wrong.
   */
  attentionPattern: z.string().regex(/^[FS.]+$/).optional(),
  /**
   * Multi-head latent attention: the compressed latent cached per layer per
   * token, `kv_lora_rank + qk_rope_head_dim`. Present only on MLA models, where
   * kvHeads and headDim are absent because they do not describe what is cached
   * — MLA stores one compressed vector, not per-head keys and values.
   */
  mlaLatentDim: z.number().int().positive().optional(),
  /** `num_experts_per_tok`, on mixture-of-experts models. */
  expertsPerToken: z.number().int().positive().optional(),
  vision: z.boolean(),
  contextLength: z.number().int().positive(),
  license: z.object({ name: z.string(), url: z.string().url().optional() }),
  releaseDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  ggufUrl: z.string().url().optional(),
  mlxUrl: z.string().url().optional(),
  /**
   * KV cache from layers that cache the FULL context, in bytes per token at
   * fp16: `2 * globalLayers * num_key_value_heads * head_dim * 2`, taken from
   * the model's config.json. Only layers that really hold a KV cache count —
   * Qwen3.6's DeltaNet layers hold none, so 16 of its 64 layers contribute.
   * This term grows linearly with context.
   */
  kvBytesPerToken: z.number().positive().optional(),
  /**
   * Same, for sliding-window layers, which stop accumulating at
   * `slidingWindow` tokens. Gemma interleaves 5 local layers per global one, so
   * this term is large but bounded. Omit for models without sliding attention.
   */
  kvWindowedBytesPerToken: z.number().positive().optional(),
  /** Window size those layers cap at. Required if kvWindowedBytesPerToken is set. */
  slidingWindow: z.number().int().positive().optional(),
  /**
   * Gemma 4 gives its full-attention layers their own geometry: a wider head
   * (`global_head_dim`, 512 against the sliding layers' 256) over far fewer KV
   * heads (`num_global_key_value_heads`, as low as one). Both fields are needed
   * or neither, and they apply ONLY to the `F` layers -- `kvHeads`/`headDim`
   * keep describing the sliding ones. Absent on models whose layers are all the
   * same shape, which is most of them.
   */
  globalKvHeads: z.number().int().positive().optional(),
  globalHeadDim: z.number().int().positive().optional(),
  /**
   * Trailing layers that reuse an earlier layer's keys and values instead of
   * computing their own (`num_kv_shared_layers`). They allocate no cache at all
   * -- in Gemma 4 E4B the last 18 of 42 layers, which is most of why a 4B-active
   * model caches so little. Counted from the end.
   */
  kvSharedLayers: z.number().int().nonnegative().optional(),
  /** config.json the two figures were computed from. */
  kvSourceUrl: z.string().url().optional(),
  status: z.enum(['active', 'deprecated']).default('active'),
  summary: z.string(),
  /** Which bands exist/are practical for this model. At least one required. */
  bands: z
    .object({
      low: BandInfoSchema.optional(),
      mid: BandInfoSchema.optional(),
      high: BandInfoSchema.optional(),
      full: BandInfoSchema.optional(),
    })
    .refine((b) => Object.values(b).some(Boolean), { message: 'model needs at least one band' }),
});

/* -------------------------------------------------------------------------- */
/* Scores + provenance                                                        */
/* -------------------------------------------------------------------------- */

export const ProvenanceSchema = z.object({
  sourceId: z.string(),
  metric: z.string(),
  /** Raw value as published/measured. Forbidden for republishOk:false sources. */
  rawValue: z.number().optional(),
  /** Exact quant/precision the measurement was taken at. Never erased. */
  quant: QuantSchema.optional(),
  url: z.string().url().optional(),
  retrieved: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  method: z.enum(['aggregated', 'owner-verified', 'vendor-reported']),
  /** How a measurement at another precision was mapped to this band, if applicable. */
  quantNote: z.string().optional(),
  hardware: z.string().optional(),
  notes: z.string().optional(),
});

export const ScoreEntrySchema = z.object({
  band: BandSchema,
  dimension: DimensionIdSchema,
  /** Editorial site score, set by a human from the provenance evidence. */
  score: z.number().int().min(0).max(10),
  provenance: z.array(ProvenanceSchema).min(1),
  notes: z.string().optional(),
});

/** Sparse exception: a specific quant that deviates from its band's score. */
export const QuantOverrideSchema = z.object({
  quant: QuantSchema,
  dimension: DimensionIdSchema,
  score: z.number().int().min(0).max(10),
  provenance: z.array(ProvenanceSchema).min(1),
  note: z.string(),
});

export const ScoresFileSchema = z.object({
  modelId: z.string(),
  entries: z.array(ScoreEntrySchema),
  quantOverrides: z.array(QuantOverrideSchema).default([]),
});

/* -------------------------------------------------------------------------- */
/* Weaknesses → mitigations                                                   */
/* -------------------------------------------------------------------------- */


export const CitationSchema = z.object({
  url: z.string().url(),
  title: z.string(),
  sourceType: z.enum(['reddit', 'github', 'leaderboard', 'paper', 'blog', 'vendor', 'other']),
  accessed: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
});

export const WeaknessEntrySchema = z.object({
  id: z.string(),
  bands: z.array(BandSchema).min(1),
  dimensions: z.array(DimensionIdSchema).min(1),
  severity: z.enum(SEVERITIES),
  summary: z.string(),
  mitigations: z
    .array(
      z.object({
        mitigationId: z.string(),
        note: z.string(),
        effectiveness: z.enum(['strong', 'partial', 'weak']).optional(),
      }),
    )
    .min(1),
  citations: z.array(CitationSchema),
  confidence: z.enum(['community-consensus', 'anecdotal', 'owner-verified']),
});

export const WeaknessesFileSchema = z.object({
  modelId: z.string(),
  entries: z.array(WeaknessEntrySchema),
});

/* -------------------------------------------------------------------------- */
/* Site meta                                                                  */
/* -------------------------------------------------------------------------- */

export const SiteMetaSchema = z.object({
  lastEditorialReview: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  seedDataNote: z.string().optional(),
});

/* -------------------------------------------------------------------------- */
/* Inferred TS types                                                          */
/* -------------------------------------------------------------------------- */

export type Platform = z.infer<typeof PlatformSchema>;
export type HardwareAdvice = z.infer<typeof HardwareAdviceSchema>;
export type Thresholds = z.infer<typeof ThresholdsSchema>;
export type Preset = z.infer<typeof PresetSchema>;
export type Hardware = z.infer<typeof HardwareSchema>;
export type Dimension = z.infer<typeof DimensionSchema>;
export type GlossaryTerm = z.infer<typeof GlossaryTermSchema>;
export type Task = z.infer<typeof TaskSchema>;
export type Mitigation = z.infer<typeof MitigationSchema>;
export type Source = z.infer<typeof SourceSchema>;
export type Model = z.infer<typeof ModelSchema>;
export type Provenance = z.infer<typeof ProvenanceSchema>;
export type ScoreEntry = z.infer<typeof ScoreEntrySchema>;
export type QuantOverride = z.infer<typeof QuantOverrideSchema>;
export type ScoresFile = z.infer<typeof ScoresFileSchema>;
export type WeaknessEntry = z.infer<typeof WeaknessEntrySchema>;
export type WeaknessesFile = z.infer<typeof WeaknessesFileSchema>;
export type SiteMeta = z.infer<typeof SiteMetaSchema>;
