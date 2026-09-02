import { BANDS } from './constants';
import type {
  Preset,
  Band,
  Hardware,
  Machine,
  Model,
  Platform,
} from './schemas';

const BYTES_PER_GB = 1024 ** 3;

/**
 * comfortable — runs with room to spare.
 * tight       — runs, but only just, or only by spilling into system RAM.
 * no          — does not run.
 */
export type FitState = 'comfortable' | 'tight' | 'no';

export interface Budget {
  /** Memory a model may actually occupy at full speed. */
  gb: number;
  ceilingGb: number;
  reserveGb: number;
  /** True when the platform ceiling — not the OS reserve — is the binding limit. */
  ceilingBinds: boolean;
  /** Ceiling + spare system RAM. Above `gb`, below this, you are offloading. */
  offloadGb: number;
  platform: Platform;
}

export interface Fit {
  state: FitState;
  demandGb: number;
  weightsGb: number;
  kvGb: number;
  budgetGb: number;
  headroomGb: number;
  /**
   * Framework, kernels and the compute buffer — everything that is not weights
   * or KV. The pessimistic end of the range, because this decides whether a
   * model is reported as fitting.
   */
  overheadGb: number;
  /** The same figure broken into its parts, for saying what the reader is charged for. */
  overhead: Overhead;
  /** Runs only by spilling layers into system RAM — works, several times slower. */
  offload: boolean;
  /**
   * Share of the model held in fast memory, 0–1. This is what governs speed —
   * 98% resident and 35% resident are not the same kind of offload — and it maps
   * onto the layer count a reader actually sets in their runtime.
   */
  residentFraction: number;
  /**
   * How much of the model the CPU holds and runs rather than the graphics card.
   * Not streamed anywhere: it is read from system RAM in place, at the speed of
   * that memory. Zero when fully resident.
   */
  spilledGb: number;
  /** Would fit if the platform's memory ceiling were raised. Actionable advice. */
  ceilingBlocked: boolean;
  /** Context actually used: the task's ask, clamped to what the model supports. */
  contextTokens: number;
  /**
   * The longest context this band can hold on this machine, capped at the
   * model's own window. Spare memory can only be spent on context once the band
   * is chosen, so this is what headroom is actually worth.
   */
  maxContextTokens: number;
  /**
   * The same figure at the optimistic end of the framework range.
   *
   * Computed, kept, and deliberately NOT published. The site showed both ends
   * as a range until the only real-machine report available said the estimate
   * reads high; with the framework term unmeasured, the flattering end is not
   * something to lead with. Retained so the spread stays visible in the data
   * and so a future measurement has something to land against.
   */
  maxContextTokensOptimistic: number;
}

export function platformFor(hardware: Hardware, id: Machine['platform']): Platform {
  const p = hardware.platforms.find((x) => x.id === id);
  if (!p) throw new Error(`unknown platform '${id}'`);
  return p;
}

/** OS + other applications. Proportional, with an absolute floor. */
function reserveGb(platform: Platform, ramGb: number): number {
  return Math.max(platform.reserve.minGb, platform.reserve.fractionOfRam * ramGb);
}

/**
 * What this machine will actually let a model occupy. The ceiling and the OS
 * reserve are separate limits; whichever bites first wins. On an unmodified Mac
 * the ceiling always binds (75% < 80%), so the reserve only matters once the
 * user raises `iogpu.wired_limit_mb` — or on platforms with no GPU cap at all.
 */
export function budgetFor(machine: Machine, hardware: Hardware): Budget {
  const platform = platformFor(hardware, machine.platform);
  const reserve = reserveGb(platform, machine.ramGb);
  const spareRam = Math.max(0, machine.ramGb - reserve);

  let ceiling: number;
  let gb: number;
  let offloadGb: number;

  switch (platform.ceiling.kind) {
    case 'fraction-of-ram': {
      const dflt = (platform.ceiling.value ?? 1) * machine.ramGb;
      ceiling =
        platform.ceiling.overridable && machine.ceilingOverrideGb
          ? machine.ceilingOverrideGb
          : dflt;
      // The OS reserve stays in force even when the cap is raised, and that is
      // deliberate: raising iogpu.wired_limit_mb grants a PERMISSION, it does
      // not free memory. macOS and everything else still draw on the same pool,
      // and llama.cpp has no view of that — it takes the limit as its ceiling
      // with no system-wide pressure signal. A reader who sets 30720 on a 32GB
      // machine does not thereby have 30GB for a model.
      //
      // What was wrong was the other half: maxWiredGb offered raise steps past
      // this line, so two of the three options on a 32GB Mac changed nothing.
      // That is fixed there, by capping the offer at what this actually uses.
      gb = Math.min(ceiling, spareRam);
      // No split offered, and NOT because there is nowhere to put the layers:
      // there is a real gap between the GPU cap and RAM-minus-reserve, 1.6GB on
      // a 32GB Mac. It is because moving a layer to the CPU here buys nothing.
      // One pool means the CPU brings no extra memory and no extra bandwidth,
      // and it reads that pool more slowly than the GPU does — measured at
      // 220-240GB/s against the GPU's ~330, which is not an assumption: solving
      // llama.cpp's own Apple table gives 332GB/s on an M2 Max and 338 on an
      // M3 Max for Llama-7B at F16. llama.cpp's
      // maintainer puts it plainly: "if you already saturated it with the GPU,
      // then the CPU won't help". Nor is there parallelism to win, since layer
      // n needs layer n-1. On these machines the answer to a model that does
      // not fit is to raise the cap, which is what `advice` says.
      // @see https://github.com/ggml-org/llama.cpp/discussions/3083
      offloadGb = gb;
      break;
    }
    case 'vram-minus': {
      ceiling = Math.max(0, (machine.vramGb ?? 0) - (platform.ceiling.value ?? 0));
      gb = ceiling;
      // Layers that miss VRAM live in system RAM, where the CPU runs them.
      offloadGb = platform.offload?.allowed ? ceiling + spareRam : ceiling;
      break;
    }
    case 'ram-only':
    default: {
      ceiling = spareRam;
      gb = spareRam;
      offloadGb = spareRam;
      break;
    }
  }

  return {
    gb: Math.round(gb * 10) / 10,
    ceilingGb: Math.round(ceiling * 10) / 10,
    reserveGb: Math.round(reserve * 10) / 10,
    // Strictly less: a 'ram-only' platform sets ceiling = spareRam, and there is
    // no separate cap there to be "bound by".
    ceilingBinds: ceiling < spareRam,
    offloadGb: Math.round(offloadGb * 10) / 10,
    platform,
  };
}

/**
 * Where the budget number came from, in one clause — the answer to "of your
 * 27GB... says who?". The BUDGET line under Step 01 explains this at length,
 * but by the time you are reading a card it is several screens up, and the
 * number arrives with no provenance attached.
 *
 * Always the arithmetic plus the name of whatever imposed it, because either
 * half alone leaves a question: "75% of 36GB" does not say who chose 75%, and
 * "the macOS GPU cap" does not say how it got to 27.
 */
export function budgetSource(machine: Machine, budget: Budget): string {
  const c = budget.platform.ceiling;
  if (budget.ceilingBinds) {
    if (c.kind === 'vram-minus') {
      return `${formatMemory(machine.vramGb ?? 0)} VRAM − ${formatMemory(c.value ?? 0)} ${c.capLabel}`;
    }
    if (c.kind === 'fraction-of-ram') {
      // An override replaces the fraction outright, so quoting 75% would be a lie.
      if (machine.ceilingOverrideGb) return `the ${formatMemory(budget.ceilingGb)} limit you set`;
      return `${Math.round((c.value ?? 1) * 100)}% of ${formatMemory(machine.ramGb)}, ${c.capLabel}`;
    }
  }
  // No cap, or a cap loose enough that the OS reserve bites first.
  return `${formatMemory(machine.ramGb)} RAM − ${formatMemory(budget.reserveGb)} for the OS`;
}

/**
 * KV cache size at a given context. Returns null when the model has no measured
 * `kvBytesPerToken` — we fail closed rather than assume, because assuming a flat
 * allowance is the bug this model replaces.
 */
/**
 * Two terms, because attention is not uniform across layers: global layers
 * cache the whole context, sliding-window layers stop at their window.
 * Split out from `kvGb` so callers holding loose numbers (the Explore table)
 * use the same formula instead of reimplementing it.
 */
export function kvGbFrom(
  bytesPerToken: number,
  windowedBytesPerToken: number | undefined,
  slidingWindow: number | undefined,
  contextTokens: number,
): number {
  const capped = Math.min(contextTokens, slidingWindow ?? contextTokens);
  return (bytesPerToken * contextTokens + (windowedBytesPerToken ?? 0) * capped) / BYTES_PER_GB;
}

/**
 * The inverse of `kvGbFrom`: the longest context a given amount of memory can
 * hold. Piecewise, because sliding-window layers stop accumulating at their
 * window — past that point only the global layers keep growing, so the cost per
 * token drops and the curve bends.
 *
 * Lives next to the forward function so the two cannot drift apart.
 */
export function maxContextFrom(
  availableGb: number,
  bytesPerToken: number,
  windowedBytesPerToken: number | undefined,
  slidingWindow: number | undefined,
): number {
  if (availableGb <= 0 || bytesPerToken <= 0) return 0;
  const bytes = availableGb * BYTES_PER_GB;
  const windowed = windowedBytesPerToken ?? 0;

  if (windowed > 0 && slidingWindow) {
    // Past the window the windowed layers cost a fixed amount.
    const fixed = windowed * slidingWindow;
    if (bytes > fixed) {
      const beyond = Math.floor((bytes - fixed) / bytesPerToken);
      if (beyond >= slidingWindow) return beyond;
    }
    // Still inside the window: every layer is still growing.
    return Math.min(slidingWindow, Math.floor(bytes / (bytesPerToken + windowed)));
  }
  return Math.floor(bytes / bytesPerToken);
}

export function kvGb(model: Model, contextTokens: number): number | null {
  if (model.kvBytesPerToken === undefined) return null;
  return kvGbFrom(
    model.kvBytesPerToken,
    model.kvWindowedBytesPerToken,
    model.slidingWindow,
    contextTokens,
  );
}

/** The runtime's own memory, broken into the pieces it is actually made of. */
export interface Overhead {
  /** llama.cpp, its kernels and the serving process. A range: it is not derived. */
  frameworkLoGb: number;
  frameworkHiGb: number;
  /** The final vocabulary projection for one micro-batch. Usually the largest piece. */
  logitsGb: number;
  /** Peak live activations through the widest part of the graph. */
  activationsGb: number;
  /** The KQ mask — the only piece that grows with context. */
  kqMaskGb: number;
  /** Everything summed, at the optimistic and pessimistic ends of the framework range. */
  loGb: number;
  hiGb: number;
}

/**
 * What the runtime costs on top of weights and KV cache — derived from what
 * llama.cpp allocates, not fitted to an observation.
 *
 * Three derived terms plus one that is not:
 *
 *   - the final vocabulary projection, `ubatch * vocabSize * 4`. This is the
 *     dominant piece and the one everybody misses. llama.cpp's own documented
 *     example is a 507MB compute buffer for Gemma 2 9B; 512 * 256000 * 4 is
 *     500MB. The projection IS the compute buffer, to within 1.4%.
 *   - peak live activations, `copies * ubatch * width * 4`, where width is the
 *     widest tensor a micro-batch passes through.
 *   - the KQ mask, `context * ubatch * 2`, which is the ONLY part that grows
 *     with context — about 1KB per token, so it behaves as an addition to the
 *     model's own KV bytes-per-token and keeps `maxContextFrom` closed-form.
 *   - the framework itself, which is not derivable and is therefore a range.
 *
 * This replaced `baseGb + 0.06 * weightsGb`, which used the wrong variable
 * altogether: it charged a 397B-A17B 15.6GB of scratch, and charged a model a
 * third as wide three times more than a wider one. Parameter count does not
 * size a compute buffer. Vocabulary, width and batch do.
 *
 * Returns null when the model lacks the fields to derive it, because a guessed
 * overhead is what this function exists to stop.
 */
export function overheadFor(
  model: Pick<Model, 'vocabSize' | 'width'>,
  contextTokens: number,
  hardware: Hardware,
): Overhead | null {
  const o = hardware.runtimeOverhead;
  if (!model.vocabSize || !model.width) return null;
  const logitsGb = (o.ubatch * model.vocabSize * 4) / BYTES_PER_GB;
  const activationsGb = (o.activationCopies * o.ubatch * model.width * 4) / BYTES_PER_GB;
  const kqMaskGb = (Math.max(0, contextTokens) * o.ubatch * o.kqBytesPerTokenPerUbatch) / BYTES_PER_GB;
  const derived = logitsGb + activationsGb + kqMaskGb;
  return {
    frameworkLoGb: o.frameworkGbLo,
    frameworkHiGb: o.frameworkGbHi,
    logitsGb,
    activationsGb,
    kqMaskGb,
    loGb: o.frameworkGbLo + derived,
    hiGb: o.frameworkGbHi + derived,
  };
}

/**
 * Extra bytes of context that the KQ mask costs per token, so callers solving
 * for a context length can add it to the model's own KV rate. Both are linear,
 * which is the reason `maxContextFrom` stays an inversion rather than a search.
 */
export function kqBytesPerToken(hardware: Hardware): number {
  const o = hardware.runtimeOverhead;
  return o.ubatch * o.kqBytesPerTokenPerUbatch;
}

/**
 * What running WITHOUT flash attention would add, in GB. Not part of any
 * estimate — every figure on this site assumes flash attention is on, which is
 * llama.cpp's default. Without it the attention scores are materialised rather
 * than masked, and the cost stops being a rounding error: about 2GB at 32K on a
 * 32-head model and 8GB at 128K. Returns null when the model has no head count.
 */
export function flashAttentionOffCostGb(
  model: Pick<Model, 'attentionHeads'>,
  contextTokens: number,
  hardware: Hardware,
): number | null {
  const o = hardware.runtimeOverhead;
  if (!model.attentionHeads) return null;
  const bytes =
    Math.max(0, contextTokens) *
    o.ubatch *
    model.attentionHeads *
    o.flashAttentionOffBytesPerTokenPerHead;
  return bytes / BYTES_PER_GB;
}

/**
 * Memory for display. Sub-gigabyte values get MB, because rounding a real 50MB
 * KV cache to "0.0GB" reads as missing data rather than as a small number —
 * and on a hybrid model like Nemotron 3 Nano, where only 6 of 52 layers cache,
 * that small number is the whole point.
 */
/**
 * There is no constant for the slow path any more, and the absence is the point.
 *
 * This used to price a spilled layer at PCIe 4.0 x16, ~32GB/s, on the theory
 * that its weights cross the bus every token. They do not. `-ngl` below the
 * layer count leaves those layers in system RAM and the CPU RUNS them there;
 * only activations, a few tens of kilobytes, cross PCIe. So the slow path is
 * the speed of the reader's system memory, which varies by a factor of nine
 * across the machines on this list and therefore cannot be a constant at all.
 *
 * The old constant was falsifiable from inside this file: the harmonic mean
 * against 32GB/s drops below 90GB/s once resident falls under about 67%, so
 * the site reported a machine with an RTX 3090 as slower than the same site's
 * machine with no graphics card at all. Fitting the two measured endpoints of
 * a published -ngl sweep instead (2.89 tok/s all-CPU, 43.18 all-GPU on a 14B)
 * predicts the measured 83%-resident point at 12.99 against 12.5 actual —
 * within 4%, with nothing fitted. The PCIe reading missed it by 26%.
 *
 * @see https://inventivehq.com/blog/vram-offload-cliff-gpu-layers-benchmark
 */

/**
 * The fixed per-token cost for this platform, in seconds. Everything that does
 * not scale with weight bytes: kernel launches, attention over the KV cache,
 * sampling, expert routing. See `perTokenLatency` in data/hardware.json.
 */
function perTokenLatency(hardware: Hardware, platform: Machine['platform']) {
  const p = hardware.perTokenLatency;
  const v = p.byPlatform?.[platform] ?? { loMs: p.loMs, hiMs: p.hiMs };
  return { lo: v.loMs / 1000, hi: v.hiMs / 1000 };
}

export interface Speed {
  /** Tokens per second, low and high end of the estimate. */
  lo: number;
  hi: number;
  /** What is actually read per token — the denominator, exposed for the page. */
  bytesReadGb: number;
  /** True when spilled layers drag the effective bandwidth down. */
  throttledByOffload: boolean;
}

/**
 * Generation speed, from the one thing that governs it: memory bandwidth.
 *
 * Producing a token means reading every active weight once and doing about two
 * FLOPs per byte, so this is bandwidth-bound and compute barely enters. Two
 * terms in the denominator:
 *
 *   - weights, scaled by the MoE active fraction. A 122B-A10B holds all 122B
 *     but touches a twelfth of it per token, which is the entire reason those
 *     models are worth their memory.
 *   - the KV cache, read in full on every token. Generic tok/s calculators drop
 *     this; here it is already computed exactly, so the estimate moves with the
 *     context the reader picked.
 *
 * Partial offload is a harmonic mean of two speeds, weighted by where the
 * bytes live. Time adds layer by layer: the card reads its share at full speed
 * while the CPU reads the rest at system-memory speed, and the slow term
 * dominates. That is the "several times slower" this site warns about, and it
 * stays a real slowdown now that the slow side is priced correctly — a 936GB/s
 * card half-offloaded to DDR5 runs at about 164GB/s effective. What it is NOT
 * is slower than the same machine with no graphics card in it.
 *
 * Returns null when the machine has no bandwidth on file: same fail-closed rule
 * as kvGb, because a guessed figure is worse than an absent one.
 */
export function generationSpeed(
  weightsGb: number,
  kvGb: number,
  model: Model,
  fit: Fit,
  bandwidthGbs: number | undefined,
  systemBandwidthGbs: number | undefined,
  hardware: Hardware,
  platform: Machine['platform'],
): Speed | null {
  if (!bandwidthGbs || bandwidthGbs <= 0) return null;
  const activeFraction = model.activeParamsB ? model.activeParamsB / model.paramsB : 1;
  const bytesReadGb = weightsGb * activeFraction + kvGb;
  if (bytesReadGb <= 0) return null;

  const resident = Math.min(1, Math.max(0, fit.residentFraction));
  const throttled = fit.offload && resident < 1;
  // Fail closed rather than guess at the reader's motherboard: a partial fit
  // whose slow half has no speed on file has no honest tok/s. Same rule kvGb
  // and bandwidthGbs already follow.
  if (throttled && (!systemBandwidthGbs || systemBandwidthGbs <= 0)) return null;
  const effectiveBw =
    throttled && systemBandwidthGbs
      ? 1 / (resident / bandwidthGbs + (1 - resident) / systemBandwidthGbs)
      : bandwidthGbs;

  // Time per token adds: the streaming part, plus a fixed cost that does not
  // scale with it. A big read is dominated by bandwidth; a small one — a 4-bit
  // model, or a sparse MoE reading a tenth of itself — is dominated by the
  // overhead, which is exactly the case the old bandwidth-fraction flattered.
  // Charged per layer: kernel dispatch happens per layer, which is why a flat
  // per-token figure fits a 7B and misses a 70B by a factor of three.
  const perLayer = perTokenLatency(hardware, platform);
  const layers = model.layers;
  if (!layers) return null; // fail closed, as everywhere else
  const c = { lo: perLayer.lo * layers, hi: perLayer.hi * layers };
  const stream = (bytesReadGb * BYTES_PER_GB) / (effectiveBw * 1e9);
  const round = (n: number) => (n >= 10 ? Math.round(n) : Math.round(n * 10) / 10);
  return {
    lo: round(1 / (stream + c.hi)),
    hi: round(1 / (stream + c.lo)),
    bytesReadGb: Math.round(bytesReadGb * 10) / 10,
    throttledByOffload: throttled,
  };
}

/**
 * The highest cap this site OFFERS, in GB. Not the highest that is safe: nobody
 * knows that, Apple documents the setting nowhere, and this project is not an
 * authority on what a stranger's machine survives.
 *
 * The arithmetic holds back `max(minGb, minGb * sqrt(ram / ref))` — 2GB on a
 * 32GB machine, 8GB on a 512GB one. Sub-linear because the OS does not need
 * proportionally more memory as the machine grows. It is deliberately more
 * aggressive than published guidance, which is why `publishedReserveGb` exists
 * alongside it and why every option past that figure is marked as past it.
 *
 * Returns null where the platform has no adjustable cap.
 */
export function maxWiredGb(machine: Machine, hardware: Hardware): number | null {
  const platform = platformFor(hardware, machine.platform);
  if (!platform.ceiling.overridable || !platform.ceiling.raise) return null;
  // The same reserve budgetFor holds back, and only that one.
  //
  // There used to be a second: a sqrt-scaled figure that offered a 32GB Mac
  // 30GB while the budget would only ever spend 25.6GB. Two of the dropdown's
  // three options therefore changed nothing at all, and the command printed a
  // number the site did not model. It was also uncited, which is what put the
  // whole raise range under review.
  //
  // Checked across every capacity: the OS reserve was larger at all of them, so
  // the sqrt figure never once bound. It was dead arithmetic making a claim.
  // Removing it leaves one reserve, and the offer, the budget, the command and
  // the prose all read from it.
  return Math.max(0, Math.round(machine.ramGb - reserveGb(platform, machine.ramGb)));
}

/**
 * The ceiling a cited source recommends for this machine, in GB, or null when
 * no published figure covers it. Interpolates nothing and extrapolates nothing:
 * a capacity the source does not name gets no answer rather than a guess, which
 * is the whole distinction this function exists to keep.
 */
export function publishedCeilingGb(machine: Machine, hardware: Hardware): number | null {
  const { ceiling } = platformFor(hardware, machine.platform);
  const table = ceiling.raise?.publishedReserveGb;
  const reserve = table?.[String(machine.ramGb)];
  return reserve === undefined ? null : Math.max(0, Math.round(machine.ramGb - reserve));
}

/**
 * Raise options between the platform default and the highest this site offers.
 * Built from the gap rather than from the shared RAM ladder, which offered
 * exactly one step on most machines and made it the whole of memory.
 */
export function raiseSteps(machine: Machine, hardware: Hardware): number[] {
  const max = maxWiredGb(machine, hardware);
  if (max === null) return [];
  const { ceiling } = platformFor(hardware, machine.platform);
  const dflt = Math.round((ceiling.value ?? 0.75) * machine.ramGb);
  if (max <= dflt) return [];
  const increment = Math.max(1, Math.round((max - dflt) / 4));
  const steps: number[] = [];
  for (let g = dflt + increment; g < max; g += increment) steps.push(g);
  steps.push(max);
  return [...new Set(steps)];
}

/** Prompt-processing class: the preset's override, else the chip's, else the platform's default. */
export function prefillFor(machine: Machine, hardware: Hardware): string | undefined {
  const preset = matchPreset(machine, hardware);
  return (
    preset?.prefillClass ?? chipFor(machine, hardware)?.prefillClass ?? platformFor(hardware, machine.platform).prefillClass
  );
}

/**
 * The preset a described machine corresponds to. Prefers the explicit id, since
 * the numbers alone cannot identify a machine — eight presets are `discrete`
 * with 24GB of VRAM and their bandwidths span 288 to 1008 GB/s. Falls back to a
 * numeric match for a hand-described machine, which is a guess and is only used
 * for figures those machines broadly agree on.
 */
function matchPreset(machine: Machine, hardware: Hardware) {
  if (machine.presetId) {
    const exact = hardware.presets.find((x) => x.id === machine.presetId);
    if (exact) return exact;
  }
  return hardware.presets.find(
    (x) =>
      x.platform === machine.platform &&
      x.ramGb === machine.ramGb &&
      (x.vramGb ?? undefined) === (machine.vramGb ?? undefined),
  );
}

/**
 * One entry per Apple machine that exists: every chip crossed with every
 * capacity it shipped with, 61 of them.
 *
 * Derived rather than stored, because the chip table already holds every fact
 * involved — writing them into hardware.json would be a second copy that can
 * disagree with the first, which is how the last three commits' bugs happened.
 * Capacity alone cannot identify a Mac, so the machine list has to be the
 * chip list; there is no shorter honest version.
 */
export function chipPresets(hardware: Hardware): Preset[] {
  return hardware.chips.flatMap((c) =>
    c.ramOptions.map((ramGb) => ({
      id: `${c.id}-${ramGb}`,
      label: `${c.label} · ${ramGb}GB`,
      group: 'Apple silicon',
      subgroup: c.family,
      platform: c.platform,
      ramGb,
      chipId: c.id,
      note: c.note,
      noteUrl: c.noteUrl,
    })),
  );
}

/** Every preset that holds the same amount of memory as this machine. */
export function sameCapacityPresets(machine: Machine, hardware: Hardware) {
  const cap = machine.vramGb ?? machine.ramGb;
  const discrete = machine.platform === 'discrete';
  return hardware.presets.filter((p) => {
    const pcap = p.vramGb ?? p.ramGb;
    return pcap === cap && (p.platform === 'discrete') === discrete;
  });
}

/** The chip a machine is running, when its platform has a choice of one. */
export function chipFor(machine: Machine, hardware: Hardware) {
  const id = machine.chipId ?? matchPreset(machine, hardware)?.chipId;
  return id ? hardware.chips.find((c) => c.id === id) : undefined;
}

/**
 * Bandwidth for a machine, in the order the information is trustworthy: the
 * chip the reader named, then the chip their preset implies, then the preset's
 * own figure — which only discrete cards carry, because a card is one piece of
 * silicon and a Mac capacity is not.
 */
export function bandwidthFor(machine: Machine, hardware: Hardware): number | undefined {
  const chip = chipFor(machine, hardware);
  if (chip) {
    // The M6 reads slower with 16GB fitted than with 24 or more. Only chip in
    // the table that varies by capacity, but it does, so it is honoured.
    return chip.bandwidthAtRamGb?.[String(machine.ramGb)] ?? chip.bandwidthGbs;
  }
  // With no graphics card there is no second tier: the system memory is the
  // whole answer, so it is read from the one table that describes it rather
  // than copied onto every CPU preset, where the copy would drift.
  if (machine.platform === 'cpu') return systemBandwidthFor(machine, hardware);
  return matchPreset(machine, hardware)?.memoryBandwidthGbs;
}

/**
 * How fast this machine's CPU reads system RAM, GB/s.
 *
 * On a machine with a graphics card too small for the model, this is the speed
 * of every layer the card could not take — half of what decides the answer, and
 * until now the half the site did not model at all. On a machine with no card
 * it is the whole of it.
 *
 * Resolved from what the reader said, then from the machine they picked, then
 * from the platform's default. Returns undefined only when the table itself is
 * missing an id, which validate.ts makes a build error.
 */
export function systemBandwidthFor(machine: Machine, hardware: Hardware): number | undefined {
  // Shared-pool machines have no such thing as a separate system memory, and
  // answering anyway is a trap. On Apple Silicon this used to return the
  // DDR5-desktop default of 90GB/s for a machine whose memory runs at 400 —
  // wrong by more than four times, and sitting one code change away from being
  // displayed. There is no honest number to give here, so give none.
  if (!hasSeparateSystemMemory(machine.platform)) return undefined;
  const id = machine.systemMemoryId ?? matchPreset(machine, hardware)?.systemMemoryId;
  const table = hardware.systemMemory;
  const entry = id ? table.find((m) => m.id === id) : undefined;
  return (entry ?? table.find((m) => m.id === DEFAULT_SYSTEM_MEMORY_ID) ?? table[0])?.bandwidthGbs;
}

/**
 * Whether this platform's CPU has memory of its own, separate from the GPU's.
 *
 * True for a graphics card in a tower and for a machine with no card at all.
 * False for every unified-memory design, and that difference is the whole
 * reason Macs behave differently everywhere in this file.
 */
export function hasSeparateSystemMemory(platform: Machine['platform']): boolean {
  return platform === 'discrete' || platform === 'cpu';
}

/** What an unspecified machine is assumed to have: two channels of DDR5. */
export const DEFAULT_SYSTEM_MEMORY_ID = 'ddr5-dual';

/**
 * The partial fit restated as the number the reader actually types.
 *
 * `residentFraction` is a byte ratio, which is the right thing to compute speed
 * from and the wrong thing to show someone: nobody sets 62% in a runtime, they
 * set `-ngl 39`. Layers are near enough uniform in a transformer that dividing
 * the model's own layer count by its own weight gives a per-layer cost worth
 * quoting, once the fixed runtime overhead is taken off the top.
 *
 * Deliberately approximate, and labelled as such wherever it is shown: the
 * embedding and output tensors are not layers and do not divide evenly, and
 * quantized builds vary the per-layer size. It is a starting value to tune, not
 * a setting to trust — which is exactly how every runtime guide describes it.
 *
 * Returns null when the model has no layer count on file, or when nothing is
 * spilling and the question does not arise.
 */
export function layerSplit(
  model: Pick<Model, 'layers'>,
  fit: Fit,
): { onGpu: number; onCpu: number; total: number } | null {
  const total = model.layers;
  if (!total || !fit.offload) return null;
  const perLayer = (fit.weightsGb + fit.kvGb) / total;
  if (perLayer <= 0) return null;
  const forLayers = fit.budgetGb - fit.overheadGb;
  const onGpu = Math.max(0, Math.min(total, Math.floor(forLayers / perLayer)));
  return { onGpu, onCpu: total - onGpu, total };
}

/**
 * Capacities to offer for a machine — exactly what its chip was sold with, so
 * "M1 Max 48GB" stops being reachable. Falls back to the full ladder for a
 * machine with no chip, where we have nothing to narrow it by.
 */
export function ramOptionsFor(machine: Machine, hardware: Hardware): number[] {
  return chipFor(machine, hardware)?.ramOptions ?? hardware.ramOptions;
}

export function formatMemory(gb: number): string {
  if (gb <= 0) return '0GB';
  if (gb < 1) return `${Math.max(1, Math.round(gb * 1024))}MB`;
  return `${Math.round(gb * 10) / 10}GB`;
}

/** A model cannot use more context than it supports, whatever the task wants. */
export function effectiveContext(model: Model, requestedTokens: number): number {
  return Math.min(requestedTokens, model.contextLength);
}

export function fitFor(
  model: Model,
  band: Band,
  machine: Machine,
  hardware: Hardware,
  requestedContextTokens: number,
): Fit | null {
  const info = model.bands[band];
  if (!info) return null;
  const contextTokens = effectiveContext(model, requestedContextTokens);
  const kv = kvGb(model, contextTokens);
  if (kv === null) return null;

  const budget = budgetFor(machine, hardware);
  const oh = overheadFor(model, contextTokens, hardware);
  if (oh === null) return null; // same fail-closed rule as an unmeasured KV figure
  // The pessimistic end for anything that decides whether it FITS: the framework
  // figure is a published range, not a measurement, and a fit promised on the
  // optimistic end is a fit the reader may not get.
  const overhead = oh.hiGb;
  const demand = info.weightsGb + kv + overhead;
  const headroom = budget.gb - demand;

  const t = hardware.thresholds;
  const floor = Math.max(t.tightBelowHeadroomGb, t.tightBelowHeadroomFraction * budget.gb);

  const offload = demand > budget.gb && demand <= budget.offloadGb;
  let state: FitState;
  if (demand > budget.offloadGb) state = 'no';
  else if (offload) state = 'tight'; // a speed cliff, not a headroom shortage
  else if (demand > t.tightAtFractionOfBudget * budget.gb || headroom < floor) state = 'tight';
  else state = 'comfortable';

  // Would a bigger ceiling alone rescue it? Only meaningful where it's adjustable.
  const spareRam = Math.max(0, machine.ramGb - budget.reserveGb);
  const ceilingBlocked =
    state === 'no' &&
    budget.platform.ceiling.overridable === true &&
    budget.ceilingBinds &&
    demand <= spareRam;

  // What the leftover buys, after the runtime has taken its share. Based on the
  // resident budget, not the offload ceiling — context living in system RAM is
  // not context you want.
  //
  // The KQ mask is part of that share and grows with the very quantity being
  // solved for, so it is added to the per-token rate rather than subtracted as
  // a constant. Both terms are linear in context, so this stays an inversion.
  // The mask is already inside `overhead` at the requested context, so take it
  // back out before solving or it is charged twice.
  const kqRate = kqBytesPerToken(hardware);
  const contextRange = (frameworkGb: number) =>
    Math.min(
      model.contextLength,
      maxContextFrom(
        budget.gb - info.weightsGb - (frameworkGb + oh.logitsGb + oh.activationsGb),
        model.kvBytesPerToken! + kqRate,
        model.kvWindowedBytesPerToken,
        model.slidingWindow,
      ),
    );
  const maxContextTokens = contextRange(oh.frameworkHiGb);
  const maxContextTokensOptimistic = contextRange(oh.frameworkLoGb);

  const residentFraction = demand > 0 ? Math.min(1, budget.gb / demand) : 1;

  return {
    state,
    demandGb: Math.round(demand * 10) / 10,
    residentFraction,
    spilledGb: Math.max(0, Math.round((demand - budget.gb) * 10) / 10),
    weightsGb: info.weightsGb,
    kvGb: Math.round(kv * 1000) / 1000,
    overheadGb: Math.round(overhead * 10) / 10,
    overhead: oh,
    budgetGb: budget.gb,
    headroomGb: Math.round(headroom * 10) / 10,
    offload,
    ceilingBlocked,
    contextTokens,
    maxContextTokens,
    maxContextTokensOptimistic,
  };
}

/**
 * The best band reachable if partial offload is allowed, subject to a floor on
 * how much must stay in fast memory. Paired with `bestFit`, this answers the
 * question system RAM exists to answer: what quality could I step up to by
 * streaming part of the model?
 */
export function bestOffloadFit(
  model: Model,
  machine: Machine,
  hardware: Hardware,
  contextTokens: number,
  minResidentFraction: number,
): { band: Band; fit: Fit } | null {
  const rated = BANDS.map((band) => ({ band, fit: fitFor(model, band, machine, hardware, contextTokens) }))
    .filter((x): x is { band: Band; fit: Fit } =>
      x.fit !== null && x.fit.state !== 'no' && x.fit.residentFraction >= minResidentFraction);
  return rated[rated.length - 1] ?? null;
}

/**
 * The largest band this machine can hold, and how comfortably. Memory only —
 * this knows nothing about scores.
 *
 * NOT what the picker recommends: `bestScoredFit` in picker.ts chooses the band
 * by score, because the largest band that fits is frequently not the best one.
 * This remains the right primitive where no task is in play — the peer-machine
 * speed table, and the fit tests.
 *
 * Taking the highest runnable band regardless produced bad advice on discrete
 * GPUs: an 11GB card was offered a 31GB model at Q6–Q8 because it technically
 * "runs" with most of its layers handed to the CPU. It does, at a few tokens a
 * second. A band that fits in memory beats a larger one that crawls.
 */
export function bestFit(
  model: Model,
  machine: Machine,
  hardware: Hardware,
  contextTokens: number,
): { band: Band; fit: Fit } | null {
  const rated = BANDS.map((band) => ({ band, fit: fitFor(model, band, machine, hardware, contextTokens) }))
    .filter((x): x is { band: Band; fit: Fit } => x.fit !== null && x.fit.state !== 'no');

  // 1. Anything that fits in memory — take the best of them, tight or not.
  //    "Tight" only means low headroom, and the card says so. Trading two quant
  //    bands for spare gigabytes is a worse deal than running close to the line:
  //    preferring comfortable here once dropped a 32GB Mac from Q4-Q5 to Q2-Q3.
  const resident = rated.filter((x) => !x.fit.offload);
  if (resident.length > 0) return resident[resident.length - 1]!;

  // 2. Only offloading options left. Now take the SMALLEST: every extra
  //    gigabyte past VRAM is another layer the CPU has to run, so the biggest
  //    band is the slowest. This is a speed cliff, not a headroom shortage,
  //    which is why it is treated differently from the case above.
  return rated[0] ?? null;
}
