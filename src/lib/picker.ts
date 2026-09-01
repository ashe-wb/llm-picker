import { BANDS, BAND_DESCRIPTIONS, BAND_LABELS } from './constants';
import type {
  Band,
  DimensionId,
  Mitigation,
  Model,
  ScoresFile,
  Severity,
  Hardware,
  HardwareAdvice,
  Machine,
  Task,
  WeaknessEntry,
  WeaknessesFile,
} from './schemas';
import {
  bandwidthFor,
  bestFit,
  bestOffloadFit,
  budgetFor,
  budgetSource,
  fitFor,
  generationSpeed,
  maxWiredGb,
  prefillFor,
  systemBandwidthFor,
  type Budget,
  type Fit,
  type FitState,
  type Speed,
} from './fit';

/**
 * Everything the picker needs, pre-baked at build time and inlined into the
 * page as JSON. Deliberately independent of data.ts so the client bundle
 * never pulls in node:fs.
 */
export interface PickerData {
  hardware: Hardware;
  tasks: Task[];
  mitigations: Mitigation[];
  models: Model[];
  scores: Record<string, ScoresFile>;
  weaknesses: Record<string, WeaknessesFile>;
}

export interface MitigationChip {
  mitigationId: string;
  label: string;
  note: string;
  effectiveness?: 'strong' | 'partial' | 'weak';
  /** True when the fix depends on a capability this model is weak at. */
  limited: boolean;
  /** True for the honest "no known fix" entry. */
  noFix: boolean;
}

export interface Warning {
  entry: WeaknessEntry;
  chips: MitigationChip[];
}

export interface Recommendation {
  model: Model;
  band: Band;
  bandLabel: string;
  /**
   * Task-weighted 0–10 used for ranking and display, discounted for missing
   * evidence — see `NEUTRAL_SCORE`. Never higher than `measuredScore`.
   */
  score: number;
  /** The plain mean over rated dimensions only. What `score` was before v2. */
  measuredScore: number;
  /** Share of the task's weight that actually has evidence, 0–1. */
  coverage: number;
  /** Share of task weight with no evidence at all. Drives `incompleteData`. */
  missingWeight: number;
  incompleteData: boolean;
  /** How comfortably it runs, and why. */
  fit: Fit;
  fitState: FitState;
  /** Context this verdict assumes — the task's ask, clamped to the model's max. */
  assumedContextTokens: number;
  warnings: Warning[];
  tags: Array<'top-pick' | 'lightest'>;
  /**
   * A top pick that is the best available and still not good.
   *
   * True when the leading recommendation scores below NEUTRAL_SCORE — the value
   * this file already assigns to a dimension with NO evidence at all. So this
   * fires when the best model a machine can run scores worse than the site's own
   * placeholder for total ignorance, which is the honest definition of weak and
   * is not a threshold picked by taste.
   *
   * Ranking is unaffected. The pick is still correct; what changes is that the
   * card stops stamping "Top pick" on a 4/10 and calling it a recommendation.
   */
  weakBest?: boolean;
  /**
   * Why this one, when the score alone can't say. Populated only for models
   * that tie for the top slot — see `qualify`.
   */
  qualifiers: string[];
  /** Set on co-picks when more than two models tie. */
  tieNote?: string;
  /**
   * A larger band that also fit and was passed over. Carried so the card can
   * show its working: a reader whose 80GB card can obviously hold FP16 should
   * see why it was not chosen, rather than assume the site got it wrong.
   */
  skippedBand?: { band: Band; label: string; demandGb: number; scoreDelta: number };
}

const SEVERITY_RANK: Record<Severity, number> = { severe: 2, moderate: 1, mild: 0 };

/**
 * What an unrated dimension is worth when ranking. Renormalising over only the
 * dimensions that have evidence rewards ignorance: a model rated on 10% of a
 * task's weight, scoring 8 there, used to beat one rated on 70% averaging 7.
 * That is an implicit claim that the 90% nobody measured is as good as the 10%
 * somebody did.
 *
 * So unmeasured weight counts as merely average instead. No cell is invented and
 * nothing is stored — model sheets are unchanged and still show blanks. This
 * only decides what gets recommended, and it is a weaker assumption than the one
 * it replaces.
 */
export const NEUTRAL_SCORE = 5;

/** Flagged on the card. Equivalent to the previous `missingWeight > 0.3` rule. */
const LOW_COVERAGE = 0.7;

/**
 * Score lookup with band fallback: exact band, else the nearest band that has
 * this dimension, minus 1 point per band of distance (clamped to 0).
 * Returns null when no band has evidence for the dimension.
 */
export function scoreFor(
  scores: ScoresFile | undefined,
  band: Band,
  dimension: DimensionId,
): { score: number; borrowed: boolean } | null {
  if (!scores) return null;
  const at = (b: Band) => scores.entries.find((e) => e.band === b && e.dimension === dimension);

  const exact = at(band);
  if (exact) return { score: exact.score, borrowed: false };

  const idx = BANDS.indexOf(band);
  for (let dist = 1; dist < BANDS.length; dist++) {
    // Prefer the lower (more conservative) band on ties.
    for (const j of [idx - dist, idx + dist]) {
      const candidate = j >= 0 && j < BANDS.length ? at(BANDS[j]!) : undefined;
      if (candidate) return { score: Math.max(0, candidate.score - dist), borrowed: true };
    }
  }
  return null;
}

function buildChips(
  entry: WeaknessEntry,
  band: Band,
  scores: ScoresFile | undefined,
  mitigations: Mitigation[],
): MitigationChip[] {
  return entry.mitigations.map((m) => {
    const lib = mitigations.find((x) => x.id === m.mitigationId);
    let limited = false;
    if (lib?.worksVia) {
      const capability = scoreFor(scores, band, lib.worksVia);
      limited = (capability?.score ?? 0) < 4;
    }
    return {
      mitigationId: m.mitigationId,
      label: lib?.label ?? m.mitigationId,
      note: m.note,
      effectiveness: m.effectiveness,
      limited,
      noFix: m.mitigationId === 'none',
    };
  });
}

/* -------------------------------------------------------------------------- */
/* Tie qualifiers                                                             */
/* -------------------------------------------------------------------------- */

/** Effective inference cost: an MoE only activates a fraction of its weights. */
function activeParams(r: Recommendation): number {
  return r.model.activeParamsB ?? r.model.paramsB;
}

/** Worst task-relevant weakness at this band; -1 when there are none. */
function worstSeverity(r: Recommendation): number {
  return r.warnings.reduce((w, x) => Math.max(w, SEVERITY_RANK[x.entry.severity]), -1);
}

function isPermissive(r: Recommendation): boolean {
  return /^(apache|mit|bsd)/i.test(r.model.license.name);
}

/** 262144 → "256K". */
function ctxLabel(n: number): string {
  return n >= 1024 ? `${Math.round(n / 1024)}K` : String(n);
}

/**
 * Axes used to separate models whose aggregate scores are equal, ordered by
 * how much they should sway a choice. `value` is higher-is-better; an axis is
 * only claimed when one model beats every rival on it.
 */
interface Axis {
  value: (r: Recommendation) => number;
  /** Extra margin requirement beyond "strictly greater". */
  decisive?: (best: number, next: number) => boolean;
  phrase: (r: Recommendation, rivals: Recommendation[]) => string;
  /** Verb phrase describing the winner, for the model that wins nothing. */
  comparative: string;
}

const AXES: Axis[] = [
  {
    value: (r) => -activeParams(r),
    // A speed claim needs a real gap, not 27B vs 26B.
    decisive: (best, next) => -best <= 0.8 * -next,
    phrase: (r, rivals) =>
      r.model.activeParamsB !== undefined
        ? `Faster — ${r.model.activeParamsB}B active of ${r.model.paramsB}B`
        : `Faster — ${r.model.paramsB}B vs ${Math.max(...rivals.map(activeParams))}B active`,
    comparative: 'runs faster',
  },
  {
    value: (r) => r.model.contextLength,
    phrase: (r, rivals) =>
      `Bigger window — ${ctxLabel(r.model.contextLength)} vs ${ctxLabel(
        Math.max(...rivals.map((x) => x.model.contextLength)),
      )}, better for whole-repo work`,
    comparative: 'has a bigger context window',
  },
  {
    value: (r) => -worstSeverity(r),
    phrase: (r) =>
      worstSeverity(r) < 0
        ? 'No known issues for this workload at this quant band'
        : 'Milder known issues for this workload',
    comparative: 'has fewer known issues here',
  },
  {
    value: (r) => -r.fit.demandGb,
    phrase: (r, rivals) =>
      `Lighter — ${r.fit.demandGb}GB vs ${Math.max(...rivals.map((x) => x.fit.demandGb))}GB at this context`,
    comparative: 'needs less memory',
  },
  {
    value: (r) => (r.model.vision ? 1 : 0),
    phrase: () => 'Reads images and screenshots',
    comparative: 'reads images',
  },
  {
    value: (r) => -r.missingWeight,
    phrase: () => 'Better evidenced — fewer unrated dimensions for this task',
    comparative: 'has more evidence behind it',
  },
  {
    value: (r) => (isPermissive(r) ? 1 : 0),
    phrase: (r) => `Permissive license — ${r.model.license.name}`,
    comparative: 'has a more permissive license',
  },
];

/** Indices of the axes `r` wins outright against everyone else in `set`. */
function axesWon(r: Recommendation, set: Recommendation[]): number[] {
  const rivals = set.filter((x) => x !== r);
  if (rivals.length === 0) return [];
  return AXES.reduce<number[]>((won, axis, i) => {
    const mine = axis.value(r);
    const best = Math.max(...rivals.map(axis.value));
    if (mine > best && (!axis.decisive || axis.decisive(mine, best))) won.push(i);
    return won;
  }, []);
}

/**
 * Why pick this one over the others it tied with. Returns the two strongest
 * differentiators, or — when a model wins nothing — an honest line saying so
 * rather than inventing an advantage.
 */
export function qualify(r: Recommendation, tied: Recommendation[]): string[] {
  const rivals = tied.filter((x) => x !== r);
  if (rivals.length === 0) return [];

  const won = axesWon(r, tied);
  if (won.length > 0) return won.slice(0, 2).map((i) => AXES[i]!.phrase(r, rivals));

  // Dominated: name what it gives up, and to whom.
  const lostTo = rivals.find((x) => axesWon(x, tied).length > 0);
  const beat = lostTo ? axesWon(lostTo, tied).slice(0, 2).map((i) => AXES[i]!.comparative) : [];
  if (!lostTo || beat.length === 0) {
    return ['Nothing separates these two in our data — pick either.'];
  }
  return [
    `No measured edge here — same score, but ${lostTo.model.name} ${beat.join(' and ')}.`,
  ];
}

/** Context ladder used when nothing runs the workload as specified. */
const FALLBACK_CONTEXTS = [131072, 32768, 16384, 8192, 4096];

export interface PickResult {
  recommendations: Recommendation[];
  /** Context these results were actually computed at. */
  contextTokens: number;
  /** The task's original ask, when we had to fall back to something shorter. */
  fallbackFrom?: number;
  budget: Budget;
  /**
   * Models that run only by streaming part of themselves from system RAM.
   * Ranked separately: slower, but on a machine with plenty of RAM they are
   * real options rather than noise.
   */
  offloadOptions: Recommendation[];
  /**
   * The best-scoring option that beats a weak top pick, when there is one.
   * Already computed as an offload option; surfaced separately so the card can
   * name it rather than leaving the reader to compare two lists themselves.
   * Undefined when the top pick is fine, or when nothing on offer does better.
   */
  betterThanWeak?: Recommendation;
  /** Advice shown only when raising the ceiling would change these results. */
  ceilingAdvice?: HardwareAdvice;
  /** What raising it would actually buy, e.g. "Gemma 4 26B-A4B at FP16 instead of Q6–Q8". */
  ceilingGain?: string;
}

/** Build one recommendation for a model at a specific band. */
function buildRecommendation(
  model: Model,
  band: Band,
  fit: Fit,
  task: Task,
  data: PickerData,
): Recommendation | null {
  const scores = data.scores[model.id];
  let acc = 0;
  let usedWeight = 0;
  let missingWeight = 0;
  for (const [dim, weight] of Object.entries(task.weights) as [DimensionId, number][]) {
    const s = scoreFor(scores, band, dim);
    if (s === null) missingWeight += weight;
    else {
      acc += weight * s.score;
      usedWeight += weight;
    }
  }
  if (usedWeight === 0) return null; // no evidence at all — can't recommend honestly

  const measured = acc / usedWeight;
  const coverage = usedWeight / (usedWeight + missingWeight);
  const adjusted = measured * coverage + NEUTRAL_SCORE * (1 - coverage);

  const weakFile = data.weaknesses[model.id];
  const warnings: Warning[] = (weakFile?.entries ?? [])
    .filter((e) => e.bands.includes(band) && e.dimensions.some((d) => (task.weights[d] ?? 0) >= 0.2))
    .sort((a, b) => SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity])
    .map((entry) => ({ entry, chips: buildChips(entry, band, scores, data.mitigations) }));

  return {
    model,
    band,
    bandLabel: `${BAND_LABELS[band]} (${BAND_DESCRIPTIONS[band]})`,
    score: Math.round(adjusted * 10) / 10,
    measuredScore: Math.round(measured * 10) / 10,
    coverage,
    missingWeight,
    incompleteData: coverage < LOW_COVERAGE,
    fit,
    fitState: fit.state,
    assumedContextTokens: fit.contextTokens,
    warnings,
    tags: [],
    qualifiers: [],
  };
}

/**
 * The band to actually run, chosen by score rather than by size.
 *
 * `bestFit` takes the largest band that fits, which was wrong in two ways once
 * the site had per-band scores to check it against. It picked bands with no
 * measured evidence — `scoreFor` borrows from a neighbour and docks a point per
 * band of distance, so those score BELOW the band they borrowed from, and the
 * picker showed them anyway. And where two bands score the same it took the
 * expensive one: 6-8 bit is lossless on 16 of the 17 models scored at both, so
 * an H100 was told to run a 30B at FP16 for 39GB more memory and half the speed
 * than Q6-Q8, for no measured gain at all.
 *
 * Scoring every band that fits and taking the best fixes both, and the tie-break
 * toward the smaller band is what turns "no worse" into "cheaper and faster".
 * Where the evidence does favour a bigger band it still wins — nanbeige4.2-3b
 * keeps FP16, because its math score really is a point higher there.
 */
export function bestScoredFit(
  model: Model,
  machine: Machine,
  task: Task,
  data: PickerData,
  contextTokens: number,
): Recommendation | null {
  const rated = BANDS.map((band) => ({
    band,
    fit: fitFor(model, band, machine, data.hardware, contextTokens),
  })).filter((x): x is { band: Band; fit: Fit } => x.fit !== null && x.fit.state !== 'no');
  if (rated.length === 0) return null;

  // Offloading keeps the old rule: the SMALLEST band, because past VRAM every
  // extra gigabyte is another layer the CPU has to run at system-memory speed.
  // That is a speed cliff, not a question about quality, so no amount of score
  // justifies climbing it.
  const resident = rated.filter((x) => !x.fit.offload);
  const pool = resident.length > 0 ? resident : rated.slice(0, 1);

  const recs = pool
    .map((x) => buildRecommendation(model, x.band, x.fit, task, data))
    .filter((r): r is Recommendation => r !== null);
  if (recs.length === 0) return null;

  recs.sort((a, b) => b.score - a.score || BANDS.indexOf(a.band) - BANDS.indexOf(b.band));
  const winner = recs[0]!;

  // Name the largest band that fit and lost, so the card can show its working.
  const passedOver = recs
    .filter((r) => BANDS.indexOf(r.band) > BANDS.indexOf(winner.band))
    .sort((a, b) => BANDS.indexOf(b.band) - BANDS.indexOf(a.band))[0];
  if (passedOver) {
    winner.skippedBand = {
      band: passedOver.band,
      label: BAND_LABELS[passedOver.band],
      demandGb: passedOver.fit.demandGb,
      scoreDelta: Math.round((winner.score - passedOver.score) * 10) / 10,
    };
  }
  return winner;
}

function collect(
  machine: Machine,
  task: Task,
  data: PickerData,
  contextTokens: number,
  /** Filled with the offloading alternatives, ranked. Left empty when unused. */
  offloadPool: Recommendation[] = [],
): Recommendation[] {
  const candidates: Recommendation[] = [];
  const offloadCandidates: Recommendation[] = [];

  const floor = data.hardware.thresholds.offloadMinResidentFraction;

  for (const model of data.models) {
    if (model.status === 'deprecated') continue;
    if (task.requires?.vision && !model.vision) continue;

    // Fit is computed, not looked up: weights + real KV against a real budget.
    // The band is then chosen by score, not by size.
    const best = bestScoredFit(model, machine, task, data, contextTokens);
    if (best) candidates.push(best);

    // What system RAM buys: the best band reachable by streaming part of the
    // model, when that is BETTER than what fits outright. Without this a model
    // with any resident band never appeared as an offload option, and RAM
    // changed nothing.
    const upgrade = bestOffloadFit(model, machine, data.hardware, contextTokens, floor);
    if (upgrade?.fit.offload) {
      const rec = buildRecommendation(model, upgrade.band, upgrade.fit, task, data);
      // Must be a real upgrade in SCORE, not merely a larger band. Moving
      // layers onto the CPU to reach an equal-scoring band buys nothing.
      if (rec && (!best || rec.score > best.score)) offloadCandidates.push(rec);
    }
  }

  const resident = candidates.filter((c) => !c.fit.offload);
  const offloadable = [...candidates.filter((c) => c.fit.offload), ...offloadCandidates]
    .filter((c) => c.fit.residentFraction >= floor)
    .sort((a, b) => b.score - a.score);

  // Nothing resident? Promote the offloaders rather than showing an empty page.
  const pool = resident.length > 0 ? resident : offloadable;
  pool.sort((a, b) => b.score - a.score);
  candidates.length = 0;
  candidates.push(...pool);
  offloadPool.length = 0;
  offloadPool.push(...(resident.length > 0 ? offloadable : []));
  if (candidates.length === 0) return [];

  // Models sharing the top score are contiguous at the front after the sort.
  const tied = candidates.filter((c) => c.score === candidates[0]!.score);
  if (tied.length > 1) {
    // Rank the tie so the strongest members are the ones that get shown.
    const won = new Map(tied.map((r) => [r, axesWon(r, tied).length]));
    tied.sort(
      (a, b) =>
        won.get(b)! - won.get(a)! ||
        a.fit.demandGb - b.fit.demandGb ||
        a.model.name.localeCompare(b.model.name),
    );
    candidates.splice(0, tied.length, ...tied);
  }

  const top = candidates.slice(0, 3);

  // Up to two models may share the badge; beyond that the axes above decide.
  const coPicks = tied.slice(0, 2);
  for (const r of coPicks) {
    r.tags.push('top-pick');
    r.weakBest = r.score < NEUTRAL_SCORE;
  }
  if (tied.length > 1) {
    for (const r of coPicks) r.qualifiers = qualify(r, coPicks);
    if (tied.length > 2) {
      const others = tied.length - 2;
      for (const r of coPicks) {
        r.tieNote = `Tied at ${r.score} with ${others} other model${others === 1 ? '' : 's'} — the two shown lead on the differences below.`;
      }
    }
  }

  // Lightest good option: smallest footprint within 15% of the top score.
  const threshold = top[0]!.score * 0.85;
  const lightest = candidates
    .filter((c) => c.score >= threshold)
    .sort((a, b) => a.fit.demandGb - b.fit.demandGb)[0];
  // Never on a co-pick: it already carries the badge, and its "Lighter — …"
  // qualifier makes the same point.
  if (lightest && !coPicks.includes(lightest)) {
    if (!top.includes(lightest)) top.push(lightest);
    lightest.tags.push('lightest');
  }

  return top;
}

/**
 * What a bigger memory ceiling would actually buy, in the user's terms — a better
 * quant band for something already recommended, or a better model entering the
 * list. Returns undefined when the results would be identical.
 */
function firstImprovement(
  now: Recommendation[],
  raised: Recommendation[],
): string | undefined {
  for (const r of raised) {
    const same = now.find((x) => x.model.id === r.model.id);
    // A bigger band is only worth buying memory for if it scores better. It
    // often does not: 6-8 bit and FP16 are level on almost every model here.
    if (same && BANDS.indexOf(r.band) > BANDS.indexOf(same.band) && r.score > same.score) {
      return `run ${r.model.name} at ${BAND_LABELS[r.band]} instead of ${BAND_LABELS[same.band]}.`;
    }
  }
  const bestNow = now[0]?.score ?? 0;
  const better = raised.find((r) => r.score > bestNow && !now.some((x) => x.model.id === r.model.id));
  if (better) return `put ${better.model.name} within reach, which scores higher than anything that fits now.`;
  return undefined;
}

/**
 * Recommendations for a described machine and workload. When nothing runs the
 * workload at the context it asks for, we say so and retry shorter rather than
 * returning an empty page — shortening context is the fix, and most readers
 * don't know it is a dial.
 */
export function pickModels(
  machine: Machine,
  taskId: string,
  data: PickerData,
  /** Overrides the workload's assumed context when the reader sets one. */
  contextOverride?: number,
): PickResult {
  const task = data.tasks.find((t) => t.id === taskId);
  const budget = budgetFor(machine, data.hardware);
  if (!task) return { recommendations: [], contextTokens: 0, budget, offloadOptions: [] };

  const asked = contextOverride ?? task.assumedContextTokens;
  const offloadOptions: Recommendation[] = [];
  let recommendations = collect(machine, task, data, asked, offloadOptions);
  let contextTokens = asked;
  let fallbackFrom: number | undefined;

  if (recommendations.length === 0) {
    for (const ctx of FALLBACK_CONTEXTS.filter((c) => c < asked)) {
      offloadOptions.length = 0;
      const retry = collect(machine, task, data, ctx, offloadOptions);
      if (retry.length > 0) {
        recommendations = retry;
        contextTokens = ctx;
        fallbackFrom = asked;
        break;
      }
    }
  }

  // Only advise raising the ceiling when it would change what we are offering.
  // "Some model somewhere is blocked" is almost always true on unified memory and
  // reads as though the recommendation on screen needs the change, which it doesn't.
  let ceilingAdvice: HardwareAdvice | undefined;
  let ceilingGain: string | undefined;
  // The same figure the dropdown offers and the command substitutes. This used
  // the OS reserve instead, so one screen printed three different numbers: the
  // prose said "about 26GB", the select offered 30GB, and the command said
  // 30720. Route everything through maxWiredGb so they cannot disagree.
  const headroom = maxWiredGb(machine, data.hardware) ?? Math.max(0, machine.ramGb - budget.reserveGb);
  if (budget.platform.advice && budget.ceilingBinds && headroom > budget.gb) {
    const raised = collect({ ...machine, ceilingOverrideGb: headroom }, task, data, contextTokens);
    const gain = firstImprovement(recommendations, raised);
    if (gain) {
      ceilingAdvice = budget.platform.advice;
      ceilingGain = `Raising it to about ${Math.round(headroom)}GB would ${gain}`;
    }
  }

  // Only when the lead pick is weak, and only if something actually beats it:
  // four times in five there is one, and naming it is more use than leaving the
  // reader to spot it in the list below.
  const lead = recommendations[0];
  const betterThanWeak = lead?.weakBest
    ? offloadOptions.filter((o) => o.score > lead.score).sort((a, b) => b.score - a.score)[0]
    : undefined;

  return {
    recommendations,
    contextTokens,
    fallbackFrom,
    budget,
    offloadOptions: offloadOptions.slice(0, 3),
    betterThanWeak,
    ceilingAdvice,
    ceilingGain,
  };
}

/** One side of a machine comparison. */
export interface ComparisonSide {
  machine: Machine;
  budgetGb: number;
  /** Where the budget came from, e.g. "75% of 32GB, the macOS GPU cap". */
  budgetSource: string;
  /** Ceiling reachable by spilling into system RAM. Equal to budgetGb when there is nowhere to spill. */
  offloadGb: number;
  bandwidthGbs?: number;
  /** How fast this machine's CPU reads system RAM — the speed of any offloaded layer. */
  systemBandwidthGbs?: number;
  prefill?: string;
  /** Top recommendation for the task being compared, if anything runs at all. */
  top?: { model: Model; band: Band; bandLabel: string; score: number; speed: Speed | null };
}

export interface MachineComparison {
  a: ComparisonSide;
  b: ComparisonSide;
  /** Workloads where both machines lead with the same model AND band. */
  agreeOn: number;
  /** Workloads where both machines recommend anything at all. */
  comparableTasks: number;
  /** True when the two lead with the same model on the task being compared. */
  sameModel: boolean;
  /** True when they also agree on the band. */
  sameBand: boolean;
  /**
   * b's speed divided by a's on the compared task — only when both run the SAME
   * model at the same band. A ratio between two different models is a number
   * about two different workloads: an 8GB laptop running a 3B model posts a
   * similar tok/s to a 3090 running a 27B, and 1.06x would read as parity.
   */
  speedRatio?: number;
}

function side(machine: Machine, task: Task, data: PickerData, ctx?: number): ComparisonSide {
  const budget = budgetFor(machine, data.hardware);
  const out = pickModels(machine, task.id, data, ctx);
  const rec = out.recommendations[0];
  const bandwidthGbs = bandwidthFor(machine, data.hardware);
  const systemBandwidthGbs = systemBandwidthFor(machine, data.hardware);
  return {
    machine,
    budgetGb: budget.gb,
    budgetSource: budgetSource(machine, budget),
    offloadGb: budget.offloadGb,
    bandwidthGbs,
    systemBandwidthGbs,
    prefill: prefillFor(machine, data.hardware),
    top: rec && {
      model: rec.model,
      band: rec.band,
      bandLabel: rec.bandLabel,
      score: rec.score,
      speed: generationSpeed(
        rec.fit.weightsGb,
        rec.fit.kvGb,
        rec.model,
        rec.fit,
        bandwidthGbs,
        systemBandwidthGbs,
        data.hardware,
        machine.platform,
      ),
    },
  };
}

/**
 * Two machines, one workload, and what actually differs.
 *
 * The shape of the answer changes with the pair, which is why this returns
 * structure rather than a sentence. An M1 Max 32GB and an RTX 3090 hold 24GB and
 * 23GB, so they lead with the SAME model on every workload and the only real
 * difference is that one reads memory 2.3x faster. An 8GB laptop against the same
 * 3090 agrees on nothing. A page that always printed a table of models would bury
 * the answer in the first case and a page that always printed a speed ratio would
 * miss it in the second.
 *
 * `agreeOn` is counted across every workload, not just the one on screen, because
 * "same pick on 7 of 7" is the sentence that saves a reader clicking through all
 * of them. pickModels is pure and uncached, so calling it 14 times is free.
 */
export function compareMachines(
  a: Machine,
  b: Machine,
  taskId: string,
  data: PickerData,
  contextOverride?: number,
): MachineComparison | null {
  const task = data.tasks.find((t) => t.id === taskId);
  if (!task) return null;

  const sideA = side(a, task, data, contextOverride);
  const sideB = side(b, task, data, contextOverride);

  let agreeOn = 0;
  let comparableTasks = 0;
  for (const t of data.tasks) {
    const ra = pickModels(a, t.id, data, contextOverride).recommendations[0];
    const rb = pickModels(b, t.id, data, contextOverride).recommendations[0];
    if (!ra || !rb) continue;
    comparableTasks++;
    if (ra.model.id === rb.model.id && ra.band === rb.band) agreeOn++;
  }

  const sameModel = !!sideA.top && !!sideB.top && sideA.top.model.id === sideB.top.model.id;
  const sameBand = sameModel && sideA.top!.band === sideB.top!.band;
  const speedRatio =
    sameBand && sideA.top?.speed && sideB.top?.speed
      ? Math.round((sideB.top.speed.hi / sideA.top.speed.hi) * 100) / 100
      : undefined;

  return { a: sideA, b: sideB, agreeOn, comparableTasks, sameModel, sameBand, speedRatio };
}
