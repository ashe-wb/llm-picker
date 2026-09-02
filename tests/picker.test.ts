import { describe, expect, it } from 'vitest';
import { compareMachines, pickModels, scoreFor, NEUTRAL_SCORE, type PickerData } from '../src/lib/picker';
import {
  bandwidthFor,
  chipPresets,
  maxWiredGb,
  raiseSteps,
  ramOptionsFor,
  bestFit,
  budgetFor,
  budgetSource,
  generationSpeed,
  prefillFor,
  sameCapacityPresets,
  overheadFor,
  kqBytesPerToken,
  flashAttentionOffCostGb,
  systemBandwidthFor,
  hasSeparateSystemMemory,
  publishedCeilingGb,
  layerSplit,
  fitFor,
  formatMemory,
  kvGbFrom,
  maxContextFrom,
} from '../src/lib/fit';
import { BANDS } from '../src/lib/schemas';
import type { Band, Hardware, Machine, Model, ScoresFile } from '../src/lib/schemas';
import { loadSiteData } from '../src/lib/data';
import { readFileSync } from 'node:fs';

/* ------------------------------- fixtures -------------------------------- */

/**
 * The real hardware file, not a copy of it. A hand-maintained duplicate drifted
 * three times — missing offloadMinResidentFraction, then `group`, then
 * runtimeOverhead — and each time failed as "Cannot read properties of
 * undefined" a long way from the cause. The tests read platforms, thresholds,
 * runtimeOverhead, the chip table and a handful of preset ids — all values the
 * fixture had been copying by hand anyway, and the chip figures could not be
 * duplicated honestly at all.
 *
 * The coupling is deliberate: the 0.75 macOS ceiling, the 0.2 OS reserve and
 * the 1.0GB + 6% runtime overhead are what these tests encode, so a change to
 * any of them SHOULD fail here rather than pass quietly.
 */
const hardware: Hardware = loadSiteData().hardware;

const mac8: Machine = { platform: 'unified', ramGb: 8 };
const mac32: Machine = { platform: 'unified', ramGb: 32 };

/** ~1KB/token: KV stays under 0.05GB at test contexts, so weights dominate. */
const TINY_KV = 1024;

function model(id: string, opts: Partial<Model> & Pick<Model, 'bands'>): Model {
  return {
    id, name: id, family: 'f', vendor: 'v', architecture: 'dense', paramsB: 8,
    vision: false, contextLength: 262144, license: { name: 'MIT' },
    releaseDate: '2026-01-01', status: 'active', summary: 's',
    kvBytesPerToken: TINY_KV,
    // The runtime overhead is derived from these, and fitFor fails closed
    // without them. Small round values so the fixtures' arithmetic stays
    // readable: 128K vocab and 8192 width put the derived scratch near 0.3GB.
    vocabSize: 131072,
    width: 8192,
    attentionHeads: 32,
    // Speed now charges a per-layer dispatch cost, so a fixture without a layer
    // count gets no tok/s at all — the same fail-closed rule real models meet
    // via validate.ts. 32 is a 7-8B's layer count, matching TINY_KV's scale.
    layers: 32,
    ...opts,
  };
}

const small = model('small', {
  bands: {
    low: { weightsGb: 4, exampleQuants: ['Q3_K_M'] },
    mid: { weightsGb: 5, exampleQuants: ['Q4_K_M'] },
    high: { weightsGb: 7, exampleQuants: ['Q6_K'] },
  },
});
const big = model('big', {
  paramsB: 30,
  bands: { mid: { weightsGb: 19, exampleQuants: ['Q4_K_M'] } },
});
const seeing = model('seeing', {
  vision: true,
  bands: { mid: { weightsGb: 5, exampleQuants: ['Q4_K_M'] } },
});

function scores(modelId: string, entries: ScoresFile['entries']): ScoresFile {
  return { modelId, entries, quantOverrides: [] };
}

const baseData: PickerData = {
  hardware,
  tasks: [
    { id: 'coding', label: 'c', assumedContextTokens: 32768, weights: { coding: 0.55, 'instruction-following': 0.25, 'tool-calling': 0.2 } },
    { id: 'vision', label: 'v', assumedContextTokens: 8192, requires: { vision: true }, weights: { 'instruction-following': 0.5, factuality: 0.5 } },
  ],
  mitigations: [
    { id: 'code-interpreter', label: 'CI', newbieExplainer: 'x', howToLinks: [], worksVia: 'tool-calling' },
    { id: 'none', label: 'No fix', newbieExplainer: 'x', howToLinks: [] },
  ],
  models: [small, big, seeing],
  scores: {
    small: scores('small', [
      { band: 'high', dimension: 'coding', score: 7, provenance: [] as never },
      { band: 'high', dimension: 'instruction-following', score: 8, provenance: [] as never },
      { band: 'high', dimension: 'tool-calling', score: 7, provenance: [] as never },
      { band: 'mid', dimension: 'coding', score: 6, provenance: [] as never },
      { band: 'mid', dimension: 'instruction-following', score: 7, provenance: [] as never },
      { band: 'mid', dimension: 'tool-calling', score: 6, provenance: [] as never },
      { band: 'low', dimension: 'coding', score: 4, provenance: [] as never },
    ]),
    big: scores('big', [
      { band: 'mid', dimension: 'coding', score: 8, provenance: [] as never },
      { band: 'mid', dimension: 'instruction-following', score: 8, provenance: [] as never },
      { band: 'mid', dimension: 'tool-calling', score: 8, provenance: [] as never },
    ]),
    seeing: scores('seeing', [
      { band: 'mid', dimension: 'instruction-following', score: 6, provenance: [] as never },
      { band: 'mid', dimension: 'factuality', score: 6, provenance: [] as never },
    ]),
  },
  weaknesses: {},
};

const pick = (m: Machine, task: string, d: PickerData = baseData) =>
  pickModels(m, task, d).recommendations;

/* --------------------------------- budget --------------------------------- */

describe('budgetFor', () => {
  it('uses the GPU ceiling when it binds before the OS reserve', () => {
    // 0.75 * 32 = 24 vs 32 - max(2, 6.4) = 25.6 → the ceiling wins.
    const b = budgetFor(mac32, hardware);
    expect(b.gb).toBe(24);
    expect(b.ceilingBinds).toBe(true);
  });

  it('uses the OS reserve once the ceiling is raised past it', () => {
    const b = budgetFor({ ...mac32, ceilingOverrideGb: 30 }, hardware);
    expect(b.gb).toBe(25.6); // 32 - 20% reserve, not the 30GB ceiling
    expect(b.ceilingBinds).toBe(false);
  });

  it('keeps the reserve proportional so small machines are not wiped out', () => {
    // A flat 6GB reserve would leave an 8GB machine with 2GB. 20% leaves 6.
    expect(budgetFor(mac8, hardware).gb).toBe(6);
  });

  it('does not claim a ceiling binds on a platform that has none', () => {
    // DGX Spark-style: one coherent pool, no GPU allocation cap. Reporting
    // "capped by ..." there would be misleading copy on the card.
    const b = budgetFor({ platform: 'cpu', ramGb: 128 }, hardware);
    expect(b.ceilingBinds).toBe(false);
    expect(b.gb).toBe(102.4); // 128 - 20% reserve, nothing else applied
  });

  it('ignores system RAM for the discrete-GPU fast path', () => {
    const b = budgetFor({ platform: 'discrete', ramGb: 64, vramGb: 12 }, hardware);
    expect(b.gb).toBe(11); // VRAM - 1GB display overhead
    expect(b.offloadGb).toBe(62.2); // 11 + (64 - 12.8) for spillover
  });
});

/* ----------------------------------- KV ----------------------------------- */

describe('budgetSource', () => {
  const source = (m: Machine) => budgetSource(m, budgetFor(m, hardware));

  it('names the fraction AND the cap that imposed it', () => {
    // Either half alone leaves a question: "75% of 32GB" does not say who chose
    // 75%, and "the macOS GPU cap" does not say how it reached 24GB.
    expect(source(mac32)).toBe('75% of 32GB, the macOS GPU cap');
  });

  it('shows the subtraction on a discrete card', () => {
    expect(source({ platform: 'discrete', ramGb: 64, vramGb: 24 })).toBe(
      '24GB VRAM − 1GB reserved for the display',
    );
  });

  it('falls back to the OS reserve on a platform with no ceiling', () => {
    expect(source({ platform: 'cpu', ramGb: 64, vramGb: 0 })).toBe('64GB RAM − 12.8GB for the OS');
  });

  it('credits a raised ceiling instead of quoting a fraction it replaced', () => {
    // 0.75 * 32 = 24 by default; raised to 25 it still binds before the 25.6GB
    // reserve, so the number on screen is the user's, not the platform's.
    expect(source({ ...mac32, ceilingOverrideGb: 25 })).toBe('the 25GB limit you set');
  });

  it('stops crediting the raise once the OS reserve binds instead', () => {
    // Raised past the reserve, the ceiling is no longer what limits you, and
    // saying "the 30GB limit you set" next to "OF YOUR 25.6GB" would not add up.
    expect(source({ ...mac32, ceilingOverrideGb: 30 })).toBe('32GB RAM − 6.4GB for the OS');
  });

  it('always explains the number the card actually prints', () => {
    // The parenthetical is an explanation, not decoration: whatever arithmetic
    // it states has to land on the budget shown beside it.
    const machines: Machine[] = [
      mac32,
      { ...mac32, ceilingOverrideGb: 25 },
      { ...mac32, ceilingOverrideGb: 30 },
      { platform: 'discrete', ramGb: 64, vramGb: 24 },
      { platform: 'cpu', ramGb: 64, vramGb: 0 },
    ];
    for (const m of machines) {
      const b = budgetFor(m, hardware);
      const text = source(m);
      expect(text, JSON.stringify(m)).not.toContain('undefined');
      const pct = text.match(/^(\d+)% of ([\d.]+)GB/);
      const sub = text.match(/^([\d.]+)GB .*? − ([\d.]+)GB/);
      const flat = text.match(/^the ([\d.]+)GB limit/);
      const derived = pct
        ? (Number(pct[1]) / 100) * Number(pct[2])
        : sub
          ? Number(sub[1]) - Number(sub[2])
          : Number(flat![1]);
      expect(Math.round(derived * 10) / 10, text).toBe(b.gb);
    }
  });
});

describe('formatMemory', () => {
  it('shows sub-gigabyte values in MB instead of rounding them to 0GB', () => {
    // Nemotron 3 Nano caches 6 of 52 layers: 6144 B/token = 50MB at 8K, which
    // used to render as "0GB KV" and read as missing data.
    expect(formatMemory(kvGbFrom(6144, undefined, undefined, 8192))).toBe('48MB');
    expect(formatMemory(kvGbFrom(6144, undefined, undefined, 32768))).toBe('192MB');
  });

  it('uses GB with one decimal at and above a gigabyte', () => {
    expect(formatMemory(2)).toBe('2GB');
    expect(formatMemory(33.44)).toBe('33.4GB');
  });

  it('never rounds a real value down to nothing', () => {
    expect(formatMemory(0.0001)).toBe('1MB');
    expect(formatMemory(0)).toBe('0GB');
  });
});

describe('kvGbFrom', () => {
  it('grows linearly with context for full-attention layers', () => {
    expect(kvGbFrom(65536, undefined, undefined, 32768)).toBeCloseTo(2, 5);
    expect(kvGbFrom(65536, undefined, undefined, 131072)).toBeCloseTo(8, 5);
  });

  it('stops growing past the window for sliding-window layers', () => {
    const at32k = kvGbFrom(0.0001, 819200, 1024, 32768);
    const at128k = kvGbFrom(0.0001, 819200, 1024, 131072);
    expect(at128k - at32k).toBeLessThan(0.02); // windowed term is capped
  });
});

/* ---------------------------------- fit ----------------------------------- */

describe('maxContextFrom — the inverse of kvGbFrom', () => {
  it('returns the largest context the memory holds, and not one block more', () => {
    const bpt = 65536; // Qwen3.8-27B: 64 KiB per token
    const t = maxContextFrom(24, bpt, undefined, undefined);
    expect(kvGbFrom(bpt, undefined, undefined, t)).toBeLessThanOrEqual(24);
    expect(kvGbFrom(bpt, undefined, undefined, t + 1024)).toBeGreaterThan(24);
  });

  it('handles the sliding-window bend, where cost per token drops past the window', () => {
    // Gemma-shaped: a small global term plus a large term capped at 1024 tokens.
    const global = 163840, windowed = 819200, win = 1024;
    const t = maxContextFrom(20, global, windowed, win);
    expect(t).toBeGreaterThan(win); // past the bend
    expect(kvGbFrom(global, windowed, win, t)).toBeLessThanOrEqual(20);
    expect(kvGbFrom(global, windowed, win, t + 1024)).toBeGreaterThan(20);
  });

  it('stays inside the window when memory is too tight to pass it', () => {
    const t = maxContextFrom(0.5, 163840, 819200, 1024);
    expect(t).toBeLessThanOrEqual(1024);
    expect(kvGbFrom(163840, 819200, 1024, t)).toBeLessThanOrEqual(0.5);
  });

  it('returns 0 rather than a negative context when the weights already overflow', () => {
    expect(maxContextFrom(-5, 65536, undefined, undefined)).toBe(0);
  });
});

describe('fitFor', () => {
  const heavy = model('heavy', {
    kvBytesPerToken: 65536, // 2GB at 32K
    bands: { high: { weightsGb: 20.5, exampleQuants: ['Q6_K'] } },
  });

  it('REGRESSION: weights + KV + runtime overflow a default 32GB Mac', () => {
    // The bug that started this: 25GB read as fitting a "26GB" tier. Since the
    // overhead term was re-derived, the same case is 20.5 weights + 2 KV + 1.8
    // runtime — the overhead no longer tracks weights, so a heavier band no
    // longer inflates it.
    const f = fitFor(heavy, 'high', mac32, hardware, 32768)!;
    expect(f.demandGb).toBe(24.3);
    expect(f.overheadGb).toBe(1.8);
    expect(f.budgetGb).toBe(24);
    expect(f.state).toBe('no');
  });

  it('becomes tight — not comfortable — once the wired limit is raised', () => {
    const f = fitFor(heavy, 'high', { ...mac32, ceilingOverrideGb: 30 }, hardware, 32768)!;
    expect(f.state).toBe('tight');
    expect(f.headroomGb).toBeCloseTo(1.3, 5);
  });

  it('flags a ceiling-only block as fixable', () => {
    expect(fitFor(heavy, 'high', mac32, hardware, 32768)!.ceilingBlocked).toBe(true);
  });

  it('REGRESSION: prefers a band that fits in VRAM over a bigger one that offloads', () => {
    // An 11GB card was being offered a 31GB model at Q6-Q8 because it
    // technically "runs" by streaming most of its weights over PCIe.
    const gpu: Machine = { platform: 'discrete', ramGb: 32, vramGb: 12 };
    const wide = model('wide', {
      bands: {
        mid: { weightsGb: 6, exampleQuants: ['Q4_K_M'] },   // fits 11GB VRAM
        high: { weightsGb: 28, exampleQuants: ['Q6_K'] },   // only via offload
      },
    });
    const best = bestFit(wide, gpu, hardware, 8192)!;
    expect(best.band).toBe('mid');
    expect(best.fit.state).toBe('comfortable');
    expect(best.fit.offload).toBe(false);
  });

  it('still returns a tight band when nothing is comfortable', () => {
    const gpu: Machine = { platform: 'discrete', ramGb: 32, vramGb: 12 };
    const chunky = model('chunky', {
      bands: { mid: { weightsGb: 28, exampleQuants: ['Q4_K_M'] } },
    });
    const best = bestFit(chunky, gpu, hardware, 8192)!;
    expect(best.band).toBe('mid');
    expect(best.fit.state).toBe('tight');
    expect(best.fit.offload).toBe(true);
  });

  it('REGRESSION: system RAM changes what is offered', () => {
    // With a blanket "drop all offloaders" rule, 32GB and 512GB of RAM produced
    // identical results and the control was inert.
    const roomy = model('roomy', {
      kvBytesPerToken: TINY_KV,
      bands: {
        mid: { weightsGb: 8, exampleQuants: ['Q4_K_M'] },   // fits 11GB VRAM
        high: { weightsGb: 18, exampleQuants: ['Q6_K'] },   // needs ~7GB of spill
      },
    });
    const data: PickerData = {
      ...baseData,
      models: [roomy],
      scores: {
        roomy: scores('roomy', [
          { band: 'mid', dimension: 'coding', score: 6, provenance: [] as never },
          { band: 'high', dimension: 'coding', score: 8, provenance: [] as never },
        ]),
      },
    };
    const gpu = (ramGb: number): Machine => ({ platform: 'discrete', ramGb, vramGb: 12 });

    // Too little RAM to hold the spill: no upgrade on offer.
    expect(pickModels(gpu(8), 'coding', data).offloadOptions).toHaveLength(0);
    // Plenty of RAM: the better band becomes reachable.
    const rich = pickModels(gpu(64), 'coding', data);
    expect(rich.offloadOptions).toHaveLength(1);
    expect(rich.offloadOptions[0]!.band).toBe('high');
    expect(rich.offloadOptions[0]!.fit.offload).toBe(true);
    // ...and it never displaces the thing that runs at full speed.
    expect(rich.recommendations[0]!.band).toBe('mid');
    expect(rich.recommendations.every((r) => !r.fit.offload)).toBe(true);
  });

  it('REGRESSION: still refuses an offload that leaves too little in VRAM', () => {
    // The Arc B580 case: an 11GB card must not be offered a 31GB model, which
    // would sit around a third resident.
    const huge = model('huge', {
      kvBytesPerToken: TINY_KV,
      bands: { mid: { weightsGb: 30, exampleQuants: ['Q4_K_M'] } },
    });
    const out = pickModels({ platform: 'discrete', ramGb: 64, vramGb: 12 }, 'coding', {
      ...baseData,
      models: [small, huge],
      scores: {
        ...baseData.scores,
        huge: scores('huge', [{ band: 'mid', dimension: 'coding', score: 10, provenance: [] as never }]),
      },
    });
    expect(out.recommendations[0]!.model.id).toBe('small');
    expect(out.offloadOptions.map((o) => o.model.id)).not.toContain('huge');
  });

  it('reports what share of a model stays in fast memory', () => {
    const f = fitFor(big, 'mid', { platform: 'discrete', ramGb: 64, vramGb: 12 }, hardware, 8192)!;
    expect(f.offload).toBe(true);
    expect(f.residentFraction).toBeCloseTo(11 / f.demandGb, 2);
    expect(f.spilledGb).toBeCloseTo(f.demandGb - 11, 1);
    // Resident fits are always whole.
    expect(fitFor(small, 'mid', mac32, hardware, 8192)!.residentFraction).toBe(1);
  });

  it('calls partial offload tight regardless of headroom', () => {
    const f = fitFor(big, 'mid', { platform: 'discrete', ramGb: 64, vramGb: 12 }, hardware, 32768)!;
    expect(f.offload).toBe(true);
    expect(f.state).toBe('tight');
  });

  it('REGRESSION: reports the context spare memory buys, not just the workload figure', () => {
    // 80GB card, Qwen3.8-27B at FP16: 55GB weights + 2GB KV at 32K left 22GB
    // looking like waste. That spare runs the model's entire window.
    const big: Machine = { platform: 'discrete', ramGb: 128, vramGb: 80 };
    const heavy = model('heavy', {
      contextLength: 262144,
      kvBytesPerToken: 65536,
      bands: { full: { weightsGb: 55, exampleQuants: ['bf16'] } },
    });
    const f = fitFor(heavy, 'full', big, hardware, 32768)!;
    expect(f.contextTokens).toBe(32768);
    expect(f.maxContextTokens).toBe(262144); // the whole window
  });

  it('RETRACTED: there is no real-machine check on the overhead term', () => {
    // A ~80K context reading on a 32GB Mac used to stand here as corroboration.
    // It was taken with the wired limit RAISED, so it was never a 24GB-budget
    // measurement and could not test this term — at a raised cap the same
    // reading implies roughly 8GB of overhead, which the derivation cannot
    // produce. Withdrawn rather than reworded, because a check whose conditions
    // are known to be wrong is worse than no check.
    //
    // What survives is the mechanism, pinned by the 507MB Gemma 2 test above.
    // This test exists so the absence is deliberate and visible: if a real
    // measurement is ever taken, replace this, and record the wired limit with
    // it. See docs/measuring-runtime-overhead.md.
    const real = loadSiteData();
    expect(real.hardware.runtimeOverhead.note).toContain('NOT YET CHECKED AGAINST A REAL MACHINE');
  });

  it('counts weights, KV and runtime overhead in demand', () => {
    const f = fitFor(small, 'mid', mac32, hardware, 8192)!;
    expect(f.overheadGb).toBeGreaterThan(0);
    expect(f.demandGb).toBeCloseTo(f.weightsGb + f.kvGb + f.overheadGb, 1);
  });

  it('caps max context at the model window, never beyond it', () => {
    const small_ctx = model('shortwin', {
      contextLength: 8192,
      kvBytesPerToken: 1024,
      bands: { mid: { weightsGb: 2, exampleQuants: ['Q4_K_M'] } },
    });
    expect(fitFor(small_ctx, 'mid', mac32, hardware, 4096)!.maxContextTokens).toBe(8192);
  });

  it('clamps context to what the model actually supports', () => {
    const shortCtx = model('short', {
      contextLength: 8192,
      bands: { mid: { weightsGb: 5, exampleQuants: ['Q4_K_M'] } },
    });
    expect(fitFor(shortCtx, 'mid', mac32, hardware, 131072)!.contextTokens).toBe(8192);
  });

  it('refuses to guess when a model has no measured KV figure', () => {
    const unknown = { ...small, kvBytesPerToken: undefined };
    expect(fitFor(unknown, 'mid', mac32, hardware, 32768)).toBeNull();
  });
});

/* --------------------------------- picker --------------------------------- */

describe('pickModels', () => {
  it('excludes models too big for the machine', () => {
    expect(pick(mac8, 'coding').map((r) => r.model.id)).not.toContain('big');
    expect(pick(mac8, 'coding').map((r) => r.model.id)).toContain('small');
  });

  it('chooses the highest-SCORING band that fits', () => {
    // `small` scores high 7/8/7 against mid 6/7/6, so the bigger band earns it.
    expect(pick(mac32, 'coding').find((r) => r.model.id === 'small')?.band).toBe('high');
  });

  it('REGRESSION: takes the smaller band when a bigger one scores no better', () => {
    // The FP16 trap. 6-8 bit is lossless on 16 of the 17 models scored at both
    // bands, and the picker was still spending the memory and the speed on FP16.
    const flat = model('flat', {
      bands: {
        mid: { weightsGb: 4, exampleQuants: ['Q4_K_M'] },
        high: { weightsGb: 6, exampleQuants: ['Q6_K'] },
        full: { weightsGb: 12, exampleQuants: ['bf16'] },
      },
    });
    const data: PickerData = {
      ...baseData,
      models: [flat],
      scores: {
        flat: scores('flat', [
          { band: 'mid', dimension: 'coding', score: 6, provenance: [] as never },
          { band: 'high', dimension: 'coding', score: 8, provenance: [] as never },
          { band: 'full', dimension: 'coding', score: 8, provenance: [] as never },
        ]),
      },
    };
    const rec = pickModels(mac32, 'coding', data).recommendations[0]!;
    expect(rec.band).toBe('high'); // not 'full' — same score, twice the weights
    expect(rec.skippedBand?.band).toBe('full');
    expect(rec.skippedBand?.scoreDelta).toBe(0);
  });

  it('REGRESSION: never shows a band that scores worse than one that also fits', () => {
    // scoreFor borrows from a neighbour and docks a point per band of distance,
    // so an unmeasured band scores BELOW the one it borrowed from. The picker
    // displayed those anyway — 34 recommendations across the real dataset.
    const unmeasuredTop = model('unmeasured', {
      bands: {
        mid: { weightsGb: 4, exampleQuants: ['Q4_K_M'] },
        high: { weightsGb: 6, exampleQuants: ['Q6_K'] }, // no scores at all
      },
    });
    const data: PickerData = {
      ...baseData,
      models: [unmeasuredTop],
      scores: {
        unmeasured: scores('unmeasured', [
          { band: 'mid', dimension: 'coding', score: 7, provenance: [] as never },
        ]),
      },
    };
    const rec = pickModels(mac32, 'coding', data).recommendations[0]!;
    expect(rec.band).toBe('mid'); // 'high' would borrow 7 and dock to 6
    // measuredScore, not score: only one of the task's three dimensions is
    // rated here, so the coverage discount pulls the headline down.
    expect(rec.measuredScore).toBe(7);
  });

  it('does not offer an offload upgrade that scores no better', () => {
    // Streaming weights over PCIe to reach an equal-scoring band buys nothing.
    const flat = model('flatoff', {
      kvBytesPerToken: TINY_KV,
      bands: {
        mid: { weightsGb: 8, exampleQuants: ['Q4_K_M'] },  // fits 11GB VRAM
        high: { weightsGb: 18, exampleQuants: ['Q6_K'] },  // only by spilling
      },
    });
    const data: PickerData = {
      ...baseData,
      models: [flat],
      scores: {
        flatoff: scores('flatoff', [
          { band: 'mid', dimension: 'coding', score: 7, provenance: [] as never },
          { band: 'high', dimension: 'coding', score: 7, provenance: [] as never },
        ]),
      },
    };
    const out = pickModels({ platform: 'discrete', ramGb: 64, vramGb: 12 }, 'coding', data);
    expect(out.recommendations[0]!.band).toBe('mid');
    expect(out.offloadOptions).toHaveLength(0);
  });

  it('vision task excludes non-vision models', () => {
    // mac32, not mac8: with runtime overhead the vision model no longer fits an
    // 8GB machine, and this test is about the filter, not about memory.
    expect(pick(mac32, 'vision').map((r) => r.model.id)).toEqual(['seeing']);
  });

  it('discounts the score for unrated dimensions and flags thin evidence', () => {
    // One band only, so the assertion cannot be moved by band-selection changes.
    const solo = model('solo', { bands: { mid: { weightsGb: 5, exampleQuants: ['Q4_K_M'] } } });
    const data: PickerData = {
      ...baseData,
      models: [solo],
      scores: { solo: scores('solo', [{ band: 'mid', dimension: 'coding', score: 6, provenance: [] as never }]) },
    };
    const recs = pick(mac32, 'coding', data);
    expect(recs[0]!.incompleteData).toBe(true);
    // Only `coding` (0.55 of the task) is rated, at 6. The measured mean is still
    // 6, but the headline is pulled toward 5 for the 45% nobody measured.
    expect(recs[0]!.measuredScore).toBe(6);
    expect(recs[0]!.coverage).toBeCloseTo(0.55, 5);
    expect(recs[0]!.score).toBe(5.6); // 6*0.55 + 5*0.45
  });

  it('REGRESSION: a model rated on a sliver of the task cannot outrank a well-covered one', () => {
    // The 128GB Mac Studio bug: Qwen3.5 9B won "a bit of everything" on one
    // rated dimension worth 10% of the task, beating a model rated on 70%.
    const sliver = model('sliver', { bands: { mid: { weightsGb: 5, exampleQuants: ['Q4_K_M'] } } });
    const rounded = model('rounded', { bands: { mid: { weightsGb: 5, exampleQuants: ['Q4_K_M'] } } });
    const recs = pick(mac32, 'coding', {
      ...baseData,
      models: [sliver, rounded],
      scores: {
        // one dimension, top marks, 20% of the task's weight
        sliver: scores('sliver', [
          { band: 'mid', dimension: 'tool-calling', score: 10, provenance: [] as never },
        ]),
        // every dimension, lower marks, all of the task's weight
        rounded: scores('rounded', [
          { band: 'mid', dimension: 'coding', score: 7, provenance: [] as never },
          { band: 'mid', dimension: 'instruction-following', score: 7, provenance: [] as never },
          { band: 'mid', dimension: 'tool-calling', score: 7, provenance: [] as never },
        ]),
      },
    });
    expect(recs[0]!.model.id).toBe('rounded');
    expect(recs[0]!.coverage).toBe(1);
    const s = recs.find((r) => r.model.id === 'sliver')!;
    expect(s.measuredScore).toBe(10); // what we measured is unchanged...
    expect(s.score).toBeLessThan(recs[0]!.score); // ...but it does not win on it
  });

  it('downgrades tool-dependent mitigations when tool-calling is weak', () => {
    const data: PickerData = {
      ...baseData,
      models: [small],
      scores: {
        small: scores('small', [
          { band: 'low', dimension: 'coding', score: 3, provenance: [] as never },
          { band: 'low', dimension: 'tool-calling', score: 2, provenance: [] as never },
        ]),
      },
      weaknesses: {
        small: {
          modelId: 'small',
          entries: [{
            id: 'small/low/coding', bands: ['low'], dimensions: ['coding'], severity: 'severe',
            summary: 'bad', citations: [], confidence: 'community-consensus',
            mitigations: [{ mitigationId: 'code-interpreter', note: 'use a tool' }],
          }],
        },
      },
    };
    // 6GB budget: only the low band fits once runtime overhead is counted.
    const recs = pick({ platform: 'unified', ramGb: 8 }, 'coding', data);
    expect(recs[0]!.band).toBe('low');
    expect(recs[0]!.warnings[0]!.chips[0]!.limited).toBe(true);
  });

  it('marks the "none" mitigation as noFix', () => {
    const data: PickerData = {
      ...baseData,
      weaknesses: {
        small: {
          modelId: 'small',
          entries: [{
            id: 'small/high/coding', bands: ['high'], dimensions: ['coding'], severity: 'moderate',
            summary: 'meh', citations: [], confidence: 'community-consensus',
            mitigations: [{ mitigationId: 'none', note: 'pick another model' }],
          }],
        },
      },
    };
    const rec = pick(mac32, 'coding', data).find((r) => r.model.id === 'small')!;
    expect(rec.warnings[0]!.chips[0]!.noFix).toBe(true);
  });

  it('tags the top pick and a lightest good option', () => {
    const recs = pick(mac32, 'coding');
    expect(recs[0]!.tags).toContain('top-pick');
    const lightest = recs.find((r) => r.tags.includes('lightest'));
    expect(lightest!.model.id).toBe('small');
  });

  it('honours a context override, and matches the workload figure without one', () => {
    const auto = pickModels(mac32, 'coding', baseData);
    const same = pickModels(mac32, 'coding', baseData, 32768); // the task's own value
    expect(same.contextTokens).toBe(auto.contextTokens);
    expect(same.recommendations[0]!.model.id).toBe(auto.recommendations[0]!.model.id);

    const long = pickModels(mac32, 'coding', baseData, 131072);
    expect(long.contextTokens).toBe(131072);
  });

  it('leaves qualifiers empty when the top score is not tied', () => {
    const recs = pick(mac32, 'coding');
    expect(recs.filter((r) => r.tags.includes('top-pick'))).toHaveLength(1);
    expect(recs[0]!.qualifiers).toEqual([]);
  });

  it('falls back to a shorter context rather than returning nothing', () => {
    // 8GB of KV per 32K: nothing fits a 32GB Mac at the task's context.
    const hungry = model('hungry', {
      kvBytesPerToken: 262144,
      bands: { mid: { weightsGb: 18, exampleQuants: ['Q4_K_M'] } },
    });
    const out = pickModels(mac32, 'coding', {
      ...baseData,
      models: [hungry],
      scores: { hungry: scores('hungry', [{ band: 'mid', dimension: 'coding', score: 7, provenance: [] as never }]) },
    });
    expect(out.fallbackFrom).toBe(32768);
    expect(out.contextTokens).toBeLessThan(32768);
    expect(out.recommendations).toHaveLength(1);
  });

  it('surfaces ceiling advice when a raise would unblock something better', () => {
    // 24GB budget, 25.6GB reachable: `heavy` needs 24.7 once runtime overhead is
    // counted, so it is out of reach until the ceiling is raised.
    const heavy = model('heavy', {
      kvBytesPerToken: 65536,
      bands: { high: { weightsGb: 20.5, exampleQuants: ['Q6_K'] } },
    });
    const out = pickModels(mac32, 'coding', {
      ...baseData,
      models: [small, heavy],
      scores: {
        ...baseData.scores,
        // Fully rated, so the coverage discount does not sink it below `small`.
        heavy: scores('heavy', [
          { band: 'high', dimension: 'coding', score: 9, provenance: [] as never },
          { band: 'high', dimension: 'instruction-following', score: 9, provenance: [] as never },
          { band: 'high', dimension: 'tool-calling', score: 9, provenance: [] as never },
        ]),
      },
    });
    expect(out.ceilingAdvice?.id).toBe('raise-wired-limit');
    expect(out.ceilingGain).toContain('heavy');
  });

  it('REGRESSION: stays quiet when raising the ceiling would change nothing', () => {
    // The 64GB creative-writing bug: a model far out of reach at ANY ceiling was
    // triggering "raise your limit" next to recommendations that ran fine already.
    const unreachable = model('unreachable', {
      bands: { full: { weightsGb: 400, exampleQuants: ['bf16'] } },
    });
    const out = pickModels(mac32, 'coding', {
      ...baseData,
      models: [small, unreachable],
      scores: {
        ...baseData.scores,
        unreachable: scores('unreachable', [
          { band: 'full', dimension: 'coding', score: 10, provenance: [] as never },
        ]),
      },
    });
    expect(out.recommendations.length).toBeGreaterThan(0);
    expect(out.ceilingAdvice).toBeUndefined();
    expect(out.ceilingGain).toBeUndefined();
  });

  it('never offers ceiling advice on a platform with no adjustable ceiling', () => {
    const out = pickModels({ platform: 'discrete', ramGb: 32, vramGb: 12 }, 'coding', baseData);
    expect(out.ceilingAdvice).toBeUndefined();
  });
});

/* ---------------------------- tie qualifiers ------------------------------ */

describe('pickModels — ties', () => {
  const codingScores: ScoresFile['entries'] = [
    { band: 'mid', dimension: 'coding', score: 8, provenance: [] as never },
    { band: 'mid', dimension: 'instruction-following', score: 8, provenance: [] as never },
    { band: 'mid', dimension: 'tool-calling', score: 8, provenance: [] as never },
  ];
  const sparse = model('sparse', {
    architecture: 'moe', paramsB: 30, activeParamsB: 3, contextLength: 262144,
    bands: { mid: { weightsGb: 17, exampleQuants: ['Q4_K_M'] } },
  });
  const twin = model('twin', {
    paramsB: 30, contextLength: 32768,
    bands: { mid: { weightsGb: 19, exampleQuants: ['Q4_K_M'] } },
  });
  const tieData: PickerData = {
    ...baseData,
    models: [big, sparse],
    scores: { ...baseData.scores, sparse: scores('sparse', codingScores) },
  };

  it('gives both tied models the top-pick badge', () => {
    const picks = pick(mac32, 'coding', tieData).filter((r) => r.tags.includes('top-pick'));
    expect(picks).toHaveLength(2);
    expect(picks.map((p) => p.model.id).sort()).toEqual(['big', 'sparse']);
  });

  it('qualifies the winner on its strongest axes and the other honestly', () => {
    const recs = pick(mac32, 'coding', tieData);
    const winner = recs.find((r) => r.model.id === 'sparse')!;
    const dominated = recs.find((r) => r.model.id === 'big')!;
    expect(winner.qualifiers[0]).toBe('Faster — 3B active of 30B');
    expect(dominated.qualifiers).toHaveLength(1);
    expect(dominated.qualifiers[0]).toContain('No measured edge here');
    expect(dominated.qualifiers[0]).toContain('sparse runs faster');
  });

  it('does not claim a speed edge without a real gap', () => {
    const near = model('near', {
      paramsB: 28,
      bands: { mid: { weightsGb: 19, exampleQuants: ['Q4_K_M'] } },
    });
    const recs = pick(mac32, 'coding', {
      ...baseData,
      models: [big, near],
      scores: { ...baseData.scores, near: scores('near', codingScores) },
    });
    expect(recs.find((r) => r.model.id === 'near')!.qualifiers.join(' ')).not.toContain('Faster');
  });

  it('caps the badge at two and explains the rest via tieNote', () => {
    const recs = pick(mac32, 'coding', {
      ...baseData,
      models: [big, sparse, twin],
      scores: {
        ...baseData.scores,
        sparse: scores('sparse', codingScores),
        twin: scores('twin', codingScores),
      },
    });
    expect(recs.filter((r) => r.tags.includes('top-pick'))).toHaveLength(2);
    expect(recs.find((r) => r.tags.includes('top-pick'))!.tieNote).toContain('1 other model');
  });
});

/* --------------------------------- scoreFor -------------------------------- */

describe('generationSpeed', () => {
  const g4090: Machine = { platform: 'discrete', ramGb: 64, vramGb: 24 };
  // A dense 8B at Q4: 5GB of weights, the shape every published tok/s figure uses.
  // Real Llama-3.1-8B KV: 4GB at 32K works out to 131072 bytes/token.
  const dense8b = model('d8', {
    paramsB: 8,
    kvBytesPerToken: 131072,
    bands: { mid: { weightsGb: 5, exampleQuants: ['Q4_K_M'] } },
  });

  const speedOf = (m: Model, band: Band, machine: Machine, ctx: number, bw: number, sys = 90) => {
    const f = fitFor(m, band, machine, hardware, ctx)!;
    return generationSpeed(f.weightsGb, f.kvGb, m, f, bw, sys, hardware, machine.platform)!;
  };

  it('reproduces the published RTX 4090 figure', () => {
    // ~85-125 tok/s is the measured range for an 8B at Q4 on a 1008GB/s card at
    // short context. If this drifts, the efficiency band needs recalibrating,
    // not the test moving.
    const s = speedOf(dense8b, 'mid', g4090, 2048, 1008);
    expect(s.lo).toBeGreaterThan(70);
    expect(s.lo).toBeLessThan(100);
    expect(s.hi).toBeGreaterThan(110);
    expect(s.hi).toBeLessThan(150);
  });

  it('scales with bandwidth, but sub-proportionally', () => {
    // 3.5x the bandwidth does NOT buy 3.5x the tokens, because a fixed per-token
    // cost is added to both and does not shrink. That gap is the whole reason
    // the old bandwidth-fraction model overpredicted small reads.
    const fast = speedOf(dense8b, 'mid', g4090, 2048, 1008); // RTX 4090
    const slow = speedOf(dense8b, 'mid', g4090, 2048, 288); // Tesla M40, same 24GB
    const speedRatio = fast.hi / slow.hi;
    expect(speedRatio).toBeGreaterThan(2);
    expect(speedRatio).toBeLessThan(1008 / 288);
  });

  it('counts the KV cache, so speed falls as context grows', () => {
    // The term generic tok/s calculators drop. An 8B at 128K reads more KV than
    // weights, which is why long-context generation crawls.
    const short = speedOf(dense8b, 'mid', g4090, 2048, 1008);
    const long = speedOf(dense8b, 'mid', g4090, 131072, 1008);
    expect(long.hi).toBeLessThan(short.hi / 2);
    expect(long.bytesReadGb).toBeGreaterThan(short.bytesReadGb);
  });

  it('reads only the active experts of an MoE', () => {
    // Same resident weights, same machine: the sparse one should be far quicker
    // because it touches a tenth of itself per token.
    const bands = { mid: { weightsGb: 20, exampleQuants: ['Q4_K_M' as const] } };
    const heavyDense = model('d', { paramsB: 30, architecture: 'dense', bands });
    const sparse = model('s', { paramsB: 30, activeParamsB: 3, architecture: 'moe', bands });
    const big: Machine = { platform: 'discrete', ramGb: 128, vramGb: 48 };
    const d = speedOf(heavyDense, 'mid', big, 2048, 1000);
    const m = speedOf(sparse, 'mid', big, 2048, 1000);
    expect(m.hi).toBeGreaterThan(d.hi * 3);
    expect(m.bytesReadGb).toBeLessThan(d.bytesReadGb);
  });

  it('slows sharply when the CPU has to run part of the model', () => {
    // The "several times slower" the site warns about, given a number. Still a
    // cliff — just not the wrong cliff.
    const bands = { mid: { weightsGb: 20, exampleQuants: ['Q4_K_M' as const] } };
    const m = model('sp', { paramsB: 30, bands });
    const resident: Machine = { platform: 'discrete', ramGb: 64, vramGb: 32 };
    const spilling: Machine = { platform: 'discrete', ramGb: 64, vramGb: 16 };
    const fast = speedOf(m, 'mid', resident, 2048, 1000);
    const slow = speedOf(m, 'mid', spilling, 2048, 1000);
    expect(fast.throttledByOffload).toBe(false);
    expect(slow.throttledByOffload).toBe(true);
    expect(slow.hi).toBeLessThan(fast.hi / 2);
  });

  it('REGRESSION: a graphics card never makes a machine slower than no card', () => {
    // The bug this replaced priced spilled layers at PCIe 32GB/s, below the
    // 90GB/s the same data file gives a machine with no GPU at all — so the
    // site reported an RTX 3090 as slower than a bare CPU below 67% resident.
    // Adding a card cannot subtract speed. Asserted across the whole range,
    // because the old model passed at the top of it and failed at the bottom.
    const bands = { mid: { weightsGb: 40, exampleQuants: ['Q4_K_M' as const] } };
    const m = model('big', { paramsB: 70, bands });
    const sys = 90;
    for (const vramGb of [8, 12, 16, 20, 24, 32]) {
      const machine: Machine = { platform: 'discrete', ramGb: 128, vramGb };
      const s = speedOf(m, 'mid', machine, 2048, 936, sys);
      const cpuOnly = speedOf(m, 'mid', { platform: 'cpu', ramGb: 128 }, 2048, sys, sys);
      expect(s.hi).toBeGreaterThanOrEqual(cpuOnly.hi);
      expect(s.hi).toBeLessThanOrEqual(speedOf(m, 'mid', { platform: 'discrete', ramGb: 128, vramGb: 80 }, 2048, 936, sys).hi);
    }
  });

  it('matches a measured -ngl sweep to within 10%', () => {
    // Pins the model to evidence rather than to arithmetic. Published sweep on
    // a 14B: 2.89 tok/s with everything on the CPU, 43.18 with everything on
    // the card, 12.5 measured at 40 of 48 layers offloaded to the GPU.
    // https://inventivehq.com/blog/vram-offload-cliff-gpu-layers-benchmark
    const cpuOnly = 2.89;
    const gpuOnly = 43.18;
    const measured = 12.5;
    const r = 40 / 48;
    // Same harmonic mean generationSpeed applies, expressed in tok/s directly:
    // the card reads its share at full speed, the CPU reads the rest at its own.
    const predicted = 1 / (r / gpuOnly + (1 - r) / cpuOnly);
    expect(Math.abs(predicted - measured) / measured).toBeLessThan(0.1);
  });

  it('returns null rather than guessing a bandwidth', () => {
    const f = fitFor(dense8b, 'mid', g4090, hardware, 2048)!;
    expect(generationSpeed(f.weightsGb, f.kvGb, dense8b, f, undefined, 90, hardware, 'discrete')).toBeNull();
    expect(generationSpeed(f.weightsGb, f.kvGb, dense8b, f, 0, 90, hardware, 'discrete')).toBeNull();
  });

  it('returns null rather than guessing how fast the CPU half is', () => {
    // A partial fit whose slow side has no speed on file has no honest tok/s.
    const bands = { mid: { weightsGb: 20, exampleQuants: ['Q4_K_M' as const] } };
    const m = model('sp2', { paramsB: 30, bands });
    const spilling: Machine = { platform: 'discrete', ramGb: 64, vramGb: 16 };
    const f = fitFor(m, 'mid', spilling, hardware, 2048)!;
    expect(f.offload).toBe(true);
    expect(generationSpeed(f.weightsGb, f.kvGb, m, f, 1000, undefined, hardware, 'discrete')).toBeNull();
    // A fit that is fully resident does not need one.
    const res = fitFor(m, 'mid', { platform: 'discrete', ramGb: 64, vramGb: 32 }, hardware, 2048)!;
    expect(generationSpeed(res.weightsGb, res.kvGb, m, res, 1000, undefined, hardware, 'discrete')).not.toBeNull();
  });
});

describe('compareMachines', () => {
  const real: PickerData = {
    hardware,
    tasks: loadSiteData().tasks,
    mitigations: loadSiteData().mitigations,
    models: loadSiteData().models,
    scores: Object.fromEntries(loadSiteData().scores),
    weaknesses: Object.fromEntries(loadSiteData().weaknesses),
  };
  const m1max: Machine = { platform: 'unified', ramGb: 32, chipId: 'm1-max' };
  const rtx3090: Machine = { platform: 'discrete', ramGb: 64, vramGb: 24, presetId: 'rtx-3090' };
  const laptop: Machine = { platform: 'discrete', ramGb: 32, vramGb: 8, presetId: 'rtx-4070-8' };
  const m5max: Machine = { platform: 'unified', ramGb: 128, chipId: 'm5-max-40c' };
  const spark: Machine = { platform: 'unified-nvidia', ramGb: 128, presetId: 'dgx-spark' };

  it('finds two near-matched machines agree on everything', () => {
    // 24GB against 23GB of budget: they hold the same models, so the honest
    // comparison is speed, not a list. This is the case the block is shaped for.
    const c = compareMachines(m1max, rtx3090, 'coding-assistant', real)!;
    expect(c.agreeOn).toBe(c.comparableTasks);
    expect(c.comparableTasks).toBe(7);
    expect(c.sameModel).toBe(true);
    expect(c.sameBand).toBe(true);
  });

  it('reports a speed ratio that tracks the bandwidth ratio', () => {
    // Same model, same band, so bytes-read cancels and the ratio should sit
    // near the bandwidth ratio: 936 / 400 = 2.34. Not exactly on it — the
    // per-layer overhead is a fixed cost, and its share of the token differs
    // between CUDA and Metal — but within a fifth of it.
    const c = compareMachines(m1max, rtx3090, 'coding-assistant', real)!;
    expect(c.speedRatio).toBeGreaterThan(1.9);
    expect(c.speedRatio).toBeLessThan(2.6);
  });

  it('withholds the ratio when the two run different models', () => {
    // An 8GB laptop on a 3B model posts a similar tok/s to a 3090 on a 27B.
    // Reporting 1.06x there would read as parity between the machines.
    const c = compareMachines(laptop, rtx3090, 'coding-assistant', real)!;
    expect(c.agreeOn).toBe(0);
    expect(c.sameModel).toBe(false);
    expect(c.speedRatio).toBeUndefined();
  });

  it('REGRESSION: same capacity can still mean a different band', () => {
    // Both hold 128GB, but the Spark's ram-only ceiling leaves 115GB resident
    // against the Mac's 96GB after the macOS cap. The result most likely to be
    // mistaken for a bug.
    const c = compareMachines(m5max, spark, 'homework', real)!;
    expect(c.a.budgetGb).toBe(96);
    expect(c.b.budgetGb).toBeGreaterThan(c.a.budgetGb);
    expect(c.a.bandwidthGbs).toBe(614);
    expect(c.b.bandwidthGbs).toBe(273);
  });

  it('carries each side offload reach, which is where unified memory loses', () => {
    const c = compareMachines(m1max, rtx3090, 'coding-assistant', real)!;
    expect(c.a.offloadGb).toBe(c.a.budgetGb); // unified: nowhere to spill
    expect(c.b.offloadGb).toBeGreaterThan(c.b.budgetGb); // discrete: into system RAM
  });

  it('is a pure read — neither machine is mutated', () => {
    const a = { ...m1max };
    const b = { ...rtx3090 };
    compareMachines(a, b, 'coding-assistant', real);
    expect(a).toEqual(m1max);
    expect(b).toEqual(rtx3090);
  });

  it('returns null for an unknown task rather than throwing', () => {
    expect(compareMachines(m1max, rtx3090, 'not-a-task', real)).toBeNull();
  });
});

describe('maxWiredGb / raiseSteps', () => {
  const mac = (ramGb: number): Machine => ({ platform: 'unified', ramGb });

  it('offers only what the budget will actually spend', () => {
    // Was 30GB, from a second sqrt-scaled reserve that budgetFor never honoured
    // — so 26, 28 and 30GB all produced the same 25.6GB budget and the command
    // printed a number the site did not model. One reserve now, and the offer
    // is it: 32 - max(2, 20% of 32) = 25.6, rounded to 26.
    expect(maxWiredGb(mac(32), hardware)).toBe(26);
    // Whatever it offers, selecting it must move the budget. That is the whole
    // defect: a control that lies about its own effect.
    for (const ram of [16, 32, 64, 128]) {
      const max = maxWiredGb(mac(ram), hardware)!;
      const before = budgetFor(mac(ram), hardware).gb;
      const after = budgetFor({ ...mac(ram), ceilingOverrideGb: max }, hardware).gb;
      expect(after, `${ram}GB`).toBeGreaterThan(before);
      expect(after, `${ram}GB`).toBeCloseTo(Math.min(max, ram - budgetFor(mac(ram), hardware).reserveGb), 1);
    }
  });

  it('holds back exactly what the budget holds back, at every capacity', () => {
    // This used to assert a sub-linear reserve: 2GB on a 32GB Mac, 8GB on a
    // 512GB one. That was the raise block's own arithmetic, and it disagreed
    // with the reserve the budget actually applies. Checked across every
    // capacity, the OS reserve was larger at all of them, so the sqrt figure
    // never once bound — it was dead arithmetic making an uncited claim.
    // The invariant that replaces it is agreement.
    for (const ram of hardware.ramOptions) {
      const max = maxWiredGb(mac(ram), hardware);
      if (max === null) continue;
      const b = budgetFor(mac(ram), hardware);
      expect(ram - max, `${ram}GB`).toBeCloseTo(b.reserveGb, 0);
    }
  });

  it('never offers the whole machine', () => {
    // The old ladder-derived list offered exactly one raise on most Macs, and
    // it was 100% of memory — the setting the advice text warns against.
    for (const g of hardware.ramOptions) {
      const steps = raiseSteps(mac(g), hardware);
      expect(steps.every((s) => s < g), `${g}GB`).toBe(true);
      expect(steps.every((s) => s > Math.round(0.75 * g)), `${g}GB`).toBe(true);
    }
  });

  it('gives an 8GB Mac nothing to raise, because it has nothing spare', () => {
    expect(raiseSteps(mac(8), hardware)).toEqual([]);
    expect(maxWiredGb(mac(8), hardware)).toBe(Math.round(0.75 * 8));
  });

  it('offers several stops, ending at the safe maximum', () => {
    const steps = raiseSteps(mac(32), hardware);
    expect(steps.length).toBeGreaterThan(1);
    expect(steps[steps.length - 1]).toBe(maxWiredGb(mac(32), hardware));
    expect([...steps].sort((a, b) => a - b)).toEqual(steps); // ascending
  });

  it('is null where the platform has no adjustable cap', () => {
    expect(maxWiredGb({ platform: 'discrete', ramGb: 64, vramGb: 24 }, hardware)).toBeNull();
    expect(raiseSteps({ platform: 'cpu', ramGb: 64 }, hardware)).toEqual([]);
  });
});

describe('bandwidthFor / prefillFor / sameCapacityPresets', () => {
  it("REGRESSION: a Mac gets its chip bandwidth, not its capacity bandwidth", () => {
    // The bug this replaced: every 32GB Mac was quoted 273 GB/s, the M4 Pro
    // figure. A 32GB Mac is anything from an M4 at 120 to an M5 Max at 614, and
    // an M1 Max reads at 400 — a third faster than it was told.
    const m: Machine = { platform: 'unified', ramGb: 32 };
    expect(bandwidthFor({ ...m, chipId: 'm1-max' }, hardware)).toBe(400);
    expect(bandwidthFor({ ...m, chipId: 'm4' }, hardware)).toBe(120);
    expect(bandwidthFor({ ...m, chipId: 'm5-max-40c' }, hardware)).toBe(614);
  });

  it('holds every chip figure, so none can drift unnoticed', () => {
    // Reprice a Mac by editing one number and this fails, which is the point:
    // two comparison sites I checked disagreed with each other on M3 Pro.
    const expected: Record<string, number> = {
      m1: 68, 'm1-pro': 200, 'm1-max': 400, 'm1-ultra': 800,
      m2: 100, 'm2-pro': 200, 'm2-max': 400, 'm2-ultra': 800,
      m3: 100, 'm3-pro': 150, 'm3-max-30c': 300, 'm3-max-40c': 400, 'm3-ultra': 819,
      m4: 120, 'm4-pro': 273, 'm4-max-32c': 410, 'm4-max-40c': 546,
      m5: 153, 'm5-pro': 307, 'm5-max-32c': 460, 'm5-max-40c': 614, 'm5-ultra': 1228,
      m6: 170,
    };
    for (const [id, bw] of Object.entries(expected)) {
      expect(hardware.chips.find((c) => c.id === id)?.bandwidthGbs, id).toBe(bw);
    }
    expect(hardware.chips).toHaveLength(Object.keys(expected).length);
  });

  it('keeps the M3 Pro below the M1 and M2 Pro', () => {
    // Apple shipped a 25% regression. The figure most likely to be "corrected"
    // by someone assuming generations only ever improve.
    const bw = (id: string) => hardware.chips.find((c) => c.id === id)!.bandwidthGbs;
    expect(bw('m3-pro')).toBe(150);
    expect(bw('m3-pro')).toBeLessThan(bw('m1-pro'));
    expect(bw('m3-pro')).toBeLessThan(bw('m2-pro'));
  });

  it('honours a chip whose bandwidth varies with capacity', () => {
    // The M6 reads at 153 with 16GB fitted and 170 at 24GB or more.
    const m6 = (ramGb: number): Machine => ({ platform: 'unified', ramGb, chipId: 'm6' });
    expect(bandwidthFor(m6(16), hardware)).toBe(153);
    expect(bandwidthFor(m6(24), hardware)).toBe(170);
    expect(bandwidthFor(m6(32), hardware)).toBe(170);
  });

  it('REGRESSION: the M5 Max is two chips, not one', () => {
    // Apple splits it on GPU cores, not CPU cores - both are 18-core CPUs. The
    // table carried only 614, so a 32-core owner was overstated by a third.
    const m: Machine = { platform: 'unified', ramGb: 64 };
    expect(bandwidthFor({ ...m, chipId: 'm5-max-32c' }, hardware)).toBe(460);
    expect(bandwidthFor({ ...m, chipId: 'm5-max-40c' }, hardware)).toBe(614);
  });

  it('offers exactly the capacities a chip shipped with', () => {
    const opts = (chipId: string) => ramOptionsFor({ platform: 'unified', ramGb: 32, chipId }, hardware);
    // An enumeration now, not a bound: 36 and 48 sit on the shared ladder
    // between 32 and 64, and an M1 Max was never sold with either.
    expect(opts('m1-max')).toEqual([32, 64]);
    expect(opts('m4-max-32c')).toEqual([36]);
    expect(opts('m3-pro')).toEqual([18, 36]);
    expect(opts('m5-ultra')).toEqual([96, 256, 512]);
    expect(opts('m1')).toEqual([8, 16]);
    // No chip named: the whole ladder, since we cannot narrow an unknown machine.
    expect(ramOptionsFor({ platform: 'discrete', ramGb: 64, vramGb: 24 }, hardware))
      .toEqual(hardware.ramOptions);
  });

  it('keeps every chip capacity reachable on the shared ladder', () => {
    // A capacity a chip offers but the ladder lacks would vanish from the
    // dropdown the moment no chip is selected.
    const ladder = new Set(hardware.ramOptions);
    for (const c of hardware.chips) {
      for (const g of c.ramOptions) expect(ladder.has(g), `${c.id}/${g}GB`).toBe(true);
    }
  });

  it('generates one entry per real Apple machine, derived from the chips', () => {
    // The Apple row is the chip list, because capacity cannot identify a Mac.
    // Derived rather than stored: a second copy in hardware.json could disagree.
    const generated = chipPresets(hardware);
    const expected = hardware.chips.reduce((n, c) => n + c.ramOptions.length, 0);
    expect(generated).toHaveLength(expected);
    expect(new Set(generated.map((p) => p.id)).size).toBe(expected); // ids unique
    expect(generated.every((p) => p.group === 'Apple silicon')).toBe(true);
    expect(generated.every((p) => p.subgroup && p.chipId)).toBe(true);
    // A machine the hand-written list of eight did not contain.
    const m1max32 = generated.find((p) => p.id === 'm1-max-32');
    expect(m1max32?.label).toBe('M1 Max · 32GB');
    expect(bandwidthFor({ platform: 'unified', ramGb: 32, chipId: m1max32!.chipId }, hardware)).toBe(400);
    // Every generated capacity is one its chip actually shipped with.
    for (const g of generated) {
      const chip = hardware.chips.find((c) => c.id === g.chipId)!;
      expect(chip.ramOptions, g.id).toContain(g.ramGb);
    }
  });

  it('REGRESSION: every machine in the dropdown can hold its own capacity', () => {
    // The invariant behind the control ordering. Selecting "M5 Max 128GB" set
    // the capacity while the select still listed the PREVIOUS chip's options;
    // 128 was absent, the assignment silently did nothing, and the clamp landed
    // on 48 — the first capacity that chip offers. The budget line then read
    // "36GB at full speed, 75% of 48GB" for a 128GB machine.
    //
    // No DOM here, so this asserts the property the fix relies on: a preset's
    // capacity is always among the options its own chip allows, which makes the
    // clamp a no-op whenever the controls are applied in the right order.
    for (const p of chipPresets(hardware)) {
      const allowed = ramOptionsFor({ platform: p.platform, ramGb: p.ramGb, chipId: p.chipId }, hardware);
      expect(allowed, `${p.label}`).toContain(p.ramGb);
      // and the nearest-match clamp therefore returns the asked-for capacity
      const nearest = allowed.reduce((a, b) => (Math.abs(b - p.ramGb) < Math.abs(a - p.ramGb) ? b : a));
      expect(allowed.includes(p.ramGb) ? p.ramGb : nearest, `${p.label}`).toBe(p.ramGb);
    }
  });

  it('REGRESSION: equal-length capacity lists still differ', () => {
    // syncControls used to rebuild the RAM dropdown only when the option COUNT
    // changed, so switching M1 Pro -> M1 Max left 16/32 on screen for a chip
    // that starts at 32. Both lists are length 2; only the contents differ.
    const opts = (chipId: string) => ramOptionsFor({ platform: 'unified', ramGb: 32, chipId }, hardware);
    const pro = opts('m1-pro');
    const max = opts('m1-max');
    expect(pro).toHaveLength(max.length);
    expect(pro).not.toEqual(max);
  });

  it('tells apart machines the numbers alone cannot', () => {
    // Eight presets are discrete/24GB; without the id they all look like a 3090.
    const at24 = { platform: 'discrete' as const, ramGb: 64, vramGb: 24 };
    expect(bandwidthFor({ ...at24, presetId: 'rtx-4090' }, hardware)).toBe(1008);
    expect(bandwidthFor({ ...at24, presetId: 'tesla-m40' }, hardware)).toBe(288);
    expect(bandwidthFor({ ...at24, presetId: 'l4' }, hardware)).toBe(300);
  });

  it('lets a preset override its platform prefill class', () => {
    // discrete defaults to CUDA-flavoured fast; ROCm and the pre-Pascal Teslas do not.
    const at24 = { platform: 'discrete' as const, ramGb: 64, vramGb: 24 };
    expect(prefillFor({ ...at24, presetId: 'rtx-4090' }, hardware)).toBe('fast');
    expect(prefillFor({ ...at24, presetId: 'tesla-p40' }, hardware)).toBe('slow');
    expect(prefillFor({ ...at24, presetId: 'arc-pro-b60' }, hardware)).toBe('moderate');
    expect(prefillFor({ platform: 'unified', ramGb: 32 }, hardware)).toBe('moderate');
  });

  it('lets a chip override its platform prefill class, and the M5 family is the one that does', () => {
    // llama.cpp discussion #4167, Llama-2-7B Q4_0 pp512: M5 Max 40-core 3,220
    // tok/s against 886 on the M4 Max 40-core and 1,471 on the M3 Ultra 80-core.
    // The base M5's 723 is a 32-core M4 Max figure, so it keeps the platform class.
    const mac = (chipId: string, ramGb: number) => ({ platform: 'unified' as const, ramGb, chipId });
    expect(prefillFor(mac('m5-max-40c', 64), hardware)).toBe('fast');
    expect(prefillFor(mac('m5-pro', 48), hardware)).toBe('fast');
    expect(prefillFor(mac('m5-ultra', 256), hardware)).toBe('fast');
    expect(prefillFor(mac('m5', 32), hardware)).toBe('moderate');
    expect(prefillFor(mac('m4-max-40c', 64), hardware)).toBe('moderate');
    expect(prefillFor(mac('m3-ultra', 256), hardware)).toBe('moderate');
    // Through the generated preset too, which is how the picker reaches it
    // (Picker.astro merges the chip presets into the list before use).
    const merged = { ...hardware, presets: [...hardware.presets, ...chipPresets(hardware)] };
    expect(prefillFor({ platform: 'unified', ramGb: 64, presetId: 'm5-max-40c-64' }, merged)).toBe('fast');
    // Every fast Apple chip says what the grade depends on, with its source.
    for (const c of hardware.chips.filter((x) => x.prefillClass === 'fast')) {
      expect(c.note, c.id).toMatch(/current llama.cpp build on macOS 26/);
      expect(c.noteUrl, c.id).toContain('llama.cpp/discussions/4167');
    }
    const preset = chipPresets(hardware).find((p) => p.id === 'm5-max-40c-64')!;
    expect(preset.note).toBe(hardware.chips.find((c) => c.id === 'm5-max-40c')!.note);
  });

  it('groups machines of equal capacity without mixing VRAM and RAM', () => {
    const peers = sameCapacityPresets({ platform: 'discrete', ramGb: 64, vramGb: 24 }, hardware);
    expect(peers.length).toBeGreaterThan(5);
    expect(peers.every((p) => p.vramGb === 24)).toBe(true);
    // A 24GB Mac holds 24GB of RAM, not VRAM, and must not appear here.
    expect(peers.some((p) => p.platform === 'unified')).toBe(false);
  });
});

describe('scoreFor', () => {
  const s = scores('m', [
    { band: 'mid', dimension: 'math', score: 6, provenance: [] as never },
  ]);
  it('returns exact band scores unborrowed', () => {
    expect(scoreFor(s, 'mid', 'math')).toEqual({ score: 6, borrowed: false });
  });
  it('borrows from adjacent bands minus distance', () => {
    expect(scoreFor(s, 'low', 'math')).toEqual({ score: 5, borrowed: true });
    expect(scoreFor(s, 'full', 'math')).toEqual({ score: 4, borrowed: true });
  });
  it('returns null with no evidence anywhere', () => {
    expect(scoreFor(s, 'mid', 'coding')).toBeNull();
    expect(scoreFor(undefined, 'mid', 'math')).toBeNull();
  });
});

describe('systemBandwidthFor / layerSplit', () => {
  it('gives every machine that can offload a speed for its CPU half', () => {
    // The rule validate.ts enforces, asserted from the other side: a machine
    // that can split a model but has no system-memory figure loses its tok/s
    // silently, which is the failure the whole change exists to remove.
    for (const p of hardware.presets) {
      if (p.platform !== 'discrete' && p.platform !== 'cpu') continue;
      const m: Machine = { platform: p.platform, ramGb: p.ramGb, vramGb: p.vramGb, presetId: p.id };
      expect(systemBandwidthFor(m, hardware), p.id).toBeGreaterThan(0);
    }
  });

  it('lets the reader override the machine default', () => {
    const base: Machine = { platform: 'discrete', ramGb: 64, vramGb: 24, presetId: 'rtx-3090' };
    const ddr4 = systemBandwidthFor(base, hardware)!;
    const server = systemBandwidthFor({ ...base, systemMemoryId: 'ddr5-twelve' }, hardware)!;
    expect(server).toBeGreaterThan(ddr4 * 5);
  });

  it('reads the whole bandwidth off system memory when there is no card', () => {
    // A CPU-only preset carries no bandwidth of its own any more: holding the
    // same number in two places is how the copy that drifts gets made.
    const m: Machine = { platform: 'cpu', ramGb: 128, presetId: 'cpu-128' };
    expect(bandwidthFor(m, hardware)).toBe(systemBandwidthFor(m, hardware));
  });

  it('states a partial fit as the layer count the reader types', () => {
    const bands = { mid: { weightsGb: 40, exampleQuants: ['Q4_K_M' as const] } };
    const m = model('L', { paramsB: 70, layers: 80, bands });
    const machine: Machine = { platform: 'discrete', ramGb: 128, vramGb: 24 };
    const f = fitFor(m, 'mid', machine, hardware, 2048)!;
    expect(f.offload).toBe(true);
    const sp = layerSplit(m, f)!;
    expect(sp.total).toBe(80);
    expect(sp.onGpu).toBeGreaterThan(0);
    expect(sp.onGpu).toBeLessThan(80);
    expect(sp.onGpu + sp.onCpu).toBe(80);
    // It has to agree with the fraction the speed estimate uses, or the site
    // quotes a speed for one split and a command for a different one.
    expect(sp.onGpu / sp.total).toBeCloseTo(f.residentFraction, 1);
  });

  it('says nothing rather than guessing when a model has no layer count', () => {
    const bands = { mid: { weightsGb: 40, exampleQuants: ['Q4_K_M' as const] } };
    const m = model('N', { paramsB: 70, layers: undefined, bands });
    const f = fitFor(m, 'mid', { platform: 'discrete', ramGb: 128, vramGb: 24 }, hardware, 2048)!;
    expect(layerSplit(m, f)).toBeNull();
    // And nothing to say when the whole model is on the card.
    const big = fitFor(m, 'mid', { platform: 'discrete', ramGb: 128, vramGb: 80 }, hardware, 2048)!;
    expect(layerSplit({ layers: 80 }, big)).toBeNull();
  });

  it('every model carries a layer count, so the split is always sayable', () => {
    const real = loadSiteData();
    for (const m of real.models) expect(m.layers, m.id).toBeGreaterThan(0);
  });
});

describe('machine coverage', () => {
  const presets = hardware.presets;

  it('never gives a laptop card its desktop namesake bandwidth', () => {
    // The entire reason the mobile rows exist. A copy-paste from the desktop
    // row would silently undo it and nothing else would notice.
    const laptops = presets.filter((p) => p.group === 'NVIDIA laptop');
    expect(laptops.length).toBeGreaterThan(9);
    for (const l of laptops) {
      const name = l.label.replace(/ \d+GB$/, '');
      const twin = presets.find((p) => p.group === 'NVIDIA' && p.label.startsWith(name + ' '));
      if (!twin) continue;
      expect(
        l.memoryBandwidthGbs === twin.memoryBandwidthGbs && l.vramGb === twin.vramGb,
        `${l.label} is indistinguishable from desktop ${twin.label}`,
      ).toBe(false);
    }
  });

  it('warns on every laptop card, because the name is the trap', () => {
    for (const l of presets.filter((p) => p.group === 'NVIDIA laptop')) {
      expect(l.note, l.id).toBeTruthy();
      expect(l.noteUrl, l.id).toBeTruthy();
    }
  });

  it('says so on every machine whose memory is not one pool', () => {
    // A multi-GPU board that reports a combined figure has to say it is not a
    // single pool, or the site promises a model that will not load.
    const multi = presets.filter((p) => /\bx\s?\d|\d\s?×|Dual|^\dx /i.test(p.label));
    expect(multi.length).toBeGreaterThan(0);
    for (const m of multi) expect(m.note, m.id).toBeTruthy();
  });

  it('covers the machine the feedback said was missing', () => {
    // "If you pick CPU Only you only get to choose 32GB or 64GB, so my machine
    // (128GB) isn't even covered." It is now, from the preset row itself.
    const cpu = presets.filter((p) => p.platform === 'cpu').map((p) => p.ramGb);
    expect(cpu).toContain(128);
    expect(Math.max(...cpu)).toBeGreaterThanOrEqual(512);
  });

  it('offers laptops as their own row, not buried among desktop cards', () => {
    // The other half of the same feedback: "desktop vs laptop".
    const groups = new Set(presets.map((p) => p.group));
    expect(groups).toContain('NVIDIA laptop');
  });

  it('resolves a bandwidth for every machine in the list', () => {
    for (const p of presets) {
      const m: Machine = { platform: p.platform, ramGb: p.ramGb, vramGb: p.vramGb, chipId: p.chipId, presetId: p.id };
      expect(bandwidthFor(m, hardware), p.id).toBeGreaterThan(0);
    }
  });
});

describe('runtime overhead — derived, not fitted', () => {
  const hw = hardware;

  it('MECHANISM: reproduces llama.cpp\'s documented compute buffer', () => {
    // The load-bearing claim of the whole term. llama.cpp's own discussion
    // reports a 507MB compute buffer for Gemma 2 9B. The final vocabulary
    // projection for one micro-batch is 512 x 256000 x 4 = 500MB. If that is
    // a coincidence rather than the mechanism, this test is where it shows.
    const gemma2 = { vocabSize: 256000, width: 14336 };
    const o = overheadFor(gemma2, 0, hw)!;
    const documentedMb = 507;
    const predictedMb = o.logitsGb * 1024;
    expect(Math.abs(predictedMb - documentedMb) / documentedMb).toBeLessThan(0.05);
    // And it really is the dominant piece, not one term among equals.
    expect(o.logitsGb).toBeGreaterThan(o.activationsGb * 3);
  });

  it('scales with vocabulary, which is what actually sizes the buffer', () => {
    const narrow = overheadFor({ vocabSize: 100352, width: 8192 }, 0, hw)!;
    const wide = overheadFor({ vocabSize: 262208, width: 8192 }, 0, hw)!;
    expect(wide.hiGb).toBeGreaterThan(narrow.hiGb);
  });

  it('scales with width at equal vocabulary', () => {
    const thin = overheadFor({ vocabSize: 131072, width: 2816 }, 0, hw)!;
    const fat = overheadFor({ vocabSize: 131072, width: 59136 }, 0, hw)!;
    expect(fat.hiGb).toBeGreaterThan(thin.hiGb);
  });

  it('REGRESSION: does not depend on weights, which was the whole defect', () => {
    // The old term charged a 397B-A17B 15.6GB of scratch and a model a third as
    // wide three times more than a wider one, because it read parameter count.
    // Two bands of the same model differ only in weights, so their overhead at
    // the same context must be identical.
    const real = loadSiteData();
    const m = real.models.find((x) => x.bands.low && x.bands.high)!;
    const a = fitFor(m, 'low', { platform: 'unified', ramGb: 512 }, hardware, 8192)!;
    const b = fitFor(m, 'high', { platform: 'unified', ramGb: 512 }, hardware, 8192)!;
    expect(a.weightsGb).not.toBe(b.weightsGb);
    expect(a.overheadGb).toBe(b.overheadGb);
  });

  it('grows with context, and only through the KQ mask', () => {
    const short = overheadFor({ vocabSize: 131072, width: 8192 }, 4096, hw)!;
    const long = overheadFor({ vocabSize: 131072, width: 8192 }, 262144, hw)!;
    expect(long.hiGb).toBeGreaterThan(short.hiGb);
    expect(long.logitsGb).toBe(short.logitsGb);
    expect(long.activationsGb).toBe(short.activationsGb);
    expect(long.kqMaskGb).toBeGreaterThan(short.kqMaskGb);
    // ~1KB per token, so it belongs with the model's own KV rate.
    expect(kqBytesPerToken(hw)).toBe(1024);
  });

  it('SANITY: no model in the catalogue is charged an absurd scratch', () => {
    // 15.6GB on a 397B was the tell that nothing was watching this.
    const real = loadSiteData();
    for (const m of real.models) {
      const o = overheadFor(m, 262144, hardware);
      expect(o, m.id).not.toBeNull();
      expect(o!.hiGb, m.id).toBeLessThan(3);
      expect(o!.hiGb, m.id).toBeGreaterThan(0.5);
    }
  });

  it('fails closed when a model cannot support the derivation', () => {
    expect(overheadFor({ vocabSize: undefined, width: 8192 }, 0, hw)).toBeNull();
    expect(overheadFor({ vocabSize: 131072, width: undefined }, 0, hw)).toBeNull();
    const noFields = model('nf', { vocabSize: undefined, bands: { mid: { weightsGb: 4, exampleQuants: ['Q4_K_M'] } } });
    expect(fitFor(noFields, 'mid', { platform: 'unified', ramGb: 64 }, hardware, 8192)).toBeNull();
  });

  it('prices what turning flash attention off would cost', () => {
    // Not part of any estimate — but it is the difference between a rounding
    // error and gigabytes, so the site has to be able to say it.
    const m = { attentionHeads: 32 };
    const at32k = flashAttentionOffCostGb(m, 32768, hw)!;
    const at128k = flashAttentionOffCostGb(m, 131072, hw)!;
    expect(at32k).toBeGreaterThan(1.5);
    expect(at32k).toBeLessThan(2.5);
    expect(at128k).toBeGreaterThan(7);
    // Dwarfs the masked term it replaces, which is the point.
    const masked = overheadFor({ vocabSize: 131072, width: 8192 }, 131072, hw)!;
    expect(at128k).toBeGreaterThan(masked.kqMaskGb * 20);
    expect(flashAttentionOffCostGb({ attentionHeads: undefined }, 1024, hw)).toBeNull();
  });

  it('reports max context as a range, the pessimistic end being the verdict', () => {
    const real = loadSiteData();
    const m = real.models.find((x) => x.id === 'qwen3.8-27b')!;
    const f = fitFor(m, 'mid', { platform: 'unified', ramGb: 32 }, hardware, 32768)!;
    expect(f.maxContextTokensOptimistic).toBeGreaterThan(f.maxContextTokens);
  });
});

describe('one answer per question', () => {
  it('REGRESSION: the Explore table charges the same demand as the picker', () => {
    // These disagreed. Explore computed weights + KV and omitted the runtime
    // entirely, so the same site answered "does it fit" two ways and Explore
    // was the optimistic one by 1-4GB. This reproduces Explore's calculation
    // from the same inputs its table rows carry, and requires it to match.
    const real = loadSiteData();
    const machine: Machine = { platform: 'unified', ramGb: 32 };
    for (const m of real.models) {
      for (const band of BANDS) {
        const info = m.bands[band];
        if (!info) continue;
        for (const ctx of [4096, 32768, 131072]) {
          const f = fitFor(m, band, machine, hardware, ctx);
          if (!f) continue;
          const effective = Math.min(ctx, m.contextLength);
          const kv = kvGbFrom(
            m.kvBytesPerToken!,
            m.kvWindowedBytesPerToken,
            m.slidingWindow || undefined,
            effective,
          );
          const oh = overheadFor(
            { vocabSize: m.vocabSize, width: m.width },
            effective,
            hardware,
          )!;
          const exploreDemand = info.weightsGb + kv + oh.hiGb;
          expect(exploreDemand, `${m.id} ${band} @${ctx}`).toBeCloseTo(f.demandGb, 1);
        }
      }
    }
  });
});

describe('unified memory has one pool, not two', () => {
  it('REGRESSION: refuses to invent a system-memory speed for a shared pool', () => {
    // This returned 90GB/s — the DDR5-desktop default — for an M1 Max, whose
    // memory runs at 400. Never displayed, because unified fits never offload,
    // but one code change away from being. There is no honest number here.
    const mac: Machine = { platform: 'unified', ramGb: 32, chipId: 'm1-max' };
    expect(systemBandwidthFor(mac, hardware)).toBeUndefined();
    expect(bandwidthFor(mac, hardware)).toBe(400);
    for (const p of ['unified', 'unified-amd', 'unified-nvidia'] as const) {
      expect(hasSeparateSystemMemory(p), p).toBe(false);
      expect(systemBandwidthFor({ platform: p, ramGb: 64 }, hardware), p).toBeUndefined();
    }
    // And still answers where the CPU really does have its own memory.
    for (const p of ['discrete', 'cpu'] as const) {
      expect(hasSeparateSystemMemory(p), p).toBe(true);
      expect(systemBandwidthFor({ platform: p, ramGb: 64 }, hardware), p).toBeGreaterThan(0);
    }
  });

  it('never offers to split a model on a shared pool', () => {
    // Not an oversight. The split is possible on a Mac and buys nothing: one
    // pool, so the CPU adds no memory and no bandwidth, and reads it slower.
    const real = loadSiteData();
    for (const ram of [16, 32, 64, 128]) {
      const mac: Machine = { platform: 'unified', ramGb: ram, chipId: 'm1-max' };
      for (const m of real.models) {
        for (const band of BANDS) {
          if (!m.bands[band]) continue;
          const f = fitFor(m, band, mac, hardware, 8192);
          expect(f?.offload ?? false, `${m.id} ${band} @${ram}GB`).toBe(false);
        }
      }
    }
  });

  it('still leaves the wired limit as the lever it actually is', () => {
    // The honest alternative the site offers instead of a split, and the reason
    // the missing split is not a gap.
    const platform = hardware.platforms.find((p) => p.id === 'unified')!;
    expect(platform.ceiling.overridable).toBe(true);
    expect(platform.advice).toBeTruthy();
    expect(maxWiredGb({ platform: 'unified', ramGb: 32 }, hardware)).toBeGreaterThan(24);
  });
});

describe('multi-card entries', () => {
  const presets = hardware.presets;
  const nvidia = presets.filter((p) => p.group === 'NVIDIA');

  it('gives every current-generation NVIDIA card a two-card entry, directly beneath it', () => {
    // Scoped to the generations still sold, which is what this rule was always
    // protecting: someone shopping for a second card is shopping among these.
    // The legacy generations get a pair only where stacking that card was
    // itself a common build (the 1080 Ti and 2080 Ti), because listing "2x GTX
    // 1650" would be inventing a configuration rather than describing one.
    const current = new Set(['GeForce RTX 30', 'GeForce RTX 40', 'GeForce RTX 50', 'RTX professional']);
    const singles = nvidia.filter((p) => current.has(p.subgroup!) && !/^\dx /.test(p.label));
    expect(singles.length).toBeGreaterThan(9);
    for (const s of singles) {
      const i = nvidia.indexOf(s);
      const next = nvidia[i + 1];
      expect(next, `${s.label} has nothing after it`).toBeTruthy();
      expect(next!.id, `${s.label} is not followed by its 2x`).toBe(`${s.id}-x2`);
      expect(next!.vramGb).toBe(s.vramGb! * 2);
    }
  });

  it('every pair that IS listed sits directly under its own card', () => {
    // The structural half, which applies to all of them regardless of
    // generation: a pair that drifts away from its single, or reports the wrong
    // capacity, is worse than an absent one.
    const pairs = presets.filter((p) => /-x2$/.test(p.id));
    expect(pairs.length).toBeGreaterThan(9);
    for (const pair of pairs) {
      const base = presets.find((x) => x.id === pair.id.replace(/-x2$/, ''));
      expect(base, `${pair.id} has no single`).toBeTruthy();
      expect(presets[presets.indexOf(base!) + 1]!.id, `${pair.id} is not under its single`).toBe(pair.id);
      expect(pair.vramGb).toBe(base!.vramGb! * 2);
      expect(pair.subgroup, pair.id).toBe(base!.subgroup);
    }
  });

  it('every NVIDIA card is filed under a generation', () => {
    // 48 cards in one flat list is the Mac problem again. The subgroup is what
    // keeps it scannable, so a card without one disappears into the wrong place.
    for (const p of nvidia) expect(p.subgroup, `${p.label} has no subgroup`).toBeTruthy();
  });

  it('holds twice the memory at one card\'s speed, never twice the speed', () => {
    // The trap the notes exist to head off. Two cards double what fits and do
    // not double tokens per second: llama.cpp runs their layers in sequence.
    for (const m of presets.filter((p) => /^\dx /.test(p.label))) {
      const base = presets.find((p) => p.id === m.id.replace(/-x\d$/, ''));
      if (!base) continue;
      expect(m.memoryBandwidthGbs, m.id).toBe(base.memoryBandwidthGbs);
      expect(m.vramGb!, m.id).toBeGreaterThan(base.vramGb!);
      expect(m.note, `${m.id} must say it is not one pool`).toBeTruthy();
    }
  });

  it('REGRESSION: every discrete preset capacity is on the VRAM ladder', () => {
    // Same failure as the M5 Max RAM case, on the other control: a capacity the
    // preset offers but the select cannot list is silently replaced by another.
    for (const p of presets) {
      if (!p.vramGb) continue;
      expect(hardware.vramOptions, `${p.label} (${p.vramGb}GB)`).toContain(p.vramGb);
    }
  });
});

describe('the wired limit is offered, never vouched for', () => {
  it('no raise option calls itself safe', () => {
    // The request behind all of this: nobody here is the authority on what a
    // stranger's machine survives, so no string may say one value is safe.
    // Asserted over generated text rather than by grep, so a future label
    // cannot quietly reintroduce it.
    for (const ram of hardware.ramOptions) {
      const m: Machine = { platform: 'unified', ramGb: ram };
      const pub = publishedCeilingGb(m, hardware);
      for (const g of raiseSteps(m, hardware)) {
        const left = Math.round((ram - g) * 10) / 10;
        const label = `${g}GB — leaves ${left}GB${pub !== null && g > pub ? ', past published advice' : ''}`;
        expect(label.toLowerCase(), `${ram}GB`).not.toContain('safe');
        expect(label).toContain('leaves');
      }
    }
    const platform = hardware.platforms.find((p) => p.id === 'unified')!;
    for (const s of [platform.ceiling.raise?.note, platform.advice?.newbieExplainer, platform.advice?.caveat]) {
      expect((s ?? '').toLowerCase()).not.toMatch(/\bsafe max\b|\bsafely\b/);
    }
  });

  it('every raise claim carries a source', () => {
    // The site's own standard, from schemas.ts: a claim about the world carries
    // a citation. "You can wire down 30GB of a 32GB machine" is such a claim,
    // and it shipped with none.
    for (const p of hardware.platforms) {
      if (!p.ceiling.raise) continue;
      expect(p.ceiling.raise.citation, p.id).toBeTruthy();
    }
  });

  it('now sits inside published advice everywhere, and says so by not marking', () => {
    // The marker was added when the offer ran past the cited reserve — 61GB on
    // a 64GB Mac against a published 56. Dropping the second reserve moved the
    // offer below published advice at every capacity, so nothing is marked.
    // Asserted rather than left implicit: if the offer ever climbs back past a
    // cited figure, this fails and the marker has to earn its place again.
    for (const ram of [64, 128, 256]) {
      const m: Machine = { platform: 'unified', ramGb: ram };
      const pub = publishedCeilingGb(m, hardware)!;
      expect(pub).toBeGreaterThan(0);
      for (const g of raiseSteps(m, hardware)) {
        expect(g, `${ram}GB offers ${g}GB, past the published ${pub}GB`).toBeLessThanOrEqual(pub);
      }
    }
    expect(publishedCeilingGb({ platform: 'unified', ramGb: 32 }, hardware)).toBeNull();
  });

  it('REGRESSION: one number per screen', () => {
    // The prose said "about 26GB", the dropdown offered 30GB and the command
    // said 30720 — three answers in one view, because the advice used the OS
    // reserve while the control used the raise arithmetic.
    const real: PickerData = {
      hardware,
      tasks: loadSiteData().tasks,
      mitigations: loadSiteData().mitigations,
      models: loadSiteData().models,
      scores: Object.fromEntries(loadSiteData().scores),
      weaknesses: Object.fromEntries(loadSiteData().weaknesses),
    };
    for (const ram of [32, 64, 128]) {
      const m: Machine = { platform: 'unified', ramGb: ram };
      const max = maxWiredGb(m, hardware)!;
      const steps = raiseSteps(m, hardware);
      expect(steps[steps.length - 1], `${ram}GB last step`).toBe(max);
      const out = pickModels(m, real.tasks[0]!.id, real);
      if (out.ceilingGain) {
        expect(out.ceilingGain, `${ram}GB advice prose`).toContain(String(Math.round(max)));
      }
    }
  });
});

describe('a weak top pick says so', () => {
  const real = (): PickerData => {
    const d = loadSiteData();
    return {
      hardware,
      tasks: d.tasks,
      mitigations: d.mitigations,
      models: d.models,
      scores: Object.fromEntries(d.scores),
      weaknesses: Object.fromEntries(d.weaknesses),
    };
  };
  const card8 = (): Machine => ({ platform: 'discrete', ramGb: 32, vramGb: 8, systemMemoryId: 'ddr5-dual' });

  it('THE REPORTED CASE: an 8GB card asked for code', () => {
    // Gemma 3 4B is the right pick — it is the best model that fits entirely on
    // a 7GB budget. It also scores 4 out of 10 at coding, and used to be stamped
    // "Top pick" with nothing saying otherwise.
    const data = real();
    const task = data.tasks.find((t) => t.id.includes('cod'))!;
    const out = pickModels(card8(), task.id, data);
    const top = out.recommendations[0]!;
    // Was Gemma 3 4B. Nanbeige4.2 3B now ties it at 4.5 and leads on the
    // tiebreak, because correcting its doubled KV figure let it fit at this
    // context at all. The reported case is about the LABEL, not which model
    // wins it — so this asserts the tie and the stamp, not a specific id.
    expect(['nanbeige4.2-3b', 'gemma-3-4b']).toContain(top.model.id);
    expect(top.tags).toContain('top-pick');
    expect(top.score).toBeLessThan(NEUTRAL_SCORE);
    expect(top.weakBest).toBe(true);
    // and something better is named rather than left to be inferred
    expect(out.betterThanWeak).toBeTruthy();
    expect(out.betterThanWeak!.score).toBeGreaterThan(top.score);
  });

  it('uses the site\'s own neutral score as the line, not a number of its own', () => {
    // NEUTRAL_SCORE is what this file already assigns to a dimension with no
    // evidence at all. A top pick below it scores worse than total ignorance,
    // which is what makes the threshold principled rather than chosen. Read
    // from the constant so the two cannot drift apart.
    const data = real();
    for (const t of data.tasks) {
      for (const m of [card8(), { platform: 'unified', ramGb: 64, chipId: 'm1-max' } as Machine]) {
        const top = pickModels(m, t.id, data).recommendations[0];
        if (!top) continue;
        expect(!!top.weakBest, `${t.id} @${top.score}`).toBe(top.score < NEUTRAL_SCORE);
      }
    }
  });

  it('stays rare enough to keep meaning something', () => {
    // A caveat on a fifth of all answers is wallpaper. Measured at 9.8% when
    // this landed; the bound is loose enough to survive data changes and tight
    // enough to fail if it starts firing everywhere.
    //
    // Measured over machines with room to give a good answer. A flat rate over
    // every preset stopped meaning anything once the legacy GeForce cards went
    // in: on a 4GB card the best available model really is weak, so the warning
    // firing there is the feature working, and averaging it together with a
    // 48GB card only hides whether it has started firing where it should not.
    // The companion test below asserts the other half.
    const data = real();
    let total = 0;
    let weak = 0;
    for (const p of hardware.presets) {
      if ((p.vramGb ?? p.ramGb) < 12) continue;
      const m: Machine = {
        platform: p.platform,
        ramGb: p.ramGb,
        vramGb: p.vramGb,
        chipId: p.chipId,
        systemMemoryId: p.systemMemoryId,
        presetId: p.id,
      };
      for (const t of data.tasks) {
        const top = pickModels(m, t.id, data).recommendations[0];
        if (!top) continue;
        total++;
        if (top.weakBest) weak++;
      }
    }
    expect(total).toBeGreaterThan(500);
    expect(weak / total).toBeLessThan(0.15);
    expect(weak).toBeGreaterThan(0);
  });

  it('and fires reliably on the machines that need it', () => {
    // The other half. A warning that is rare everywhere is not calibrated, it is
    // broken -- on a 4 to 6GB card the honest answer is that the best thing that
    // fits is not good, and saying nothing there would be the real failure.
    const data = real();
    let total = 0;
    let weak = 0;
    for (const p of hardware.presets) {
      if (p.platform !== 'discrete' || (p.vramGb ?? 0) > 6) continue;
      const m: Machine = {
        platform: p.platform,
        ramGb: p.ramGb,
        vramGb: p.vramGb,
        systemMemoryId: p.systemMemoryId,
        presetId: p.id,
      };
      for (const t of data.tasks) {
        const top = pickModels(m, t.id, data).recommendations[0];
        if (!top) continue;
        total++;
        if (top.weakBest) weak++;
      }
    }
    expect(total).toBeGreaterThan(20);
    expect(weak / total).toBeGreaterThan(0.4);
  });

  it('never calls a strong pick weak', () => {
    const data = real();
    for (const p of hardware.presets.slice(0, 40)) {
      const m: Machine = { platform: p.platform, ramGb: p.ramGb, vramGb: p.vramGb, chipId: p.chipId, presetId: p.id };
      for (const t of data.tasks) {
        for (const r of pickModels(m, t.id, data).recommendations) {
          if (r.score >= NEUTRAL_SCORE) expect(r.weakBest ?? false, `${r.model.id} @${r.score}`).toBe(false);
        }
      }
    }
  });

  it('REGRESSION: labelling only — the ranking is untouched', () => {
    // The whole point is that the pick was already correct. If this ever starts
    // reordering anything it has stopped being a labelling change.
    const data = real();
    for (const p of hardware.presets.slice(0, 30)) {
      const m: Machine = { platform: p.platform, ramGb: p.ramGb, vramGb: p.vramGb, chipId: p.chipId, presetId: p.id };
      for (const t of data.tasks) {
        const out = pickModels(m, t.id, data);
        const top = out.recommendations[0];
        if (!top) continue;
        // the leader is still whatever scores highest among what is offered
        const best = Math.max(...out.recommendations.map((r) => r.score));
        expect(top.score, `${p.id}/${t.id}`).toBe(best);
        expect(top.tags).toContain('top-pick');
      }
    }
  });
});

describe('speed is a latency, not a bandwidth fraction', () => {
  const speedFor = (m: Model, band: Band, machine: Machine, ctx: number, bw: number) => {
    const f = fitFor(m, band, machine, hardware, ctx)!;
    return generationSpeed(f.weightsGb, f.kvGb, m, f, bw, 90, hardware, machine.platform)!;
  };

  it('carries a citation, per platform, with sane bounds', () => {
    const p = hardware.perTokenLatency;
    expect(p.citation).toBeTruthy();
    expect(p.loMs).toBeLessThan(p.hiMs);
    for (const [k, v] of Object.entries(p.byPlatform ?? {})) {
      expect(v.loMs, k).toBeLessThan(v.hiMs);
    }
  });

  it('ANCHOR: still reproduces the published RTX 4090 figure', () => {
    // The measurement the OLD model was built on. A replacement that breaks its
    // own anchor is not an improvement, so this is unchanged from before.
    const dense8b = model('a8', {
      paramsB: 8,
      kvBytesPerToken: 131072,
      bands: { mid: { weightsGb: 5, exampleQuants: ['Q4_K_M'] } },
    });
    const s = speedFor(dense8b, 'mid', { platform: 'discrete', ramGb: 64, vramGb: 24 }, 2048, 1008);
    expect(s.lo).toBeGreaterThan(70);
    expect(s.lo).toBeLessThan(100);
    expect(s.hi).toBeGreaterThan(110);
    expect(s.hi).toBeLessThan(150);
  });

  it('ANCHOR: brackets the sparse MoE measurement the old model missed', () => {
    // Qwen3.5-35B-A3B on an M4 Max 128GB, 40 layers, 4-bit on both sides:
    // llama.cpp 70.4-72.4 tok/s, MLX 126.4-131.8 (antekapetanovic.com, ten
    // runs each). Reading ~1.7GB per token, a bandwidth fraction predicted
    // 120-187 and put llama.cpp outside it entirely; the range is now wide
    // enough to hold both runtimes, which is what a Mac reader may be on.
    const moe = model('m35', {
      paramsB: 35,
      activeParamsB: 3,
      architecture: 'moe',
      layers: 40,
      kvBytesPerToken: 32768,
      bands: { mid: { weightsGb: 19, exampleQuants: ['Q4_K_M'] } },
    });
    const s = speedFor(moe, 'mid', { platform: 'unified', ramGb: 128, chipId: 'm4-max-40c' }, 2048, 546);
    expect(s.lo).toBeLessThanOrEqual(70.4);
    expect(s.hi).toBeGreaterThanOrEqual(131.8);
  });

  it('a fixed overhead means a real speed ceiling', () => {
    // The old model let tok/s grow without bound as the read shrank. Nothing
    // can beat 1/c however little it reads.
    const tiny = model('t', { paramsB: 1, kvBytesPerToken: 4096, bands: { low: { weightsGb: 0.1, exampleQuants: ['Q3_K_M'] } } });
    const s = speedFor(tiny, 'low', { platform: 'discrete', ramGb: 64, vramGb: 24 }, 2048, 2000);
    const ceiling = 1000 / (hardware.perTokenLatency.byPlatform?.discrete?.loMs ?? hardware.perTokenLatency.loMs);
    expect(s.hi).toBeLessThanOrEqual(Math.ceil(ceiling));
  });

  it('REGRESSION: small reads use less of the bandwidth than large ones', () => {
    // The direction is the mechanism: a fixed cost is a larger share of a
    // small token than of a big one, so the bandwidth a small read appears to
    // achieve is lower. A multiplicative derate cannot produce this — it gives
    // every model the same utilisation — and it is the reason sparse models
    // were overpredicted by about two times under the old fraction.
    const real = loadSiteData();
    const machine: Machine = { platform: 'unified', ramGb: 128, chipId: 'm4-max-40c' };
    const bw = 546;
    const util: { small: number[]; large: number[] } = { small: [], large: [] };
    for (const m of real.models) {
      const f = fitFor(m, 'mid', machine, hardware, 8192);
      if (!f || f.state === 'no') continue;
      const af = m.activeParamsB ? m.activeParamsB / m.paramsB : 1;
      const bytes = f.weightsGb * af + f.kvGb;
      const s = generationSpeed(f.weightsGb, f.kvGb, m, f, bw, 90, hardware, machine.platform)!;
      const mid = (s.lo + s.hi) / 2;
      // The predictor is BYTES READ, not architecture. Mistral Small 4 is
      // sparse — 6.5B of 119B active — but carries a 590KB/token KV cache, so
      // at 8K it reads ~8.9GB and behaves like a dense model. Asserting on
      // sparsity would encode the wrong mechanism.
      if (bytes < 4) util.small.push((mid * bytes) / bw);
      else if (bytes > 15) util.large.push((mid * bytes) / bw);
    }
    expect(util.small.length).toBeGreaterThan(3);
    expect(util.large.length).toBeGreaterThan(3);
    expect(Math.max(...util.small)).toBeLessThan(Math.min(...util.large));
  });
});

describe('Mac tok/s against published measurements', () => {
  // The two most authoritative Apple Silicon tables there are, one per runtime,
  // and the only ones with a stated machine, command and version on every row.
  // The range shown for a Mac has to hold both: a reader is on one or the other
  // and the site cannot tell which. See docs/mac-tok-s-validation.md for the
  // full 92-row corpus these were chosen from.
  const real = loadSiteData();
  const byId = (id: string) => real.models.find((m) => m.id === id)!;
  const speed = (chipId: string, ramGb: number, modelId: string, weightsGb: number, ctx: number) => {
    const m = byId(modelId);
    const machine: Machine = { platform: 'unified', ramGb, chipId };
    const bw = bandwidthFor(machine, hardware)!;
    const f = fitFor(m, 'mid', machine, hardware, ctx)!;
    return generationSpeed(weightsGb, f.kvGb, m, f, bw, undefined, hardware, 'unified')!;
  };

  it('brackets llama.cpp on every chip in its own Apple Silicon table', () => {
    // ggml-org/llama.cpp discussion #4167: LLaMA-2-7B Q4_0 (3.83GB, 32 layers),
    // `llama-bench -p 512 -n 128`. Llama-3.1-8B stands in for the layer count;
    // at 128 tokens the KV difference between the two is under 0.1GB.
    const rows: [string, number, number][] = [
      ['m1-max', 64, 61.19],
      ['m1-ultra', 128, 83.73],
      ['m2-max', 64, 65.95],
      ['m2-ultra', 192, 94.27],
      ['m3-max-30c', 36, 56.58],
      ['m3-max-40c', 64, 66.31],
      ['m3-ultra', 256, 92.14],
      ['m4-pro', 48, 50.74],
      ['m4-max-32c', 64, 69.95],
      ['m4-max-40c', 64, 83.06],
      ['m5', 32, 31.88],
      ['m5-pro', 48, 66.33],
      ['m5-max-40c', 128, 119.92],
    ];
    for (const [chip, ram, measured] of rows) {
      const s = speed(chip, ram, 'llama-3.1-8b', 3.83, 128);
      expect(s.lo, `${chip} lo`).toBeLessThanOrEqual(measured);
      // The site prints whole tok/s; the M5 Pro row lands on the rounding.
      expect(s.hi + 0.5, `${chip} hi`).toBeGreaterThanOrEqual(measured);
    }
  });

  it('brackets MLX in mlx-lm\'s own benchmark table', () => {
    // ml-explore/mlx-lm BENCHMARKS.md, M4 Max 64GB, `-p 2048 -g 128`, weights
    // at the sizes the table states. bf16 is the site's full band; the rest are
    // MLX's own 4.5 and 8.5 bits per weight, which no GGUF band matches.
    const rows: [string, number, number][] = [
      ['qwen3-4b', 2.5, 134.52],
      ['qwen3-4b', 4.3, 86.91],
      ['qwen3-4b', 8, 52.47],
      ['qwen3-30b-a3b', 18.2, 113.33],
      ['qwen3-30b-a3b', 33.46, 83.16],
    ];
    for (const [modelId, gb, measured] of rows) {
      const s = speed('m4-max-40c', 64, modelId, gb, 2176);
      expect(s.lo, `${modelId} ${gb}GB lo`).toBeLessThanOrEqual(measured);
      expect(s.hi, `${modelId} ${gb}GB hi`).toBeGreaterThanOrEqual(measured);
    }
  });

  it('does not open the range wider than the measurements need', () => {
    // A range that held everything by spanning 10-1000 would pass the two
    // tests above and tell the reader nothing. The 7B row on an M1 Max is the
    // narrowest case the corpus has, and it stays under 2x wide.
    const s = speed('m1-max', 64, 'llama-3.1-8b', 3.83, 128);
    expect(s.hi / s.lo).toBeLessThan(2);
  });
});

describe('architecture data is self-consistent', () => {
  const real = loadSiteData();

  it('the drawn pattern and the memory model agree about what caches', () => {
    // Two independent derivations: the per-layer attention pattern, from
    // layer_types / intervals / sliding-window fields, and kvBytesPerToken,
    // taken from the same config for the memory model. They must agree, and
    // disagreement means one is wrong -- which is how Nanbeige4.2 3B's doubled
    // cache figure surfaced. validate.ts enforces this at build; this asserts
    // it against the shipped data.
    //
    // Two things make this more than a division. Trailing KV-shared layers
    // allocate no cache, and full-attention layers may have a geometry of their
    // own -- both true of Gemma 4, and both missed when it was first modelled.
    let checked = 0;
    for (const m of real.models) {
      if (!m.attentionPattern || !m.kvHeads || !m.headDim || !m.kvBytesPerToken) continue;
      expect(m.attentionPattern.length, `${m.id} pattern length`).toBe(m.layers);
      expect(m.globalKvHeads === undefined, `${m.id} global pair`).toBe(m.globalHeadDim === undefined);
      const caching = m.attentionPattern.slice(0, m.layers - (m.kvSharedLayers ?? 0));
      const n = (c: string) => [...caching].filter((x) => x === c).length;
      expect(m.kvBytesPerToken, `${m.id} full`).toBe(
        4 * n('F') * (m.globalKvHeads ?? m.kvHeads) * (m.globalHeadDim ?? m.headDim),
      );
      expect(m.kvWindowedBytesPerToken ?? 0, `${m.id} windowed`).toBe(4 * n('S') * m.kvHeads * m.headDim);
      if (n('S') > 0) expect(m.slidingWindow, `${m.id} window`).toBeTruthy();
      checked++;
    }
    expect(checked).toBeGreaterThan(30);
  });

  it('REGRESSION: Gemma 4 gives its full-attention layers their own geometry', () => {
    // The family was modelled by copying the Gemma 3 recipe, which assumes one
    // head shape across every layer. Gemma 4 broke that: global_head_dim is 512
    // against the sliding layers' 256, over num_global_key_value_heads as low
    // as one against eight. The net is a much SMALLER cache than the naive
    // figure, so all five models were overcharged 2-4x -- the direction that
    // makes a model look unrunnable at a context it handles.
    //
    // Gemma 3, with no such fields, was correct throughout, which is what
    // established the method was sound and the Gemma 4 data was not.
    const g12 = real.models.find((m) => m.id === 'gemma-4-12b')!;
    expect(g12.globalKvHeads).toBe(1);
    expect(g12.globalHeadDim).toBe(512);
    expect(g12.headDim).toBe(256); // sliding layers keep their own shape
    expect(g12.kvBytesPerToken).toBe(16384); // 8 full layers, was 65,536
    expect(g12.kvWindowedBytesPerToken).toBe(327680);

    for (const id of ['gemma-3-4b', 'gemma-3-12b', 'gemma-3-27b']) {
      const m = real.models.find((x) => x.id === id)!;
      expect(m.globalHeadDim, id).toBeUndefined();
    }
  });

  it('REGRESSION: Gemma 4 E-series KV-shared layers cache nothing', () => {
    // The last num_kv_shared_layers layers have no k_proj or v_proj at all --
    // they read an earlier layer's keys and values. mlx-lm builds a cache only
    // for range(num_hidden_layers - num_kv_shared_layers), so those layers
    // allocate nothing however they attend. Counting all 42 of E4B's layers
    // charged cache for 18 that have none.
    const e4b = real.models.find((m) => m.id === 'gemma-4-e4b')!;
    expect(e4b.kvSharedLayers).toBe(18);
    expect(e4b.layers).toBe(42);
    // Full layers sit at every 6th index; only 4 of the 7 fall inside the first 24.
    expect(e4b.kvBytesPerToken).toBe(4 * 4 * 2 * 512);
    expect(e4b.kvWindowedBytesPerToken).toBe(4 * 20 * 2 * 256);

    const e2b = real.models.find((m) => m.id === 'gemma-4-e2b')!;
    expect(e2b.kvSharedLayers).toBe(20);
    expect(e2b.kvWindowedBytesPerToken).toBe(12288); // was 28,672, all 28 sliding layers
  });

  it('REGRESSION: Ornith 1.5 35B caches 10 full layers, not 8', () => {
    // full_attention_interval 4 over 40 layers is 10 full-attention layers.
    // The stored figure implied 8, so the model was undercharged -- the other
    // direction from Gemma 4, and the reason the check has to be an equality
    // rather than a ceiling.
    const m = real.models.find((x) => x.id === 'ornith-1.5-35b-a3b')!;
    expect(m.attentionPattern).toBe('...F...F...F...F...F...F...F...F...F...F');
    expect(m.kvBytesPerToken).toBe(20480);
  });

  it('REGRESSION: Mistral Small 4 is MLA, not MHA', () => {
    // Its config carries the full MLA parameter set — kv_lora_rank 256,
    // q_lora_rank, qk_nope/qk_rope splitting qk_head_dim — so it caches one
    // 320-wide latent per layer: 36 x 320 x 2 = 23,040 bytes per token. It
    // shipped at 589,824, the full-MHA figure, 25.6x too large. At 128K that is
    // 72GB of cache charged against a real 2.8GB.
    //
    // GLM-4.7-Flash, with the same field set, was already modelled correctly —
    // so this was an inconsistency inside our own data, not a hard question.
    const m = real.models.find((x) => x.id === 'mistral-small-4')!;
    expect(m.mlaLatentDim).toBe(320);
    expect(m.kvBytesPerToken).toBe(m.layers * m.mlaLatentDim! * 2);
    expect(m.kvBytesPerToken).toBe(23040);
  });

  it('every MLA model reconciles, and none is skipped', () => {
    // The earlier cross-check SKIPPED MLA models rather than checking them
    // against the MLA formula, which is precisely how the above shipped.
    const mla = real.models.filter((m) => m.mlaLatentDim);
    expect(mla.length).toBeGreaterThan(0);
    for (const m of mla) {
      expect(m.kvBytesPerToken, m.id).toBe(m.layers * m.mlaLatentDim! * 2);
      // kvHeads/headDim describe something MLA does not do; they must be absent.
      expect(m.kvHeads, m.id).toBeUndefined();
      expect(m.headDim, m.id).toBeUndefined();
    }
  });

  it('REGRESSION: Nanbeige4.2 3B caches 22 layers, not 44', () => {
    // 22 layers, 8 KV heads, head dim 128 -> 2 x 22 x 8 x 128 x 2 = 90112.
    // It shipped at 180224, the K and V factor counted twice, which charged it
    // double the cache and cost it roughly half its speed on a small card.
    const m = real.models.find((x) => x.id === 'nanbeige4.2-3b')!;
    expect(m.kvBytesPerToken).toBe(90112);
    expect(m.kvBytesPerToken / (4 * m.kvHeads! * m.headDim!)).toBe(m.layers);
  });

  it('no pattern is invented where the config does not support one', () => {
    // Absent is a valid answer and the diagram omits the row. What must never
    // happen is a default: an early sketch rendered Gemma 3 27B as all-full
    // attention because it lacks layer_types, when it interleaves 5 sliding per
    // global. Every pattern present must be derivable and must reconcile.
    const withPattern = real.models.filter((m) => m.attentionPattern);
    expect(withPattern.length).toBeGreaterThan(30);
    const gemma = real.models.find((m) => m.id === 'gemma-3-27b')!;
    expect(gemma.attentionPattern).toContain('S');
    expect((gemma.attentionPattern!.match(/F/g) ?? []).length).toBeLessThan(gemma.layers);
  });
});

describe('the weights are the tiebreaker', () => {
  const real = loadSiteData();
  // data/tensor-shapes.json is measured from each model's own safetensors headers
  // by scripts/audit-shapes.ts, over HTTP range requests -- ~25KB per shard, no
  // weights downloaded. Every architecture number on this site is otherwise a
  // human reading a config.json, and four places have been found where that
  // reading was wrong, one by a factor of 25. The tensors settle it.
  const measured: Record<string, {
    kvBytesPerToken?: number;
    kvWindowedBytesPerToken?: number;
    width?: number;
    embedRows?: number;
    perLayerKvBytes?: Record<string, number>;
    widestDenseLayer?: number;
  }> = JSON.parse(readFileSync('data/tensor-shapes.json', 'utf8')).models;

  it('measures every model in the catalogue', () => {
    for (const m of real.models) expect(measured[m.id], `${m.id} unmeasured`).toBeTruthy();
  });

  it('every stored KV, width and vocabulary figure matches its own weights', () => {
    let checked = 0;
    for (const m of real.models) {
      const w = measured[m.id];
      if (!w) continue;
      if (w.kvBytesPerToken !== undefined) {
        expect(m.kvBytesPerToken, `${m.id} kvBytesPerToken`).toBe(w.kvBytesPerToken);
        checked++;
      }
      if (w.kvWindowedBytesPerToken !== undefined) {
        expect(m.kvWindowedBytesPerToken ?? 0, `${m.id} windowed`).toBe(w.kvWindowedBytesPerToken);
      }
      if (w.width !== undefined) expect(m.width, `${m.id} width`).toBe(w.width);
      if (w.embedRows !== undefined) expect(m.vocabSize, `${m.id} vocabSize`).toBe(w.embedRows);
    }
    expect(checked).toBeGreaterThan(35);
  });

  it('REGRESSION: Gemma 4 E2B is double-wide on its KV-shared layers', () => {
    // use_double_wide_mlp doubles the FFN on the layers that share KV, so the
    // widest tensor a micro-batch flows through is not layer 0's. The weights
    // say so plainly: gate_proj is [6144, 1536] at layer 0 and [12288, 1536] at
    // layer 15. width was stored as the config's intermediate_size, 6144.
    const m = real.models.find((x) => x.id === 'gemma-4-e2b')!;
    expect(m.width).toBe(12288);
    expect(measured['gemma-4-e2b'].widestDenseLayer).toBeGreaterThan(0);
    // E4B sets use_double_wide_mlp false, so it must NOT be doubled.
    expect(real.models.find((x) => x.id === 'gemma-4-e4b')!.width).toBe(10240);
  });

  it('REGRESSION: Mistral Small 4 has no dense FFN to be wide', () => {
    // Its config carries intermediate_size 12288, which width was taken from,
    // but no layer instantiates a dense MLP -- every one is MoE. The widest
    // path is moe_intermediate_size 2048 across 4 experts per token.
    const m = real.models.find((x) => x.id === 'mistral-small-4')!;
    expect(m.width).toBe(8192);
    expect(m.expertsPerToken).toBe(4);
  });

  it('does not count a multi-token-prediction head as part of the stack', () => {
    // Eight checkpoints ship an MTP head named mtp.layers.0.*: its own layer 0,
    // numbered from zero, next to the main stack's. Keying off `layers.(\d+)`
    // anywhere in a tensor name merges the two and makes a full-attention block
    // appear on a linear-attention layer -- which is exactly what a first pass
    // of the audit reported, for eight models that are fine.
    for (const id of ['qwen3.6-27b', 'qwen3.5-397b-a17b', 'ornith-1.5-35b-a3b']) {
      const m = real.models.find((x) => x.id === id)!;
      // layer 0 of the MAIN stack is linear attention and caches nothing
      expect(m.attentionPattern![0]).toBe('.');
      expect(measured[id].perLayerKvBytes!['.']).toBe(0);
      expect(m.kvBytesPerToken).toBe(
        [...m.attentionPattern!].filter((c) => c === 'F').length * measured[id].perLayerKvBytes!.F,
      );
    }
  });
});

describe('the block diagram draws only what it can place', () => {
  const real = loadSiteData();
  const shapes: Record<string, any> = JSON.parse(readFileSync('data/tensor-shapes.json', 'utf8')).models;

  it('every model either has a drawable block or is explicitly marked undrawable', () => {
    // No silent blanks: a sheet with neither is a rendering hole nobody notices.
    for (const m of real.models) {
      const blocks = shapes[m.id]?.blocks;
      expect(blocks, `${m.id} has no blocks`).toBeTruthy();
      expect(Object.keys(blocks).length, `${m.id} has no block kinds`).toBeGreaterThan(0);
    }
  });

  it('input_layernorm is what makes a block drawable', () => {
    // It is the one name whose position is never in doubt, so it anchors the
    // rest. Nemotron names its per-layer norm plain `norm`, which could be the
    // pre-attention or the pre-FFN one -- so its blocks are not drawn at all
    // rather than drawn with a guess.
    for (const [id, e] of Object.entries(shapes)) {
      for (const [kind, b] of Object.entries((e as any).blocks ?? {}) as [string, any][]) {
        expect(b.drawable, `${id} [${kind}]`).toBe(b.norms.includes('input_layernorm'));
      }
    }
    for (const id of ['nemotron-3-nano-30b-a3b', 'nemotron-3-super-120b-a12b']) {
      expect(Object.values(shapes[id].blocks).every((b: any) => !b.drawable), id).toBe(true);
      expect(Object.values(shapes[id].blocks).every((b: any) => b.unknown.includes('norm')), id).toBe(true);
    }
  });

  it('post_attention_layernorm is placed by whether a pre-FFN norm exists, not by its name', () => {
    // The trap the whole fail-closed rule exists for. mlx-lm's own blocks:
    //   llama   h = x + self_attn(input_layernorm(x));  out = h + mlp(post_attention_layernorm(h))
    //   gemma3  h = x + post_attention_layernorm(self_attn(input_layernorm(x)))
    //           out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
    // Same tensor name, one position inside the residual and one after it. What
    // separates them is the presence of pre_feedforward_layernorm.
    const sandwich = (id: string) =>
      Object.values(shapes[id].blocks as Record<string, any>).some((b) =>
        b.norms.includes('pre_feedforward_layernorm'),
      );
    for (const id of ['gemma-3-12b', 'gemma-3-27b', 'gemma-3-4b']) expect(sandwich(id), id).toBe(true);
    for (const id of ['llama-3.1-8b', 'qwen3-8b', 'phi-4', 'mistral-small-4']) expect(sandwich(id), id).toBe(false);
    // A sandwich model must carry all four norms, never a partial set.
    for (const [id, e] of Object.entries(shapes)) {
      for (const b of Object.values((e as any).blocks ?? {}) as any[]) {
        if (b.norms.includes('pre_feedforward_layernorm')) {
          expect(b.norms, `${id}`).toContain('post_feedforward_layernorm');
          expect(b.norms, `${id}`).toContain('post_attention_layernorm');
        }
      }
    }
  });

  it('REGRESSION: gpt-oss runs four experts of 2,880, not one tensor of 2,880', () => {
    // width was stored as hidden_size. The expert weights are MXFP4-packed and
    // unreadable by shape, but the bias beside them is plain BF16:
    // gate_up_proj_bias is [32, 5760] -- 32 experts, gate and up fused, so the
    // FFN is 2,880 wide and four of them run per token.
    for (const id of ['gpt-oss-20b', 'gpt-oss-120b']) {
      const m = real.models.find((x) => x.id === id)!;
      expect(m.width, id).toBe(11520);
      expect(m.expertsPerToken, id).toBe(4);
      expect(shapes[id].moeFfn, id).toBe(2880);
      expect(shapes[id].width, id).toBe(11520);
    }
  });

  it('names a component rather than drawing it when its place is not fixed', () => {
    // Attention sinks and Gemma's per-layer-embedding tensors are real and are
    // named on the sheet, but nothing claims to know where they sit.
    expect(shapes['gpt-oss-20b'].blocks.F.unknown).toContain('sinks');
    expect(shapes['gemma-4-e4b'].blocks.F.unknown).toContain('per_layer_projection');
    // A bias belongs to the projection it hangs off, whose place IS fixed, so
    // it must not be reported as unplaced.
    for (const e of Object.values(shapes) as any[]) {
      for (const b of Object.values(e.blocks ?? {}) as any[]) {
        expect(b.unknown).not.toContain('bias');
      }
    }
  });
});
