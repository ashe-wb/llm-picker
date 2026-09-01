import { describe, expect, it } from 'vitest';
import { loadSiteData } from '../src/lib/data';
import {
  coveredQuants,
  duplicateQuants,
  termForMetric,
  termForQuant,
} from '../src/lib/glossary';
import { QUANT_VOCAB } from '../src/lib/schemas';

const { glossary, scores, sources } = loadSiteData();

describe('glossary coverage', () => {
  it('explains every quant string in the vocabulary', () => {
    const covered = coveredQuants(glossary);
    const missing = Object.keys(QUANT_VOCAB).filter((q) => !covered.has(q));
    expect(missing).toEqual([]);
  });

  it('resolves every vocabulary quant to exactly one entry', () => {
    expect(duplicateQuants(glossary)).toEqual([]);
    for (const quant of Object.keys(QUANT_VOCAB)) {
      expect(termForQuant(glossary, quant), quant).toBeDefined();
    }
  });

  it('claims no quant that is not in the vocabulary', () => {
    const stray = [...coveredQuants(glossary)].filter((q) => !(q in QUANT_VOCAB));
    expect(stray).toEqual([]);
  });

  it('has unique ids and resolvable seeAlso references', () => {
    const ids = glossary.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
    const unknown = glossary.flatMap((t) =>
      t.seeAlso.filter((r) => !ids.includes(r)).map((r) => `${t.id} -> ${r}`),
    );
    expect(unknown).toEqual([]);
  });

  it('distinguishes MXFP4 from NVFP4 — the confusion that prompted the page', () => {
    const entry = termForQuant(glossary, 'mxfp4');
    expect(entry).toBeDefined();
    expect(termForQuant(glossary, 'nvfp4')?.id).toBe(entry!.id);
    expect(entry!.newbieExplainer).toMatch(/microscaling/i);
    expect(entry!.newbieExplainer).toMatch(/NVIDIA/);
  });
});

describe('termForMetric', () => {
  it('matches the awkward metric strings the dataset actually contains', () => {
    const cases: [string, string][] = [
      ["AIME'24 (thinking mode)", 'aime'],
      ['AIME-120 (temperature 0.6)', 'aime'],
      ['HHEM hallucination rate % (lower is better)', 'hhem'],
      ['LiveCodeBench v6', 'livecodebench'],
      ['SWE-bench Verified', 'swe-bench'],
      ['BFCL v3 (thinking)', 'bfcl'],
      ['MRCR v2 @128K (8 needle)', 'mrcr'],
      ['Tau2 average (with tools)', 'tau-bench'],
      ['IFEval strict prompt (thinking)', 'ifeval'],
      ['HumanEval (0-shot)', 'humaneval'],
      ['MATH (4-shot)', 'math-bench'],
    ];
    for (const [metric, id] of cases) {
      expect(termForMetric(glossary, metric)?.id, metric).toBe(id);
    }
  });

  it('prefers the longest matching name', () => {
    // "MATH-500" must not be swallowed by the shorter "MATH".
    expect(termForMetric(glossary, 'MATH-500 (thinking mode)')?.id).toBe('math-bench');
  });

  it('returns undefined rather than guessing', () => {
    expect(termForMetric(glossary, 'community consensus')).toBeUndefined();
    expect(termForMetric(glossary, 'degradation reports')).toBeUndefined();
  });

  it('covers every benchmark metric whose numbers this site republishes', () => {
    const linkOnly = new Set(sources.filter((s) => !s.republishOk).map((s) => s.id));
    const skip = new Set(['community-consensus', 'quant-degradation-community']);
    const unmatched = new Set<string>();
    for (const [, file] of scores) {
      for (const entry of [...file.entries, ...file.quantOverrides]) {
        for (const p of entry.provenance) {
          if (skip.has(p.sourceId) || linkOnly.has(p.sourceId)) continue;
          if (!termForMetric(glossary, p.metric)) unmatched.add(p.metric);
        }
      }
    }
    expect([...unmatched]).toEqual([]);
  });
});
