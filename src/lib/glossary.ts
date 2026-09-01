import type { GlossaryTerm } from './schemas';

/**
 * The glossary entry that documents a given quant string. One entry covers a
 * whole family — Q4_K_S/M/L are one idea at three sizes — so this is a lookup
 * through each entry's `quants` list rather than a match on the term name.
 */
export function termForQuant(
  glossary: GlossaryTerm[],
  quant: string,
): GlossaryTerm | undefined {
  return glossary.find((t) => t.quants.includes(quant));
}

/** Every quant string the glossary claims, for the completeness gate. */
export function coveredQuants(glossary: GlossaryTerm[]): Set<string> {
  return new Set(glossary.flatMap((t) => t.quants));
}

/** Quants claimed by more than one entry — an authoring mistake, not a feature. */
export function duplicateQuants(glossary: GlossaryTerm[]): string[] {
  const seen = new Map<string, number>();
  for (const q of glossary.flatMap((t) => t.quants)) seen.set(q, (seen.get(q) ?? 0) + 1);
  return [...seen.entries()].filter(([, n]) => n > 1).map(([q]) => q);
}

/**
 * The benchmark entry a provenance `metric` refers to. Metric strings are free
 * text written per-cell — "AIME'24 (thinking mode)", "HHEM hallucination rate %
 * (lower is better)" — so we match the entry's name or any alias as a prefix,
 * longest first so "MATH-500" wins over "MATH".
 *
 * Returns undefined rather than throwing: an unmatched metric must render as
 * plain text, never as a broken link.
 */
export function termForMetric(
  glossary: GlossaryTerm[],
  metric: string,
): GlossaryTerm | undefined {
  const needle = normalize(metric);
  const candidates = glossary
    .filter((t) => t.category === 'benchmark')
    // A term may name several benchmarks — "HumanEval / EvalPlus / MBPP" — so
    // each side of the slash is matchable on its own. Short fragments ("Pro"
    // out of "SWE-bench Verified / Pro") are dropped as too generic.
    .flatMap((t) =>
      [...t.term.split('/'), ...t.aliases]
        .map((name) => normalize(name))
        .filter((key) => key.length >= 4)
        .map((key) => ({ term: t, key })),
    )
    .sort((a, b) => b.key.length - a.key.length);
  return candidates.find(({ key }) => needle.startsWith(key))?.term;
}

/** Lowercase, strip punctuation that varies between spellings of the same name. */
function normalize(s: string): string {
  return s.toLowerCase().replace(/[’'`]/g, '').replace(/[\s\-_]+/g, ' ').trim();
}
