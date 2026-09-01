import { describe, expect, it } from 'vitest';
import { checkedLabel, daysBetween, isStale, newestRetrieved } from '../src/lib/aggregate';
import type { Provenance, ScoreEntry } from '../src/lib/schemas';

const prov = (retrieved: string): Provenance => ({
  sourceId: 'livebench',
  metric: 'x',
  retrieved,
  method: 'aggregated',
});

describe('aggregate', () => {
  it('finds the newest retrieved date', () => {
    expect(newestRetrieved([prov('2026-01-01'), prov('2026-06-15'), prov('2026-03-01')])).toBe('2026-06-15');
    expect(newestRetrieved([])).toBeNull();
  });

  it('computes day differences across months', () => {
    expect(daysBetween('2026-01-31', '2026-02-01')).toBe(1);
    expect(daysBetween('2026-01-01', '2026-03-02')).toBe(60);
  });

  it('flags stale entries past the threshold only', () => {
    const entry = (d: string): ScoreEntry => ({ band: 'mid', dimension: 'math', score: 5, provenance: [prov(d)] });
    expect(isStale(entry('2026-07-15'), '2026-08-30')).toBe(false); // 46 days
    expect(isStale(entry('2026-06-01'), '2026-08-30')).toBe(true); // 90 days
  });

  it('renders a human badge label', () => {
    expect(checkedLabel([prov('2026-08-30')])).toBe('data checked Aug 2026');
    expect(checkedLabel([])).toBe('unverified');
  });
});
