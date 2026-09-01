import type { Provenance, ScoreEntry } from './schemas';

export const STALE_AFTER_DAYS = 60;

/** Most recent `retrieved` date across provenance entries (ISO yyyy-mm-dd). */
export function newestRetrieved(provenance: Provenance[]): string | null {
  if (provenance.length === 0) return null;
  return provenance.map((p) => p.retrieved).sort().at(-1) ?? null;
}

export function daysBetween(isoA: string, isoB: string): number {
  const ms = Date.parse(`${isoB}T00:00:00Z`) - Date.parse(`${isoA}T00:00:00Z`);
  return Math.round(ms / 86_400_000);
}

/** True when the newest evidence behind a score is older than the threshold. */
export function isStale(entry: ScoreEntry, today: string, staleDays = STALE_AFTER_DAYS): boolean {
  const newest = newestRetrieved(entry.provenance);
  if (!newest) return true;
  return daysBetween(newest, today) > staleDays;
}

/** "data checked Aug 2026" — badge text from the newest provenance date. */
export function checkedLabel(provenance: Provenance[]): string {
  const newest = newestRetrieved(provenance);
  if (!newest) return 'unverified';
  const d = new Date(`${newest}T00:00:00Z`);
  const month = d.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' });
  return `data checked ${month} ${d.getUTCFullYear()}`;
}

export function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}
