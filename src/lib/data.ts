import fs from 'node:fs';
import path from 'node:path';
import {
  DimensionSchema,
  MitigationSchema,
  ModelSchema,
  ScoresFileSchema,
  SiteMetaSchema,
  SourceSchema,
  TaskSchema,
  GlossaryTermSchema,
  HardwareSchema,
  WeaknessesFileSchema,
  type Dimension,
  type Mitigation,
  type Model,
  type ScoresFile,
  type SiteMeta,
  type Source,
  type Task,
  type GlossaryTerm,
  type Hardware,
  type WeaknessesFile,
} from './schemas';
import { z } from 'zod';

export interface SiteData {
  hardware: Hardware;
  glossary: GlossaryTerm[];
  dimensions: Dimension[];
  tasks: Task[];
  mitigations: Mitigation[];
  sources: Source[];
  models: Model[];
  scores: Map<string, ScoresFile>;
  weaknesses: Map<string, WeaknessesFile>;
  siteMeta: SiteMeta;
}

const DATA_DIR = path.resolve(process.cwd(), 'data');

function readJson(rel: string): unknown {
  return JSON.parse(fs.readFileSync(path.join(DATA_DIR, rel), 'utf8'));
}

function parseFile<T>(schema: z.ZodType<T>, rel: string): T {
  const result = schema.safeParse(readJson(rel));
  if (!result.success) {
    const issues = result.error.issues
      .map((i) => `  data/${rel} :: ${i.path.join('.')} — ${i.message}`)
      .join('\n');
    throw new Error(`Schema violation in data/${rel}:\n${issues}`);
  }
  return result.data;
}

function listJsonFiles(subdir: string): string[] {
  const dir = path.join(DATA_DIR, subdir);
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith('.json'))
    .sort()
    .map((f) => path.join(subdir, f));
}

let cache: SiteData | null = null;

/**
 * Loads and zod-parses every data file. Throws on any schema violation, which
 * makes `astro build` (and anything else importing this) a hard quality gate.
 */
export function loadSiteData(): SiteData {
  if (cache) return cache;

  const scores = new Map<string, ScoresFile>();
  for (const f of listJsonFiles('scores')) {
    const parsed = parseFile(ScoresFileSchema, f);
    scores.set(parsed.modelId, parsed);
  }
  const weaknesses = new Map<string, WeaknessesFile>();
  for (const f of listJsonFiles('weaknesses')) {
    const parsed = parseFile(WeaknessesFileSchema, f);
    weaknesses.set(parsed.modelId, parsed);
  }

  cache = {
    hardware: parseFile(HardwareSchema, 'hardware.json'),
    glossary: parseFile(z.array(GlossaryTermSchema), 'glossary.json'),
    dimensions: parseFile(z.array(DimensionSchema), 'dimensions.json'),
    tasks: parseFile(z.array(TaskSchema), 'tasks.json'),
    mitigations: parseFile(z.array(MitigationSchema), 'mitigations.json'),
    sources: parseFile(z.array(SourceSchema), 'sources.json'),
    models: listJsonFiles('models').map((f) => parseFile(ModelSchema, f)),
    scores,
    weaknesses,
    siteMeta: parseFile(SiteMetaSchema, 'site-meta.json'),
  };
  return cache;
}

/** Test seam: clear the module cache (used by validate.ts watch mode / tests). */
export function clearDataCache(): void {
  cache = null;
}
