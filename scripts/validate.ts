/**
 * Data quality gate. Zod-parses every file in /data (via loadSiteData) and
 * enforces cross-file rules zod can't see. Exits non-zero on any error.
 * Staleness is reported but does not fail the build.
 */
import { loadSiteData } from '../src/lib/data';
import { bandOfQuant, QUANT_VOCAB } from '../src/lib/schemas';
import { coveredQuants, duplicateQuants, termForMetric } from '../src/lib/glossary';
import { newestRetrieved, daysBetween, todayIso, STALE_AFTER_DAYS } from '../src/lib/aggregate';
import { readFileSync } from 'node:fs';

const errors: string[] = [];
const warnings: string[] = [];

let data;
try {
  data = loadSiteData();
} catch (e) {
  console.error(String(e instanceof Error ? e.message : e));
  process.exit(1);
}

const modelIds = new Set(data.models.map((m) => m.id));
const sourceById = new Map(data.sources.map((s) => [s.id, s]));
const mitigationIds = new Set(data.mitigations.map((m) => m.id));
const dimById = new Map(data.dimensions.map((d) => [d.id, d]));

// --- hardware: every real ceiling names itself ---
// The card prints "OF YOUR 27GB (75% of 36GB, the macOS GPU cap)". Without a
// capLabel the parenthetical loses the half that says *who* imposed the number.
for (const platform of data.hardware.platforms) {
  const { kind, capLabel } = platform.ceiling;
  if (kind !== 'ram-only' && !capLabel) {
    errors.push(`hardware.json :: ${platform.id} — ceiling.kind '${kind}' needs a capLabel`);
  }
}

// --- hardware: every preset needs a bandwidth, every platform a prefill class ---
// A machine with no bandwidth silently loses its speed estimate, which is the
// one thing that separates it from the other machines of the same capacity.
const chipIds = new Set(data.hardware.chips.map((c) => c.id));
const systemMemoryIds = new Set(data.hardware.systemMemory.map((m) => m.id));
for (const preset of data.hardware.presets) {
  // One or the other: a discrete card names its own bandwidth, a Mac names a
  // chip. A preset with neither loses its speed estimate silently.
  // A CPU-only machine names neither: with no graphics card its system memory
  // IS its bandwidth, so systemMemoryId is where the figure comes from.
  if (!preset.memoryBandwidthGbs && !preset.chipId && !preset.systemMemoryId) {
    errors.push(
      `hardware.json :: ${preset.id} — preset needs memoryBandwidthGbs, chipId or systemMemoryId`,
    );
  }
  if (preset.chipId && !chipIds.has(preset.chipId)) {
    errors.push(`hardware.json :: ${preset.id} — unknown chipId '${preset.chipId}'`);
  }
  if (preset.systemMemoryId && !systemMemoryIds.has(preset.systemMemoryId)) {
    errors.push(
      `hardware.json :: ${preset.id} — unknown systemMemoryId '${preset.systemMemoryId}'`,
    );
  }
  // A capacity the preset row offers but the VRAM ladder lacks vanishes the
  // moment the reader edits any control: assigning an absent value to a select
  // silently does nothing and the clamp lands somewhere else. Exactly how an
  // M5 Max 128GB became a 48GB machine, one tier down.
  if (preset.vramGb && !data.hardware.vramOptions.includes(preset.vramGb)) {
    errors.push(
      `hardware.json :: ${preset.id} — ${preset.vramGb}GB is not in vramOptions, so it is unreachable`,
    );
  }
  // Any machine that can spill needs a speed for the half the CPU runs, or its
  // partial fits lose their tok/s estimate silently — the exact failure the
  // bandwidth rule above exists to prevent, on the other tier of memory.
  if ((preset.platform === 'discrete' || preset.platform === 'cpu') && !preset.systemMemoryId) {
    errors.push(`hardware.json :: ${preset.id} — preset can offload, so it needs a systemMemoryId`);
  }
  const chip = data.hardware.chips.find((c) => c.id === preset.chipId);
  if (chip && !chip.ramOptions.includes(preset.ramGb)) {
    errors.push(
      `hardware.json :: ${preset.id} — ${chip.label} was never sold with ${preset.ramGb}GB (has ${chip.ramOptions.join(', ')})`,
    );
  }
}
const ladder = new Set(data.hardware.ramOptions);
const seenChips = new Set<string>();
for (const chip of data.hardware.chips) {
  if (seenChips.has(chip.id)) errors.push(`hardware.json :: duplicate chip id '${chip.id}'`);
  seenChips.add(chip.id);
  // A capacity a chip offers but the global ladder lacks would vanish from the
  // dropdown the moment no chip is selected.
  for (const g of chip.ramOptions) {
    if (!ladder.has(g)) {
      errors.push(`hardware.json :: ${chip.id} — ${g}GB is not in ramOptions, so it is unreachable`);
    }
  }
}
for (const platform of data.hardware.platforms) {
  // A raise range is a claim about what someone else's machine will tolerate,
  // and this site's own rule is that claims about the world carry citations.
  // The figure used to have nowhere to put one, which is how it shipped for
  // months labelled "safe max" with nothing behind it.
  const raise = platform.ceiling.raise;
  if (raise && !raise.citation) {
    errors.push(
      `hardware.json :: ${platform.id} — ceiling.raise needs a citation, or it is an unsourced claim about what is survivable`,
    );
  }
  if (raise?.publishedReserveGb) {
    for (const [gb, reserve] of Object.entries(raise.publishedReserveGb)) {
      if (reserve >= Number(gb)) {
        errors.push(`hardware.json :: ${platform.id} — published reserve ${reserve}GB is not less than ${gb}GB`);
      }
    }
  }
  if (!platform.prefillClass) {
    errors.push(`hardware.json :: ${platform.id} — platform needs prefillClass`);
  }
}

// --- the engine derate is a claim about the world, so it carries a source ---
if (!data.hardware.perTokenLatency.citation) {
  errors.push('hardware.json :: perTokenLatency needs a citation');
}
const ptl = data.hardware.perTokenLatency;
for (const [name, v] of [['default', ptl], ...Object.entries(ptl.byPlatform ?? {})] as [string, {loMs:number;hiMs:number}][]) {
  if (v.loMs >= v.hiMs) errors.push(`hardware.json :: perTokenLatency ${name} loMs must be below hiMs`);
}

// --- and both must agree with the weights ---
// data/tensor-shapes.json is measured from each model's own safetensors headers
// by scripts/audit-shapes.ts. A config.json is a description of a model read by
// a person; the tensors are the model. Where they disagree, the tensors win, and
// this is the check that says so out loud.
//
// It compares shape only. Whether a tensor is USED is decided by the config and
// the implementation -- eight of these checkpoints carry a multi-token-prediction
// head that llama.cpp never runs -- so nothing here infers a layer's behaviour
// from a tensor's existence.
type Measured = {
  kvBytesPerToken?: number;
  kvWindowedBytesPerToken?: number;
  width?: number;
  embedRows?: number;
};
let measured: Record<string, Measured> = {};
try {
  measured = JSON.parse(readFileSync('data/tensor-shapes.json', 'utf8')).models ?? {};
} catch {
  warnings.push('data/tensor-shapes.json missing — run `npx tsx scripts/audit-shapes.ts`');
}
for (const model of data.models) {
  const m = measured[model.id];
  if (!m) {
    warnings.push(`models/${model.id}.json — no measured shapes; run scripts/audit-shapes.ts`);
    continue;
  }
  const cmp: [string, number | undefined, number | undefined][] = [
    ['kvBytesPerToken', model.kvBytesPerToken, m.kvBytesPerToken],
    ['kvWindowedBytesPerToken', model.kvWindowedBytesPerToken ?? 0, m.kvWindowedBytesPerToken],
    ['width', model.width, m.width],
    ['vocabSize', model.vocabSize, m.embedRows],
  ];
  for (const [field, stored, weights] of cmp) {
    if (weights === undefined) continue;
    if ((stored ?? 0) !== weights) {
      errors.push(
        `models/${model.id}.json — ${field} is ${stored}, its weights say ${weights}`,
      );
    }
  }
}

// --- the diagram and the memory model must agree about what caches ---
// Two independent derivations: the per-layer attention pattern a diagram draws,
// and the kvBytesPerToken the fit model spends. If they disagree one is wrong,
// which is how Nanbeige4.2 3B's doubled cache figure was found.
for (const model of data.models) {
  const { attentionPattern: pat, kvHeads, headDim, kvBytesPerToken: kv, layers, mlaLatentDim } = model;
  // MLA caches one compressed latent per layer, not per-head keys and values.
  // This branch used to SKIP such models, which is how Mistral Small 4 shipped
  // with a full-MHA figure 25x too large. Skipping is not checking.
  if (mlaLatentDim) {
    if (kv !== layers * mlaLatentDim * 2) {
      errors.push(
        `models/${model.id}.json — MLA: kvBytesPerToken should be ${layers * mlaLatentDim * 2}, is ${kv}`,
      );
    }
    continue;
  }
  if (!pat || !kvHeads || !headDim || !kv) continue;
  if (pat.length !== layers) {
    errors.push(`models/${model.id}.json — attentionPattern is ${pat.length} long, layers is ${layers}`);
    continue;
  }
  if ((model.globalKvHeads === undefined) !== (model.globalHeadDim === undefined)) {
    errors.push(`models/${model.id}.json — globalKvHeads and globalHeadDim must be set together`);
    continue;
  }
  // Trailing KV-shared layers reuse an earlier layer's cache and allocate none
  // of their own, so they are not part of either count.
  const caching = pat.slice(0, layers - (model.kvSharedLayers ?? 0));
  const count = (c: string) => [...caching].filter((x) => x === c).length;
  // K and V, per caching layer, per KV head, per head dimension, at 2 bytes.
  // Full-attention layers may have a geometry of their own -- Gemma 4 gives
  // them a 512-wide head over as little as one KV head, against 256 over eight
  // on the sliding layers. Modelling the two with one geometry is what left the
  // whole Gemma 4 family charged 2-4x its real cache.
  const bytes = (n: number, h: number, d: number) => 4 * n * h * d;
  const wantFull = bytes(count('F'), model.globalKvHeads ?? kvHeads, model.globalHeadDim ?? headDim);
  if (kv !== wantFull) {
    errors.push(
      `models/${model.id}.json — ${count('F')} caching full-attention layers imply kvBytesPerToken ${wantFull}, is ${kv}`,
    );
  }
  const win = model.kvWindowedBytesPerToken;
  const wantWin = bytes(count('S'), kvHeads, headDim);
  if ((win ?? 0) !== wantWin) {
    errors.push(
      `models/${model.id}.json — ${count('S')} caching sliding layers imply kvWindowedBytesPerToken ${wantWin || 'none'}, is ${win ?? 'none'}`,
    );
  }
  if (wantWin > 0 && !model.slidingWindow) {
    errors.push(`models/${model.id}.json — has sliding layers but no slidingWindow`);
  }
}

// --- models: every model needs a layer count ---
// It is what turns a partial fit into the `-ngl` number the reader types. A
// model without one still gets an estimate, but the card falls back to prose
// and the reader is left to convert a percentage themselves.
for (const model of data.models) {
  if (!model.layers) {
    errors.push(`models/${model.id}.json — needs layers (num_hidden_layers from its config.json)`);
  }
  // Without these the runtime overhead cannot be derived, and fitFor returns
  // null rather than guessing — so the model silently disappears from the site.
  if (!model.vocabSize) {
    errors.push(`models/${model.id}.json — needs vocabSize (vocab_size from its config.json)`);
  }
  if (!model.width) {
    errors.push(
      `models/${model.id}.json — needs width (max of hidden_size, intermediate_size, moe_intermediate_size x num_experts_per_tok)`,
    );
  }
  if (!model.attentionHeads) {
    errors.push(
      `models/${model.id}.json — needs attentionHeads (num_attention_heads from its config.json)`,
    );
  }
  // A width below the hidden size means the max() was taken over the wrong
  // fields — the MoE branch is easy to miss and silently understates scratch.
  if (model.width && model.vocabSize && model.width > model.vocabSize) {
    errors.push(`models/${model.id}.json — width ${model.width} exceeds vocabSize, check the source fields`);
  }
}

// --- models: an MoE must say how much of itself it activates ---
// Without it, models/[id].astro renders the literal string "~undefinedB active",
// and picker.ts silently falls back to total params for the speed tiebreak —
// ranking a 397B-A17B as if all 397B were doing work.
for (const model of data.models) {
  if (model.architecture === 'moe' && model.activeParamsB === undefined) {
    errors.push(`models/${model.id}.json — architecture 'moe' requires activeParamsB`);
  }
}

// --- tasks: weights sum to 1.0 ---
for (const task of data.tasks) {
  const sum = Object.values(task.weights).reduce((a, b) => a + (b ?? 0), 0);
  if (Math.abs(sum - 1) > 1e-9) {
    errors.push(`tasks.json :: ${task.id} — weights sum to ${sum}, expected 1.0`);
  }
}

// --- scores ---
for (const [modelId, file] of data.scores) {
  const where = `scores/${modelId}.json`;
  if (!modelIds.has(modelId)) errors.push(`${where} — modelId '${modelId}' has no matching model file`);
  const model = data.models.find((m) => m.id === modelId);

  const checkProvenance = (label: string, provenance: typeof file.entries[number]['provenance']) => {
    for (const p of provenance) {
      const source = sourceById.get(p.sourceId);
      if (!source) {
        errors.push(`${where} :: ${label} — unknown sourceId '${p.sourceId}'`);
        continue;
      }
      if (!source.republishOk && p.rawValue !== undefined) {
        errors.push(
          `${where} :: ${label} — source '${p.sourceId}' does not permit republication; rawValue must be removed (link-only citations allowed)`,
        );
      }
      const newest = p.retrieved;
      if (daysBetween(newest, todayIso()) > STALE_AFTER_DAYS) {
        warnings.push(`${where} :: ${label} — provenance from ${p.sourceId} retrieved ${newest} is stale (> ${STALE_AFTER_DAYS} days)`);
      }
    }
  };

  for (const entry of file.entries) {
    const label = `${entry.band}/${entry.dimension}`;
    if (model && !model.bands[entry.band]) {
      errors.push(`${where} :: ${label} — model does not declare band '${entry.band}'`);
    }
    checkProvenance(label, entry.provenance);
  }
  for (const o of file.quantOverrides) {
    const label = `override ${o.quant}/${o.dimension}`;
    const band = bandOfQuant(o.quant);
    if (model && band && !model.bands[band]) {
      errors.push(`${where} :: ${label} — quant '${o.quant}' maps to band '${band}' which the model does not declare`);
    }
    checkProvenance(label, o.provenance);
  }

  // Editorial completeness: a bad low-band score on a quant-sensitive dimension
  // must be explained by a weakness entry — the site's whole point.
  const weakFile = data.weaknesses.get(modelId);
  for (const entry of file.entries) {
    if (entry.band !== 'low' || entry.score >= 5) continue;
    if (!dimById.get(entry.dimension)?.quantSensitive) continue;
    const covered = (weakFile?.entries ?? []).some(
      (w) => w.bands.includes('low') && w.dimensions.includes(entry.dimension),
    );
    if (!covered) {
      errors.push(
        `${where} :: low/${entry.dimension} scored ${entry.score} but weaknesses/${modelId}.json has no matching low-band entry — document the weakness (and its mitigation)`,
      );
    }
  }
}

// --- weaknesses ---
const CITATION_MINIMUM = { 'community-consensus': 2, anecdotal: 1, 'owner-verified': 0 } as const;
for (const [modelId, file] of data.weaknesses) {
  const where = `weaknesses/${modelId}.json`;
  if (!modelIds.has(modelId)) errors.push(`${where} — modelId '${modelId}' has no matching model file`);
  for (const entry of file.entries) {
    for (const m of entry.mitigations) {
      if (!mitigationIds.has(m.mitigationId)) {
        errors.push(`${where} :: ${entry.id} — unknown mitigationId '${m.mitigationId}'`);
      }
    }
    const min = CITATION_MINIMUM[entry.confidence];
    if (entry.citations.length < min) {
      errors.push(
        `${where} :: ${entry.id} — confidence '${entry.confidence}' requires ≥${min} citation(s), found ${entry.citations.length}`,
      );
    }
  }
}

// --- glossary: every quant in the vocabulary must be explained ---
// This is what stops the page rotting: adding a quant to QUANT_VOCAB without
// writing a definition fails the build.
{
  const covered = coveredQuants(data.glossary);
  for (const quant of Object.keys(QUANT_VOCAB)) {
    if (!covered.has(quant)) {
      errors.push(
        `glossary.json — quant '${quant}' is in QUANT_VOCAB but no glossary entry covers it; add it to an entry's 'quants' list`,
      );
    }
  }
  for (const quant of duplicateQuants(data.glossary)) {
    errors.push(`glossary.json — quant '${quant}' is claimed by more than one entry`);
  }
  const ids = new Set<string>();
  for (const t of data.glossary) {
    if (ids.has(t.id)) errors.push(`glossary.json — duplicate id '${t.id}'`);
    ids.add(t.id);
  }
  for (const t of data.glossary) {
    for (const ref of t.seeAlso) {
      if (!ids.has(ref)) errors.push(`glossary.json :: ${t.id} — seeAlso '${ref}' matches no entry`);
    }
  }
  // Benchmark names are free text, so this can only ever be a nudge.
  const unmatched = new Set<string>();
  for (const [, file] of data.scores) {
    for (const entry of [...file.entries, ...file.quantOverrides]) {
      for (const p of entry.provenance) {
        if (p.sourceId === 'community-consensus' || p.sourceId === 'quant-degradation-community') continue;
        // Link-only aggregators publish proprietary indices we may not restate.
        if (sourceById.get(p.sourceId)?.republishOk === false) continue;
        if (!termForMetric(data.glossary, p.metric)) unmatched.add(p.metric);
      }
    }
  }
  if (unmatched.size > 0) {
    warnings.push(
      `glossary.json — ${unmatched.size} provenance metric(s) match no benchmark entry: ${[...unmatched].slice(0, 5).join(' · ')}${unmatched.size > 5 ? ' …' : ''}`,
    );
  }
}

// --- models without any scores (can never be recommended) ---
for (const model of data.models) {
  if (!data.scores.has(model.id)) warnings.push(`models/${model.id}.json — no scores file; model can never be recommended`);
}

// --- report ---
for (const w of warnings) console.warn(`⚠ ${w}`);
if (errors.length > 0) {
  for (const e of errors) console.error(`✗ ${e}`);
  console.error(`\n${errors.length} error(s), ${warnings.length} warning(s).`);
  process.exit(1);
}
console.log(`✓ data valid — ${data.models.length} models, ${warnings.length} warning(s).`);
