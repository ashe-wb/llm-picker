/**
 * Measure each model's architecture from its own weights, and write the result
 * to data/tensor-shapes.json for validate.ts and the tests to check against.
 *
 * Every architectural number on this site is read out of a config.json by a
 * human, and four separate places have been found where that reading was
 * wrong -- one of them by a factor of 25. A config is a description of a model.
 * The weights ARE the model, so they settle any disagreement about shape.
 *
 * It does not download them. A safetensors file opens with an 8-byte
 * little-endian header length followed by that much JSON: every tensor's name,
 * dtype and shape. One HTTP range request fetches ~25KB of that off a file of
 * many gigabytes.
 *
 * WHAT THIS IS NOT. Weights are authoritative for shape, not for what runs, and
 * a checkpoint is not one stack. Eight of these ship a multi-token-prediction
 * head whose tensors are named mtp.layers.0.* -- its own layer 0, numbered from
 * zero, beside the main stack's. A parser that keys off `layers.(\d+)` anywhere
 * in a name merges the two and reports a full-attention block sitting on a
 * linear-attention layer, which is how a first pass of this "found" a KV bug in
 * eight models that do not have one. Hence SKIP, and hence the rule that this
 * reports measurements for a human to reconcile and never edits data/models.
 *
 * The MTP head is real weight in the file; llama.cpp does not run it and this
 * site does not count it.
 *
 * Network-bound, so it is not part of `npm run check`. Run it when a model is
 * added or its architecture fields change:  npx tsx scripts/audit-shapes.ts
 */
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const MODELS = join(process.cwd(), 'data', 'models');
const OUT = join(process.cwd(), 'data', 'tensor-shapes.json');
const UA = 'llm-picker-data/1.0 (+https://llm-picker.dev)';
const LAYER = /(?:^|\.)layers\.(\d+)\./;
/** Tensors that are not part of the language model's own forward pass. */
const SKIP = /vision|audio|^mtp\.|\.mtp\.|scale_inv|activation_scale|input_m|output_m/;

type Shape = number[];
type Header = Record<string, { dtype: string; shape: Shape }>;

async function get(url: string, range?: string): Promise<Uint8Array | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const headers: Record<string, string> = { 'user-agent': UA };
      if (range) headers.range = `bytes=${range}`;
      const r = await fetch(url, { headers });
      if (r.ok) return new Uint8Array(await r.arrayBuffer());
      if (r.status === 404 || r.status === 401) return null;
    } catch {
      /* retried below */
    }
    await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
  }
  return null;
}

/** The shard's tensor table, from a range request for its header alone. */
async function header(repo: string, shard: string): Promise<Header | null> {
  const url = `https://huggingface.co/${repo}/resolve/main/${shard}`;
  const first = await get(url, '0-7');
  // A short body is an error page, not a header. Reading one as a length is how
  // a failed fetch turns into fabricated data.
  if (!first || first.length !== 8) return null;
  const n = Number(new DataView(first.buffer, first.byteOffset).getBigUint64(0, true));
  if (!(n >= 16 && n <= 80_000_000)) return null;
  const raw = await get(url, `8-${8 + n - 1}`);
  if (!raw) return null;
  try {
    const h = JSON.parse(new TextDecoder().decode(raw)) as Header;
    delete (h as Record<string, unknown>).__metadata__;
    return h;
  } catch {
    return null;
  }
}

async function weightMap(repo: string) {
  const b = await get(`https://huggingface.co/${repo}/resolve/main/model.safetensors.index.json`);
  if (b) {
    try {
      const j = JSON.parse(new TextDecoder().decode(b));
      if (j.weight_map) return { map: j.weight_map as Record<string, string>, heads: {} as Record<string, Header> };
    } catch {
      /* fall through to the single-file case */
    }
  }
  const h = await header(repo, 'model.safetensors');
  if (!h) return null;
  const map: Record<string, string> = {};
  for (const k of Object.keys(h)) map[k] = 'model.safetensors';
  return { map, heads: { 'model.safetensors': h } };
}

/**
 * The names whose position in a block is fixed by convention, and so can be
 * drawn. Anything outside this list is recorded under `unknown` and left OUT of
 * the diagram: a box in the wrong place is worse than an absent one, and the
 * one thing a tensor name never carries is its order.
 */
const NORMS = [
  'input_layernorm',
  'post_attention_layernorm',
  'pre_feedforward_layernorm',
  'post_feedforward_layernorm',
] as const;
const ATTN = [
  'q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj', 'q_norm', 'k_norm',
  'q_a_proj', 'q_a_layernorm', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_a_layernorm', 'kv_b_proj',
] as const;
const FFN = ['gate_proj', 'up_proj', 'down_proj', 'gate_up_proj'] as const;
/** Recurrent / linear-attention mixers. Drawn as one box, not unpacked. */
const LINEAR = /^(linear_attn|mixer)\.(in_proj|conv1d|A_log|D|dt_bias|norm|out_proj)/;

interface Block {
  norms: string[];
  attention: string[];
  /** dense | moe | none, and the pieces that make it up. */
  ffn: { kind: string; experts?: number; shared?: boolean; router?: boolean; intermediate?: number };
  linearMixer: boolean;
  unknown: string[];
  /**
   * False when the block cannot be drawn honestly. Nemotron names its per-layer
   * norm plain `norm`, which fixes no position: it could be the pre-attention
   * norm or the pre-FFN one, and the name does not say. Rather than guess, the
   * diagram is omitted -- the same fail-closed rule kvGb and overheadFor follow.
   */
  drawable: boolean;
}

interface Measured {
  repo: string;
  /** KV bytes per token held by ONE layer of each attention kind, at fp16. */
  perLayerKvBytes: Record<string, number>;
  kvBytesPerToken?: number;
  kvWindowedBytesPerToken?: number;
  embedRows?: number;
  hidden?: number;
  denseFfn?: number;
  widestDenseLayer?: number;
  moeFfn?: number;
  width?: number;
  /** One representative block per attention kind in `attentionPattern`. */
  blocks?: Record<string, Block>;
  /** Layers whose FFN is dense, where the rest are MoE (GLM starts dense). */
  denseFfnLayers?: number;
  finalNorm?: boolean;
  separateLmHead?: boolean;
}

async function measure(id: string, repo: string, model: Record<string, any>): Promise<Measured | null> {
  const wm = await weightMap(repo);
  if (!wm) return null;
  const { map, heads } = wm;
  const names = Object.keys(map).filter((n) => !SKIP.test(n));

  const shapeOf = async (name: string): Promise<Shape | null> => {
    const shard = map[name];
    if (!shard) return null;
    if (!heads[shard]) {
      const h = await header(repo, shard);
      if (!h) return null;
      heads[shard] = h;
    }
    return heads[shard][name]?.shape ?? null;
  };

  const byLayer = new Map<number, Map<string, string>>();
  for (const n of names) {
    const m = LAYER.exec(n);
    if (!m) continue;
    const i = Number(m[1]);
    if (!byLayer.has(i)) byLayer.set(i, new Map());
    byLayer.get(i)!.set(n.slice(m.index + m[0].length), n);
  }
  const at = async (i: number, suffix: string) => {
    const n = byLayer.get(i)?.get(suffix);
    return n ? await shapeOf(n) : null;
  };

  /** What one layer of this kind caches per token, at fp16, from its tensors. */
  const perLayerKv = async (i: number): Promise<number | null> => {
    const mla = await at(i, 'self_attn.kv_a_proj_with_mqa.weight');
    if (mla) return mla[0] * 2; // one compressed latent, not per-head K and V
    for (const pre of ['self_attn', 'mixer']) {
      // Nemotron calls every block "mixer", attention or Mamba alike.
      const k = await at(i, `${pre}.k_proj.weight`);
      if (k) {
        const v = await at(i, `${pre}.v_proj.weight`);
        // Gemma 4's attention_k_eq_v saves the v_proj WEIGHTS, not the cache:
        // mlx-lm's KVCache allocates and writes both buffers regardless.
        return (k[0] + (v ? v[0] : k[0])) * 2;
      }
    }
    // Phi fuses Q, K and V, so the split needs the stored head counts -- which
    // makes it a joint check: rows must be (heads + 2*kvHeads) * headDim.
    const qkv = await at(i, 'self_attn.qkv_proj.weight');
    if (qkv) {
      const { attentionHeads: h, kvHeads: kv, headDim: hd } = model;
      if (h && kv && hd) {
        if (qkv[0] !== (h + 2 * kv) * hd) {
          throw new Error(`${id}: fused qkv has ${qkv[0]} rows, heads/kvHeads/headDim imply ${(h + 2 * kv) * hd}`);
        }
        return 2 * kv * hd * 2;
      }
    }
    return null;
  };

  const out: Measured = { repo, perLayerKvBytes: {} };
  const pat: string | undefined = model.attentionPattern;
  if (pat) {
    for (const ch of new Set(pat)) {
      const b = await perLayerKv(pat.indexOf(ch));
      if (b !== null) out.perLayerKvBytes[ch] = b;
      else out.perLayerKvBytes[ch] = 0;
    }
    const caching = pat.slice(0, pat.length - (model.kvSharedLayers ?? 0));
    const count = (c: string) => [...caching].filter((x) => x === c).length;
    if (out.perLayerKvBytes.F !== undefined) out.kvBytesPerToken = count('F') * out.perLayerKvBytes.F;
    if (out.perLayerKvBytes.S !== undefined) out.kvWindowedBytesPerToken = count('S') * out.perLayerKvBytes.S;
  } else {
    for (const i of [...byLayer.keys()].sort((a, b) => a - b)) {
      const b = await perLayerKv(i);
      if (b) {
        out.perLayerKvBytes.all = b;
        out.kvBytesPerToken = (model.layers as number) * b;
        break;
      }
    }
  }

  const emb = names.find(
    (n) => (n.endsWith('embed_tokens.weight') || n.endsWith('backbone.embeddings.weight')) && !LAYER.test(n),
  );
  if (emb) {
    const s = await shapeOf(emb);
    if (s) {
      out.embedRows = s[0];
      out.hidden = s[1];
    }
  }

  let dense = 0;
  let moe_ = 0;
  for (const n of names) {
    const m = LAYER.exec(n);
    if (!m) continue;
    const i = Number(m[1]);
    const suffix = n.slice(m.index + m[0].length).replace(/^mixer\.(?=experts|shared_experts|gate|up)/, 'mlp.');
    if (/^mlp\.(gate_proj|up_proj)\.weight$/.test(suffix)) {
      const s = await shapeOf(n);
      // Gemma 4's use_double_wide_mlp doubles the FFN on KV-shared layers, so
      // the widest layer is not necessarily layer 0.
      if (s && s[0] > dense) {
        dense = s[0];
        out.widestDenseLayer = i;
      }
    } else if (/^mlp\.gate_up_proj\.weight$/.test(suffix)) {
      const s = await shapeOf(n);
      if (s && s[0] / 2 > dense) {
        dense = s[0] / 2;
        out.widestDenseLayer = i;
      }
    } else if (/^mlp\.experts\.\d+\.(gate|up)_proj\.weight$/.test(suffix)) {
      const s = await shapeOf(n);
      if (s) moe_ = Math.max(moe_, s[0]);
    } else if (/^mlp\.experts\.down_proj$/.test(suffix)) {
      // Stacked as [numExperts, hidden, ffn]. down_proj is the unambiguous one:
      // a fused gate_up whose FFN dim happens to equal hidden cannot be split.
      const s = await shapeOf(n);
      if (s && s.length === 3) moe_ = Math.max(moe_, Math.min(s[1], s[2]));
    }
  }
  // --- what one block is made of, for the diagram ---
  const expertIds = new Set<number>();
  let stackedExperts = 0;
  let denseLayers = 0;
  let moeLayers = 0;
  for (const [, suffixes] of byLayer) {
    const keys = [...suffixes.keys()];
    const isMoe = keys.some((k) => k.includes('experts'));
    const isDense = keys.some((k) => /^(mlp|mixer)\.(gate_proj|up_proj|gate_up_proj)\.weight$/.test(k));
    if (isMoe) moeLayers++;
    else if (isDense) denseLayers++;
    for (const k of keys) {
      const e = /\.experts\.(\d+)\./.exec(k);
      if (e) expertIds.add(Number(e[1]));
      // Stacked experts carry no index in the name: the count is the first dim.
      if (/\.experts\.(gate_up_proj|gate_proj|down_proj)$/.test(k)) {
        const n = suffixes.get(k);
        if (n) {
          const sh = await shapeOf(n);
          if (sh && sh.length === 3) stackedExperts = Math.max(stackedExperts, sh[0]);
        }
      }
      if (/\.experts\.(gate_up_proj|gate_proj|up_proj)_bias$/.test(k)) {
        const n = suffixes.get(k);
        if (n) {
          const sh = await shapeOf(n);
          if (sh && sh.length === 2) {
            stackedExperts = Math.max(stackedExperts, sh[0]);
            // gate and up fused into one row block; the FFN width is half.
            moe_ = Math.max(moe_, k.includes('gate_up') ? sh[1] / 2 : sh[1]);
          }
        }
      }
    }
  }
  if (moeLayers > 0 && denseLayers > 0) out.denseFfnLayers = denseLayers;

  const ffnLayer =
    moeLayers > denseLayers
      ? [...byLayer.keys()].sort((a, b) => a - b).find((i) =>
          [...(byLayer.get(i)?.keys() ?? [])].some((k) => k.includes('experts')))
      : undefined;

  const describe = (i: number): Block => {
    const keys = [...(byLayer.get(i)?.keys() ?? [])];
    const ffnKeys = ffnLayer === undefined ? keys : [...(byLayer.get(ffnLayer)?.keys() ?? [])];
    const has = (n: string) => keys.some((k) => k === `${n}.weight` || k.endsWith(`.${n}.weight`));
    const block: Block = {
      norms: NORMS.filter((n) => has(n)),
      attention: ATTN.filter((n) => has(n)),
      ffn: { kind: 'none' },
      linearMixer: keys.some((k) => LINEAR.test(k)),
      unknown: [],
    };
    const moe = ffnKeys.some((k) => k.includes('experts'));
    if (moe) {
      block.ffn = {
        kind: 'moe',
        experts: expertIds.size || stackedExperts || undefined,
        shared: ffnKeys.some((k) => k.includes('shared_expert')),
        router: ffnKeys.some((k) => /\.(gate|router)\.weight$/.test(k)),
        intermediate: moe_ ? moe_ : undefined,
      };
    } else if (FFN.some((n) => ffnKeys.some((k) => k === `${n}.weight` || k.endsWith(`.${n}.weight`)))) {
      block.ffn = { kind: 'dense', intermediate: dense || undefined };
    }
    for (const k of keys) {
      // A bias or a quantisation scale belongs to the projection it hangs off,
      // whose position is already settled. It is not an unplaced component.
      if (/\.(bias|scale|scales|blocks|scale_inv)$/.test(k)) continue;
      const leaf = k.replace(/\.weight$/, '').split('.').pop()!;
      const known =
        (NORMS as readonly string[]).includes(leaf) ||
        (ATTN as readonly string[]).includes(leaf) ||
        (FFN as readonly string[]).includes(leaf) ||
        LINEAR.test(k) ||
        k.includes('experts') ||
        /\.(gate|router)\.weight$|shared_expert/.test(k);
      if (!known && !block.unknown.includes(leaf)) block.unknown.push(leaf);
    }
    // input_layernorm is the anchor: it is the one name whose position is never
    // in doubt. Without it nothing else in the block can be placed either.
    block.drawable = block.norms.includes('input_layernorm');
    return block;
  };
  out.blocks = {};
  if (pat) {
    for (const ch of new Set(pat)) out.blocks[ch] = describe(pat.indexOf(ch));
  } else {
    const first = [...byLayer.keys()].sort((a, b) => a - b)[0];
    if (first !== undefined) out.blocks.all = describe(first);
  }
  out.finalNorm = names.some((n) => /\.(norm|norm_f|final_layernorm)\.weight$/.test(n) && !LAYER.test(n));
  out.separateLmHead = names.some((n) => n.endsWith('lm_head.weight'));

  if (dense) out.denseFfn = dense;
  if (moe_) out.moeFfn = moe_;
  const ept: number | undefined = model.expertsPerToken;
  const widths = [out.hidden ?? 0, dense, moe_ && ept ? moe_ * ept : 0].filter(Boolean);
  if (widths.length) out.width = Math.max(...widths);
  return out;
}

const models = readdirSync(MODELS)
  .filter((f) => f.endsWith('.json'))
  .map((f) => JSON.parse(readFileSync(join(MODELS, f), 'utf8')));

const out: Record<string, Measured> = {};
const problems: string[] = [];
for (const m of models) {
  const repo = /https:\/\/huggingface\.co\/([^/]+\/[^/]+)\//.exec(m.kvSourceUrl ?? '')?.[1];
  if (!repo) {
    problems.push(`${m.id}: no Hugging Face repo in kvSourceUrl`);
    continue;
  }
  try {
    const r = await measure(m.id, repo, m);
    if (!r) problems.push(`${m.id}: weights unreachable at ${repo}`);
    else out[m.id] = r;
    process.stdout.write(`  ${m.id.padEnd(28)} ${r ? 'ok' : 'UNREACHABLE'}\n`);
  } catch (e) {
    problems.push(String(e));
    process.stdout.write(`  ${m.id.padEnd(28)} ${String(e)}\n`);
  }
}

writeFileSync(
  OUT,
  `${JSON.stringify(
    {
      retrieved: new Date().toISOString().slice(0, 10),
      method:
        'safetensors header read over HTTP range request; see scripts/audit-shapes.ts. ' +
        'Authoritative for tensor SHAPE only -- whether a tensor is used is decided by ' +
        "the model's config and implementation, not by its presence in the checkpoint.",
      models: Object.fromEntries(Object.entries(out).sort(([a], [b]) => a.localeCompare(b))),
    },
    null,
    2,
  )}\n`,
);
console.log(`\nwrote ${OUT} — ${Object.keys(out).length} models`);
if (problems.length) {
  console.log('problems:');
  for (const p of problems) console.log(`  ${p}`);
  process.exitCode = 1;
}
