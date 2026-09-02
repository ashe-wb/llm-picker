import {
  bestScoredFit,
  compareMachines,
  pickModels,
  type MachineComparison,
  type PickerData,
  type Recommendation,
} from '../lib/picker';
import {
  bandwidthFor,
  budgetFor,
  budgetSource,
  chipFor,
  formatMemory,
  generationSpeed,
  type Speed,
  maxWiredGb,
  prefillFor,
  raiseSteps,
  ramOptionsFor,
  sameCapacityPresets,
  systemBandwidthFor,
  flashAttentionOffCostGb,
  publishedCeilingGb,
  hasSeparateSystemMemory,
  layerSplit,
} from '../lib/fit';
import type { Machine, PlatformId } from '../lib/schemas';

const ctxLabel = (n: number) => (n >= 1024 ? `${Math.round(n / 1024)}K` : String(n));

const FIT_STAMP: Record<Recommendation['fitState'], string> = {
  comfortable: 'border border-ok text-ok',
  tight: 'border border-warn text-warn',
  no: 'bg-serious text-paper',
};

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

function chipHtml(chip: Recommendation['warnings'][number]['chips'][number]): string {
  const noFix = chip.noFix;
  const tag = noFix
    ? '<span class="pt-0.5 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-serious">No fix</span>'
    : '<span class="pt-0.5 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-ok">Fix</span>';
  const limited = chip.limited
    ? ' <span class="font-mono text-[12px] text-serious">[limited — depends on a capability this model is weak at]</span>'
    : '';
  return `<div class="flex gap-3 border-l-2 ${noFix ? 'border-serious' : 'border-ok'} py-1 pl-3 text-sm">
    ${tag}
    <div>
      <span class="font-medium">${esc(chip.label)}</span>${chip.effectiveness ? ` <span class="dsnote">[${chip.effectiveness}]</span>` : ''}${limited}
      <p class="mt-0.5 text-faded">${esc(chip.note)}</p>
    </div>
  </div>`;
}

/**
 * Spare memory can only be spent on context once the band is picked, so say what
 * it buys. Without this a card reading "57GB of your 79GB" looks like 22GB wasted.
 *
 * The CONSERVATIVE end only, deliberately.
 *
 * Every term behind this is derived from the model's own config except the
 * framework itself, which published figures put anywhere between 0.5 and 1.5GB.
 * That uncertainty is worth several thousand tokens, and this used to be
 * published as a range spanning it.
 *
 * It no longer is, because the only person who has run these models on a real
 * machine reports the figure reads high — and when an input is unmeasured and
 * the one report available says the estimate is optimistic, the flattering half
 * of that input has no business being the headline. `maxContextTokensOptimistic`
 * is still computed and still on the Fit; it is simply not something to put in
 * front of a reader until a measurement earns it.
 */
function headroomNote(rec: Recommendation): string {
  const lo = rec.fit.maxContextTokens;
  if (rec.fit.offload || lo <= rec.assumedContextTokens) return '';
  if (lo >= rec.model.contextLength) {
    return ` · ${esc(`ITS FULL ${ctxLabel(rec.model.contextLength)} WINDOW, BY THIS ESTIMATE`)}`;
  }
  return ` · ${esc(`ROUGHLY ${ctxLabel(lo)} CONTEXT, BY THIS ESTIMATE`)}`;
}

/**
 * Apple Silicon only, and only because of what an Apple Silicon owner is likely
 * to be thinking when they read a tok/s figure.
 *
 * Every speed on this site is a GGUF figure, and the card says so. But MLX is
 * Apple's own runtime, it is widely reported as much faster, and a Mac owner
 * who has read that will reasonably assume these numbers already include it.
 * They do not. Saying "GGUF runtime" does not correct that expectation for
 * someone who does not yet know the two are different things.
 *
 * What it must NOT do is promise a bump. The Apple Silicon overhead range was
 * set on both runtimes (docs/mac-tok-s-validation.md), so MLX is inside the
 * figures already, mostly at the top. Same-machine comparisons at matched
 * 4-bit put it at 1.0-1.2x llama.cpp on most models and 1.6-1.9x on Qwen3.5,
 * where llama.cpp's kernels for the linear-attention layers lag. So this says
 * where in the range to look, names the one family that sits above it, and
 * corrects the 3x that is really Ollama's wrapper.
 */
function mlxNote(m: Machine): string {
  if (m.platform !== 'unified') return '';
  const body =
    'The range above was set on both runtimes. Same-machine comparisons at 4-bit on both sides put MLX at 1.0–1.2× llama.cpp on most models, so if you run MLX expect the top of the range; on Qwen3.5 it is 1.6–1.9×, and lands above it. The 3× sometimes quoted is Ollama changing its own backend, and about half of that gap is Ollama\'s wrapper rather than MLX.';
  return `<div class="border-l-2 border-edge bg-paper py-2 pl-3 pr-3">
    <p class="dslabel">On a Mac, llama.cpp or MLX</p>
    <p class="mt-1 dsnote">${esc(body)} <a href="/methodology" class="underline decoration-dotted underline-offset-2 hover:text-serious">How speed is estimated</a></p>
    <p class="mt-1.5 dsnote">The memory figures above are llama.cpp's too, including the context ceiling. MLX allocates differently and this site does not model it, so on an MLX build expect the context you can actually load to differ from what is shown here — and, on the one report available, to be lower.</p>
  </div>`;
}

/**
 * What the runtime takes before a single token of yours is stored, itemised.
 *
 * The reader is charged one opaque number today, and the largest part of it
 * surprises everybody: the final vocabulary projection, which is half a
 * gigabyte on a large-vocabulary model whatever its parameter count. Naming the
 * pieces is also what makes the figure checkable against what llama.cpp prints
 * at startup, which is the only way a reader can catch us being wrong.
 */
function overheadNote(rec: Recommendation, hardware: PickerData['hardware']): string {
  const o = rec.fit.overhead;
  const fa = flashAttentionOffCostGb(rec.model, rec.assumedContextTokens, hardware);
  const parts = [
    `${formatMemory(o.frameworkHiGb)} llama.cpp and its kernels`,
    `${formatMemory(o.logitsGb)} vocabulary projection (${rec.model.vocabSize?.toLocaleString()} tokens wide)`,
    `${formatMemory(o.activationsGb)} activations`,
    `${formatMemory(o.kqMaskGb)} attention mask at ${ctxLabel(rec.assumedContextTokens)}`,
  ];
  // Only worth the reader's attention once it is large enough to break a fit.
  const faLine =
    fa !== null && fa >= 1
      ? `<p class="mt-1 dsnote">Assumes flash attention is on, which llama.cpp does by default. Without it this model materialises attention scores instead of masking them, costing about <span class="text-ink">${formatMemory(fa)}</span> more at ${ctxLabel(rec.assumedContextTokens)} — enough to turn this fit into a failure to load.</p>`
      : '';
  return `<div class="border-l-2 border-edge bg-paper py-2 pl-3 pr-3">
    <p class="dslabel">What the runtime takes, before your context</p>
    <p class="mt-1 dsnote">${esc(parts.join(' · '))}</p>
    <p class="mt-1 dsnote">Assumes one serving slot at llama.cpp's default 512 micro-batch. Running <span class="text-ink">--parallel N</span> multiplies the cache above by N, which this estimate does not model.</p>
    ${faLine}
  </div>`;
}

/**
 * The headline score is discounted for unrated dimensions, so say so rather than
 * letting a 7.1 quietly render as a 6.5.
 */
function evidenceNote(rec: Recommendation): string {
  if (rec.coverage >= 1) return '';
  const pct = Math.round(rec.coverage * 100);
  return `<p class="dsline">SCORED ${rec.measuredScore}/10 ON THE ${pct}% OF THIS WORKLOAD WE HAVE EVIDENCE FOR${
    rec.incompleteData ? ' — the rest is unrated and counted as average, which is why the headline is lower' : ''
  }.</p>`;
}

/** Says *why* a fit is tight — headroom or offload — rather than just flagging it. */
function fitNote(rec: Recommendation): string {
  if (rec.fitState !== 'tight') return '';
  const why = rec.fit.offload
    ? 'Bigger than your graphics card, so your CPU runs the rest from system RAM — it works, several times slower.'
    : `Only ${formatMemory(rec.fit.headroomGb)} spare. Close other apps, or drop to a shorter context or a smaller quant.`;
  return `<p class="dsline is-warn border-l-2 border-warn bg-paper py-2 pl-3 pr-3">TIGHT: ${esc(why)}</p>`;
}

/**
 * Generation speed, and the class of prompt processing that goes with it. Kept
 * to a range and to a class: the first is an estimate, the second is a quantity
 * a number would overclaim. Returns nothing when the machine has no bandwidth
 * on file rather than guessing one.
 *
 * The runtime is named because the efficiency behind the figure was measured on
 * one. A reader on MLX would otherwise assume it covered them, and it does not.
 */
function speedNote(
  rec: Recommendation,
  bandwidthGbs: number | undefined,
  prefill: string | undefined,
  systemBandwidthGbs: number | undefined,
  isApple: boolean,
  hardware: PickerData['hardware'],
  platform: Machine['platform'],
): string {
  const s = generationSpeed(
    rec.fit.weightsGb,
    rec.fit.kvGb,
    rec.model,
    rec.fit,
    bandwidthGbs,
    systemBandwidthGbs,
    hardware,
    platform,
  );
  if (!s) return '';
  const active =
    rec.model.activeParamsB !== undefined
      ? ` — READS ${formatMemory(s.bytesReadGb)} PER TOKEN, NOT ${formatMemory(rec.fit.weightsGb)}, BECAUSE ONLY ${rec.model.activeParamsB}B OF ${rec.model.paramsB}B IS ACTIVE`
      : '';
  const throttle = s.throttledByOffload
    ? ` — HELD BACK BY THE LAYERS YOUR CPU IS RUNNING, WHICH READ SYSTEM RAM AT ${systemBandwidthGbs}GB/S RATHER THAN ${bandwidthGbs}GB/S`
    : '';
  const pf = prefill ? ` · PROMPT PROCESSING: ${prefill.toUpperCase()}` : '';
  // On a Mac the reader most likely to have heard MLX is fast is exactly the
  // one reading this number, so where MLX sits rides on the number itself,
  // every card, rather than only in the note below them. The Apple range was
  // set on both runtimes (docs/mac-tok-s-validation.md): MLX is 1.0-1.2x
  // llama.cpp on most models, 1.6-1.9x on Qwen3.5. No "floor" — that was the
  // old 1.5x claim, which rested on a misquoted measurement.
  //
  // No context qualifier. It used to say "at shorter contexts", resting on a
  // reported 40K crossover from a single article whose headline figure this site
  // rejects. The one controlled sweep available shows MLX, llama.cpp and Ollama
  // all roughly halving together at 32K — the ratio holds, the advantage does
  // not evaporate — so the qualifier was hedging against something unevidenced.
  const mlx = isApple
    ? ` — LLAMA.CPP OR MLX; MLX SITS NEAR THE TOP OF THIS RANGE${rec.model.id.startsWith('qwen3.5') ? ', AND ABOVE IT ON QWEN3.5' : ''}`
    : '';
  const runtime = isApple ? '' : ' ON A GGUF RUNTIME';
  return `<p class="dsline">~${s.lo}–${s.hi} TOK/S GENERATING${runtime}, BY THIS ESTIMATE${pf}<span class="text-faded/85">${esc(active + throttle + mlx)}</span></p>`;
}

/**
 * The machines that hold the same amount of memory as this one, priced on the
 * model actually being recommended. This is the one comparison the site is
 * equipped to make, and the hedge below it is part of the component rather than
 * a footnote: generation speed is a single axis, and the axes it ignores —
 * price, watts, heat, noise, desk space, portability — routinely decide the
 * question on their own.
 */
function capacityCompare(
  rec: Recommendation,
  m: Machine,
  data: PickerData,
  ctx: number,
  task: PickerData['tasks'][number],
): string {
  const peers = sameCapacityPresets(m, data.hardware);
  if (peers.length < 2) return '';
  const here = matchLabel(m, data);

  const rows = peers
    .map((p) => {
      // The reader's own row must honour the chip they picked; the others take
      // their preset's default. Resolving through bandwidthFor rather than
      // reading the preset field is also what makes Macs work at all, since an
      // Apple preset carries a chip, not a bandwidth.
      const mine = p.id === m.presetId;
      const pm: Machine = mine
        ? m
        : { platform: p.platform, ramGb: p.ramGb, vramGb: p.vramGb, presetId: p.id };
      const bw = bandwidthFor(pm, data.hardware);
      const sysBw = systemBandwidthFor(pm, data.hardware);
      // The same chooser the card uses, so the table cannot quote a band the
      // card above it does not name.
      const peer = bestScoredFit(rec.model, pm, task, data, ctx);
      const speed = peer
        ? generationSpeed(peer.fit.weightsGb, peer.fit.kvGb, rec.model, peer.fit, bw, sysBw, data.hardware, pm.platform)
        : null;
      return { p, speed, fits: !!peer, band: peer?.band, bw, chip: chipFor(pm, data.hardware), prefill: prefillFor(pm, data.hardware) };
    })
    .sort((a, b) => (b.speed?.hi ?? -1) - (a.speed?.hi ?? -1));

  const body = rows
    .map(({ p, speed, fits, band, bw, chip, prefill }) => {
      const mine = p.id === m.presetId || (!m.presetId && p.label === here);
      // Through prefillFor, not the preset field: an Apple preset carries a
      // chip, and since the M5 the chip is what sets the class.
      const pf = (prefill ?? '').toUpperCase();
      const figure = !fits
        ? '<span class="text-faded">does not fit</span>'
        : speed
          ? `~${speed.lo}–${speed.hi} tok/s <span class="text-faded">at ${esc(band ?? '')}</span>`
          : '<span class="text-faded">—</span>';
      return `<li class="flex flex-wrap items-baseline gap-x-3 font-mono text-[12px]${mine ? ' text-ink' : ' text-faded'}">
        <span class="min-w-[13rem]">${esc(p.label)}${chip && !p.label.includes(chip.label) ? ` <span class="text-faded">${esc(chip.label)}</span>` : ''}${mine ? ' <span class="text-faded">(yours)</span>' : ''}</span>
        <span class="tabular-nums">${bw ?? '—'} GB/s</span>
        <span class="tabular-nums">${figure}</span>
        <span>prefill ${esc(pf)}</span>
      </li>`;
    })
    .join('');

  return `<div class="border border-rule bg-panel px-3 py-2">
    <p class="dslabel">Same capacity, different speed</p>
    <p class="mt-1 dsnote">Running ${esc(rec.model.name)} at ${ctxLabel(ctx)}, on the other machines that hold ${formatMemory(m.vramGb ?? m.ramGb)}.</p>
    <ul class="mt-2 space-y-1">${body}</ul>
    <p class="mt-2 border-l-2 border-edge pl-3 dsnote">
      Generation speed only, and only for this model. Nothing here accounts for price,
      power draw, heat and noise, desk space or portability — machines that post similar
      numbers can suit completely different lives.
    </p>
  </div>`;
}

/**
 * Two machines, side by side, leading with whatever actually differs.
 *
 * Deliberately not a table of seven workloads: for a well-matched pair it would
 * be seven rows of the same model name with the real answer — the speed — left
 * for the reader to infer. The arithmetic lives in compareMachines so it can be
 * tested; this only decides what to say about it.
 */
function compareBlock(c: MachineComparison, bLabel: string): string {
  const row = (name: string, s: MachineComparison['a'], mine: boolean) => {
    const spill =
      s.offloadGb > s.budgetGb
        ? `${formatMemory(s.offloadGb)} with your CPU`
        : 'one pool, nowhere to spill';
    // A split pick's speed comes mostly from the second figure, so quoting only
    // the first would make the tok/s above look unexplained.
    const split = s.top?.speed?.throttledByOffload === true;
    const bw = s.bandwidthGbs
      ? split
        ? `${s.bandwidthGbs} + ${s.systemBandwidthGbs ?? '—'} GB/s`
        : `${s.bandwidthGbs} GB/s`
      : '—';
    return `<tr class="${mine ? 'text-ink' : 'text-faded'}">
      <td class="py-1 pr-4">${esc(name)}${mine ? ' <span class="text-faded">(yours)</span>' : ''}</td>
      <td class="py-1 pr-4 tabular-nums">${formatMemory(s.budgetGb)}</td>
      <td class="py-1 pr-4 tabular-nums">${esc(bw)}${split ? '<span class="text-faded"> (split)</span>' : ''}</td>
      <td class="py-1 pr-4">${esc(s.prefill ?? '—')}</td>
      <td class="py-1 tabular-nums">${esc(spill)}</td>
    </tr>`;
  };

  const speed = (s: MachineComparison['a']) => (s.top?.speed ? `~${s.top.speed.lo}–${s.top.speed.hi} tok/s` : '—');
  let verdict: string;
  if (!c.a.top || !c.b.top) {
    const dead = !c.a.top ? 'yours' : bLabel;
    verdict = `Nothing rated runs this workload on ${esc(dead)}, so there is nothing to compare here.`;
  } else if (c.sameBand && c.speedRatio) {
    const faster = c.speedRatio >= 1;
    const x = faster ? c.speedRatio : Math.round((1 / c.speedRatio) * 100) / 100;
    verdict =
      `Both lead with <strong class="text-ink">${esc(c.a.top.model.name)}</strong> at ${esc(c.a.top.bandLabel.split(' ')[0]!)}. ` +
      `${esc(faster ? bLabel : 'Yours')} reads memory ${x}× faster — ${speed(faster ? c.b : c.a)} against ${speed(faster ? c.a : c.b)}.`;
  } else if (c.sameModel) {
    verdict =
      `Both lead with <strong class="text-ink">${esc(c.a.top.model.name)}</strong>, but at different quantization: ` +
      `${esc(c.a.top.bandLabel.split(' ')[0]!)} on yours against ${esc(c.b.top.bandLabel.split(' ')[0]!)}, because the budgets differ.`;
  } else {
    verdict =
      `Different answers: <strong class="text-ink">${esc(c.a.top.model.name)}</strong> at ${esc(c.a.top.bandLabel.split(' ')[0]!)} on yours, ` +
      `<strong class="text-ink">${esc(c.b.top.model.name)}</strong> at ${esc(c.b.top.bandLabel.split(' ')[0]!)} on ${esc(bLabel)}.`;
  }

  const splitting = [c.a.top?.speed?.throttledByOffload && 'yours', c.b.top?.speed?.throttledByOffload && bLabel]
    .filter(Boolean)
    .join(' and ');
  const splitNoteText = splitting
    ? ` On ${splitting}, that figure is a model split between the graphics card and the CPU, not one running wholly on the card.`
    : '';

  const agree =
    c.comparableTasks > 0
      ? `Same pick on ${c.agreeOn} of ${c.comparableTasks} workloads.`
      : '';

  return `<div class="border border-rule bg-panel px-3 py-2">
    <div class="flex flex-wrap items-baseline justify-between gap-2">
      <p class="dslabel">Yours vs ${esc(bLabel)}</p>
      <p class="dsnote">${esc(agree)}</p>
    </div>
    <div class="mt-2 overflow-x-auto">
      <table class="w-full font-mono text-[12px]">
        <thead class="text-faded">
          <tr class="text-left">
            <th class="py-1 pr-4 font-normal">machine</th><th class="py-1 pr-4 font-normal">budget</th>
            <th class="py-1 pr-4 font-normal">card + CPU bandwidth</th><th class="py-1 pr-4 font-normal">prefill</th>
            <th class="py-1 font-normal">reach with the CPU</th>
          </tr>
        </thead>
        <tbody>${row('Yours', c.a, true)}${row(bLabel, c.b, false)}</tbody>
      </table>
    </div>
    <p class="mt-2 dsnote">${verdict}${esc(splitNoteText)}</p>
    <p class="mt-2 border-l-2 border-edge pl-3 dsnote">
      Speed and memory only. Nothing here accounts for price, power draw, heat and noise, desk
      space or portability — and between a laptop and a tower those usually decide it.
    </p>
  </div>`;
}

/** The preset label matching the described machine, for marking "yours". */
function matchLabel(m: Machine, data: PickerData): string | undefined {
  return data.hardware.presets.find(
    (x) => x.platform === m.platform && x.ramGb === m.ramGb && (x.vramGb ?? undefined) === (m.vramGb ?? undefined),
  )?.label;
}

/**
 * Said only when the leading pick is the best available AND still weak.
 *
 * The score and the split options are already on the page, but a reader has to
 * put them together themselves to learn the thing that matters: this machine
 * cannot do this task well, and here is what it would cost to do better. On an
 * 8GB card asking for code, the honest summary is that the best model which
 * fits scores 4 out of 10.
 *
 * Where nothing beats it, that is said too. Silence there would read as the
 * site having nothing to offer rather than the machine having nothing to give.
 */
function weakBestNote(rec: Recommendation, better: Recommendation | undefined, speed: Speed | null, betterSpeed: Speed | null): string {
  if (!rec.weakBest) return '';
  const body = better
    ? `The best model that fits your graphics card scores ${rec.score} out of 10 on this workload. <span class="text-ink">${esc(better.model.name)}</span> scores ${better.score}, but it does not fit — your CPU would run part of it${betterSpeed && speed ? `, at roughly ${betterSpeed.lo}–${betterSpeed.hi} tok/s against ${speed.lo}–${speed.hi}` : ''}. Whether that trade is worth making is yours to judge; the site will not pretend the faster answer is also the better one.`
    : `The best model that fits your graphics card scores ${rec.score} out of 10 on this workload, and nothing else on offer does better — not at a smaller quantization, and not by handing layers to your CPU. This machine can run a model for this job; it cannot run a good one.`;
  return `<div class="border-l-2 border-warn bg-paper py-2 pl-3 pr-3">
    <p class="dslabel">Best available, and still not good</p>
    <p class="mt-1 text-sm">${body}</p>
  </div>`;
}

/**
 * Why this band and not the bigger one that also fits. Without it, a reader
 * whose 80GB card can obviously hold FP16 sees it missing and assumes the site
 * is wrong rather than deliberate. Speed is not claimed here — the tok/s line
 * above already says it, and this figure is exact where that one is an estimate.
 */
function bandChoiceNote(rec: Recommendation): string {
  const sk = rec.skippedBand;
  if (!sk) return '';
  const lighter = Math.round((sk.demandGb - rec.fit.demandGb) * 10) / 10;
  const why =
    sk.scoreDelta > 0
      ? `SCORES ${sk.scoreDelta} HIGHER ON THIS WORKLOAD`
      : 'SAME SCORE ON THIS WORKLOAD';
  const size = lighter > 0 ? `, AND ${formatMemory(lighter)} LIGHTER` : '';
  return `<p class="dsline">${esc(rec.bandLabel.split(' ')[0]!.toUpperCase())} RATHER THAN ${esc(sk.label.toUpperCase())}: ${why}${size}.</p>`;
}

/**
 * The advice command with the real number in it. It shipped as a placeholder —
 * `iogpu.wired_limit_mb=<megabytes>` — which leaves the reader to work out both
 * the number and the unit conversion, and the unit is the easy half to
 * get wrong: this setting is in MEGABYTES, so a slip of one zero either wastes
 * the raise or wires down more than the machine has.
 */
function commandFor(command: string, m: Machine, data: PickerData): string {
  const max = maxWiredGb(m, data.hardware);
  return max ? command.replace('<megabytes>', String(max * 1024)) : command;
}

/**
 * A partial fit, restated as the thing the reader has to do about it.
 *
 * The site's own arithmetic already knows how much of the model the card can
 * hold; until now it said so as a percentage, which nobody types into anything.
 * This says it in layers, and then in the flag, because the number the site
 * computed and the number the reader sets should not be two different numbers.
 *
 * The flag differs by architecture and getting it wrong is worse than saying
 * nothing: on a mixture-of-experts model you do NOT lower -ngl. You keep every
 * layer on the card and move only the expert tensors, which are most of the
 * weight and are read rarely, with --n-cpu-moe. Handing an MoE owner a plain
 * -ngl would send them to a much slower configuration than the one this site
 * just estimated for them.
 */
function splitNote(rec: Recommendation): string {
  if (!rec.fit.offload) return '';
  const sp = layerSplit(rec.model, rec.fit);
  const moe = rec.model.activeParamsB !== undefined;
  const cmd = sp
    ? moe
      ? `-ngl 99 --n-cpu-moe ${Math.max(1, sp.onCpu)}`
      : `-ngl ${sp.onGpu}`
    : null;
  const layers = sp
    ? `About <span class="text-ink">${sp.onGpu} of its ${sp.total} layers</span> fit your graphics card. Your CPU runs the other ${sp.onCpu}, reading them from system RAM.`
    : `Part of this model runs on your graphics card and the rest runs on your CPU, out of system RAM. Both are doing work; the CPU's share is the slower half and sets the speed above.`;
  const how = cmd
    ? moe
      ? `<p class="mt-1.5 dsnote">In llama.cpp, keep every layer on the card and move the experts instead — they are most of the weight and are read rarely:</p>
         <code class="mt-1 block bg-paper px-2 py-1 font-mono text-[12px] text-ink">${esc(cmd)}</code>
         <p class="mt-1 dsnote">A starting value. Lower it until your graphics memory is nearly full.</p>`
      : `<p class="mt-1.5 dsnote">In llama.cpp:</p>
         <code class="mt-1 block bg-paper px-2 py-1 font-mono text-[12px] text-ink">${esc(cmd)}</code>
         <p class="mt-1 dsnote">A starting value, not a setting to trust — layers are not exactly equal in size. Raise it until it stops loading, then step back one.</p>`
    : '';
  return `<div class="border-l-2 border-warn bg-paper py-2 pl-3 pr-3">
    <p class="dslabel">Split across your graphics card and your CPU</p>
    <p class="mt-1 text-sm">${layers}</p>
    ${how}
    <p class="mt-1.5 dsnote">This assumes you set that split yourself. If you leave it to the driver and it quietly spills graphics memory into system RAM instead, that path really does cross PCIe every token and is far slower than the estimate above.</p>
  </div>`;
}

function recHtml(
  rec: Recommendation,
  i: number,
  source: string,
  bandwidthGbs: number | undefined,
  prefill: string | undefined,
  systemBandwidthGbs: number | undefined,
  split: string,
  overhead: string,
  isApple: boolean,
  weakNote: string,
  hardware: PickerData['hardware'],
  platform: Machine['platform'],
): string {
  const stamp = {
    severe: 'bg-serious text-paper',
    moderate: 'border border-warn text-warn',
    mild: 'border border-faded text-faded',
  };
  const tags = rec.tags
    .map((t) =>
      t === 'top-pick'
        ? rec.weakBest
          // Best available and still not good. A solid stamp reads as
          // endorsement, so this one does not get one — the ranking is right,
          // the word "top" is what was overclaiming.
          ? '<span class="border border-warn px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-warn">Best that fits</span>'
          : '<span class="bg-ink px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-paper">Top pick</span>'
        : '<span class="border border-edge px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em]">Lightest good option</span>',
    )
    .join(' ');
  const warnings = rec.warnings
    .map(
      (w) => `<div class="border border-rule bg-paper">
        <div class="flex flex-wrap items-center gap-2 border-b border-rule px-3 py-1.5">
          <span class="px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] ${stamp[w.entry.severity]}">${w.entry.severity}</span>
          <span class="font-mono text-[12px] uppercase tracking-[0.12em] text-faded">${esc(w.entry.dimensions.join(', '))}</span>
        </div>
        <div class="space-y-2 px-3 py-2">
          <p class="text-sm">${esc(w.entry.summary)}</p>
          ${w.chips.map(chipHtml).join('')}
        </div>
      </div>`,
    )
    .join('');
  const qualifiers = rec.qualifiers.length
    ? `<div class="border-l-2 border-edge bg-paper py-2 pl-3 pr-3">
        <p class="dslabel">Why this one</p>
        <p class="mt-1 text-sm">${rec.qualifiers.map(esc).join(' <span class="text-faded">·</span> ')}</p>
        ${rec.tieNote ? `<p class="mt-1 dsnote">${esc(rec.tieNote)}</p>` : ''}
      </div>`
    : '';
  return `<article class="border-2 border-edge bg-panel">
    <div class="flex flex-wrap items-center gap-3 border-b border-edge px-4 py-2.5">
      <span class="font-mono text-xs font-bold text-faded">${String(i + 1).padStart(2, '0')}</span>
      <a href="/models/${esc(rec.model.id)}" class="font-mono text-base font-bold underline decoration-rule underline-offset-4 hover:decoration-serious">${esc(rec.model.name)}</a>
      ${tags}
      <span class="px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] ${FIT_STAMP[rec.fitState]}">${rec.fitState === 'comfortable' ? 'Fits' : 'Tight fit'}</span>
      ${rec.coverage < 1 ? `<span class="border border-faded px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-faded">${Math.round(rec.coverage * 100)}% rated</span>` : ''}
      <span class="ml-auto font-mono text-sm font-semibold tabular-nums">${rec.score}<span class="font-normal text-faded">/10</span></span>
    </div>
    <div class="space-y-3 px-4 py-3">
      <p class="dsline">RUNS AT: ${esc(rec.bandLabel).toUpperCase()} · ${formatMemory(rec.fit.weightsGb)} WEIGHTS + ${formatMemory(rec.fit.kvGb)} KV AT ${ctxLabel(rec.assumedContextTokens)} + ${formatMemory(rec.fit.overheadGb)} RUNTIME = <span class="text-ink">${formatMemory(rec.fit.demandGb)}</span> OF YOUR ${formatMemory(rec.fit.budgetGb)} <span class="text-faded/85">(${esc(source.toUpperCase())})</span>${headroomNote(rec)}</p>
      ${speedNote(rec, bandwidthGbs, prefill, systemBandwidthGbs, isApple, hardware, platform)}
      ${bandChoiceNote(rec)}
      ${weakNote}
      ${overhead}
      ${split}
      ${fitNote(rec)}
      ${evidenceNote(rec)}
      ${qualifiers}
      <p class="text-sm text-faded">${esc(rec.model.summary)}</p>
      <p class="border border-rule bg-paper px-3 py-2 dsnote">NOTE: no score on this site has been spot-checked on our own hardware yet — every figure is vendor-reported or aggregated from public benchmarks.</p>
      ${warnings ? `<div class="space-y-2"><p class="dslabel">Known issues at this quantization band</p>${warnings}</div>` : '<p class="font-mono text-[12px] text-ok">NO MAJOR KNOWN ISSUES FOR THIS WORKLOAD AT THIS QUANTIZATION BAND.</p>'}
    </div>
  </article>`;
}

export function initPicker(): void {
  const dataEl = document.getElementById('picker-data');
  const results = document.getElementById('picker-results');
  if (!dataEl || !results) return;
  const data = JSON.parse(dataEl.textContent ?? '{}') as PickerData;

  const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T | null;
  const platformSel = $<HTMLSelectElement>('p-platform');
  const ramSel = $<HTMLSelectElement>('p-ram');
  const vramSel = $<HTMLSelectElement>('p-vram');
  const ceilingSel = $<HTMLSelectElement>('p-ceiling');
  const ctxSel = $<HTMLSelectElement>('p-ctx');
  const vramWrap = $('p-vram-wrap');
  const ceilingWrap = $('p-ceiling-wrap');
  const budgetLine = $('p-budget');
  const noteLine = $('p-preset-note');
  const chipSel = $('p-chip') as HTMLSelectElement | null;
  const chipWrap = $('p-chip-wrap');
  const sysSel = $('p-sysmem') as HTMLSelectElement | null;
  const sysWrap = $('p-sysmem-wrap');
  const ramLabel = $('p-ram-label');
  const compareSel = $('p-compare') as HTMLSelectElement | null;
  const compareOut = $('p-compare-out');

  let task: string | null = null;
  /**
   * Whether the reader has actually described a machine, by picking a preset or
   * editing a control. Everything in the panel has a value from the first
   * paint, so without this the panel would look answered before it was asked —
   * and the green would mean nothing.
   */
  let described = false;

  const machine = (): Machine => {
    const platform = (platformSel?.value ?? 'unified') as PlatformId;
    const ramGb = Number(ramSel?.value ?? 32);
    const vramGb = platform === 'discrete' ? Number(vramSel?.value ?? 12) : undefined;
    const override = ceilingSel?.value ? Number(ceilingSel.value) : undefined;
    const chipId = chipSel?.value || undefined;
    const systemMemoryId = sysSel?.value || undefined;
    return {
      platform,
      ramGb,
      vramGb,
      ceilingOverrideGb: override,
      chipId,
      systemMemoryId,
      presetId: presetId ?? undefined,
    };
  };

  /**
   * The RAM select's options depend on the chip, so its value can only be set
   * once the list is right. Every writer comes through here with the capacity
   * it WANTS, instead of assigning .value against whatever list happens to be
   * showing: assigning an absent value silently leaves the select empty, and
   * Number('') === 0 then clamps to the smallest option. That is how picking an
   * M5 Max 128GB produced a 48GB machine — 48 being the first capacity that
   * chip offers.
   *
   * Callers must set platform and chip first. Those decide which capacities
   * exist, so ordering is not a style question here.
   */
  const setRam = (wantGb: number) => {
    if (!ramSel) return;
    const probe: Machine = {
      platform: (platformSel?.value ?? 'unified') as PlatformId,
      ramGb: wantGb,
      chipId: chipSel?.value || undefined,
    };
    const allowed = ramOptionsFor(probe, data.hardware);
    if (allowed.length === 0) return;
    const shown = [...ramSel.options].map((o) => Number(o.value));
    if (allowed.length !== shown.length || allowed.some((g, i) => g !== shown[i])) {
      ramSel.innerHTML = allowed.map((g) => `<option value="${g}">${g} GB</option>`).join('');
    }
    const nearest = allowed.reduce((a, b) => (Math.abs(b - wantGb) < Math.abs(a - wantGb) ? b : a));
    ramSel.value = String(allowed.includes(wantGb) ? wantGb : nearest);
  };

  /** Only unified memory has an adjustable ceiling; only discrete has VRAM. */
  const syncControls = () => {
    const m = machine();
    const platform = data.hardware.platforms.find((p) => p.id === m.platform);
    if (vramWrap) vramWrap.hidden = m.platform !== 'discrete';
    // Only Apple ships many chips behind one platform. Strix Halo and DGX Spark
    // are one chip each, and a discrete card names its own silicon.
    if (chipWrap) chipWrap.hidden = m.platform !== 'unified';
    // Only where the CPU can actually get work. On unified memory there is one
    // pool and nowhere to spill to, so the control would imply a choice that
    // does not exist.
    const cpuWorks = m.platform === 'discrete' || m.platform === 'cpu';
    if (sysWrap) sysWrap.hidden = !cpuWorks;
    // The same words mean different things on the two platforms: on a Mac this
    // select IS the memory the model runs in, on a machine with a card it is
    // the second, slower tier the card overflows into.
    if (ramLabel) {
      ramLabel.textContent = m.platform === 'discrete' || m.platform === 'cpu' ? 'System RAM' : 'Unified memory';
    }

    // Green the controls that now hold a real answer. Per control, not per
    // panel: a select still on "Auto — from workload" or "Default (untouched)"
    // has an empty value and has been told nothing, so it stays quiet. Those
    // placeholders are exactly the ones that carry value="".
    for (const el of [platformSel, ramSel, vramSel, sysSel, ctxSel, chipSel, ceilingSel]) {
      el?.classList.toggle('is-set', described && el.value !== '');
    }

    // A chip lists the capacities it was sold with, so an M1 Max cannot be
    // configured with 48GB. The value here is already valid for the previous
    // chip, so it is the right thing to clamp from.
    setRam(m.ramGb);
    // Re-read after the rebuild: clamping may have moved the capacity, and the
    // ceiling below is computed from it. Reading the stale value offered a
    // 64GB machine a "384GB default", and where the clamp emptied the select
    // Number('') === 0 left the raise options gone entirely.
    const now = machine();
    const adjustable = platform?.ceiling.overridable === true;
    if (ceilingWrap) ceilingWrap.hidden = !adjustable;
    if (adjustable && ceilingSel) {
      const keep = ceilingSel.value;
      const dflt = Math.round((platform!.ceiling.value ?? 0.75) * now.ramGb);
      // Built from the gap between default and the safe maximum, not from the
      // shared RAM ladder — that offered one step on most machines and made it
      // the whole of memory, the setting the advice text warns against.
      const steps = raiseSteps(now, data.hardware);
      // No option calls itself safe. Nobody can: Apple documents this setting
      // nowhere, so there is no guidance to appeal to and this site is not an
      // authority on what someone else's machine survives. Each option states
      // the fact instead — what it leaves the system — and says when it is past
      // the only published recommendation there is.
      const published = publishedCeilingGb(now, data.hardware);
      ceilingSel.innerHTML =
        `<option value="">Default — ${dflt}GB (untouched)</option>` +
        steps
          .map((g) => {
            const left = Math.round((now.ramGb - g) * 10) / 10;
            const beyond = published !== null && g > published ? ', past published advice' : '';
            return `<option value="${g}">${g}GB — leaves ${left}GB${beyond}</option>`;
          })
          .join('');
      if (steps.some((g) => String(g) === keep)) ceilingSel.value = keep;
    }
  };

  const render = () => {
    syncControls();
    document.querySelectorAll<HTMLButtonElement>('[data-task]').forEach((b) => {
      b.classList.toggle('is-selected', b.dataset.task === task);
    });

    const m = machine();
    const budget = budgetFor(m, data.hardware);
    const source = budgetSource(m, budget);
    const bandwidth = bandwidthFor(m, data.hardware);
    const systemBandwidth = systemBandwidthFor(m, data.hardware);
    const prefill = prefillFor(m, data.hardware);
    if (budgetLine) {
      // budgetSource, the same explanation the card's RUNS AT line uses. This
      // line used to build its own, naming the PLATFORM as the cap and quoting
      // the ceiling — so an M1 Max 32GB read "24GB at full speed — capped by
      // unified memory (apple silicon) at 24GB": the wrong culprit, and the
      // same number twice. Unified memory is not the cap; the macOS wired limit
      // is, and capLabel has said so since it was added.
      // A machine with a graphics card is told what its CPU adds. A machine
      // without one has to be told why nothing is added, in the same breath and
      // the same place — otherwise a Mac owner who has read about splitting
      // models across CPU and GPU just sees the feature missing and assumes the
      // site forgot. It did not: on one pool the split is possible and pointless.
      const spill = budget.offloadGb > budget.gb
        ? ` Plus ${formatMemory(budget.offloadGb - budget.gb)} of system RAM your CPU can run the overflow from, more slowly.`
        : hasSeparateSystemMemory(m.platform)
          ? ''
          : ' Your CPU shares this same memory rather than adding a second pool, and reads it more slowly than the GPU does — so handing it layers would be slower without holding more. To fit a bigger model here, raise the cap rather than split the model.';
      budgetLine.textContent = `BUDGET: ${formatMemory(budget.gb)} at full speed — ${source}.${spill}`;
    }
    if (noteLine) {
      // Same "source" link the glossary uses, rather than a bare URL in the prose.
      noteLine.innerHTML = presetNote
        ? esc(presetNote.text) +
          (presetNote.url
            ? ` <a href="${esc(presetNote.url)}" class="underline underline-offset-2 hover:text-serious">source</a>`
            : '')
        : '';
      noteLine.hidden = !presetNote;
    }

    if (!task) {
      results.innerHTML =
        '<p class="font-mono text-sm text-faded">Awaiting input. Choose a workload above.</p>';
      return;
    }

    const ctxOverride = ctxSel?.value ? Number(ctxSel.value) : undefined;
    const out = pickModels(m, task, data, ctxOverride);
    const notes: string[] = [];
    if (out.fallbackFrom) {
      notes.push(
        `<p class="border-l-2 border-warn bg-panel py-2 pl-3 pr-3 text-sm">Nothing runs this workload at ${ctxLabel(out.fallbackFrom)} context on ${budget.gb}GB. At ${ctxLabel(out.contextTokens)} these do —</p>`,
      );
    }
    if (out.ceilingAdvice) {
      const a = out.ceilingAdvice;
      notes.push(
        `<div class="border border-rule bg-panel px-3 py-2 text-sm">
          <p class="dslabel">${esc(a.label)}</p>
          ${out.ceilingGain ? `<p class="mt-1">${esc(out.ceilingGain)}</p>` : ''}
          <p class="mt-1 text-faded">${esc(a.newbieExplainer)}</p>
          ${a.command ? `<p class="mt-1 font-mono text-[12px]">${esc(commandFor(a.command, m, data))}</p>` : ''}
          ${a.caveat ? `<p class="mt-1 dsnote">${esc(a.caveat)}</p>` : ''}
          ${a.citation ? `<p class="mt-1 font-mono text-[12px]"><a href="${esc(a.citation)}" class="text-faded underline decoration-dotted underline-offset-2 hover:text-serious" rel="noopener">Where this comes from</a></p>` : ''}
        </div>`,
      );
    }

    const offload = out.offloadOptions.length
      ? `<div class="border border-rule bg-panel px-3 py-2">
          <p class="dslabel">Also possible, splitting the model with your CPU</p>
          <p class="mt-1 dsnote">Slower: your CPU runs whatever the card cannot hold, out of system RAM.</p>
          <ul class="mt-2 space-y-1">
            ${out.offloadOptions
              .map(
                (o) => `<li class="font-mono text-[12px]">
                  <a href="/models/${esc(o.model.id)}" class="underline decoration-dotted underline-offset-2 hover:text-serious">${esc(o.model.name)}</a>
                  <span class="text-faded"> · ${esc(o.bandLabel.split(' ')[0]!)} · ${formatMemory(o.fit.demandGb)} —
                  ${Math.round(o.fit.residentFraction * 100)}% on the card, ${formatMemory(o.fit.spilledGb)} run by your CPU</span>
                </li>`,
              )
              .join('')}
          </ul>
        </div>`
      : '';

    results.innerHTML =
      notes.join('') +
      (out.recommendations.length > 0
        ? out.recommendations
            .map((r, i) =>
              recHtml(
                r,
                i,
                source,
                bandwidth,
                prefill,
                systemBandwidth,
                splitNote(r),
                overheadNote(r, data.hardware),
                m.platform === 'unified',
                i === 0
                  ? weakBestNote(
                      r,
                      out.betterThanWeak,
                      generationSpeed(r.fit.weightsGb, r.fit.kvGb, r.model, r.fit, bandwidth, systemBandwidth, data.hardware, m.platform),
                      out.betterThanWeak
                        ? generationSpeed(
                            out.betterThanWeak.fit.weightsGb,
                            out.betterThanWeak.fit.kvGb,
                            out.betterThanWeak.model,
                            out.betterThanWeak.fit,
                            bandwidth,
                            systemBandwidth,
                            data.hardware,
                            m.platform,
                          )
                        : null,
                    )
                  : '',
                data.hardware,
                m.platform,
              ),
            )
            .join('')
        : '<p class="font-mono text-sm text-faded">NOTHING RATED RUNS ON THIS MACHINE FOR THIS WORKLOAD — an honest gap, not a bug.</p>') +
      // After the cards: it talks about "the range above", so it has to follow
      // the speeds it is qualifying, not precede them.
      (out.recommendations.length > 0 ? mlxNote(m) : '') +
      offload +
      (out.recommendations[0] ? capacityCompare(out.recommendations[0], m, data, out.contextTokens, data.tasks.find((x) => x.id === task)!) : '');

    // Machine B, when one is chosen. Built from its preset alone — A stays the
    // configurable one — and priced at A's context so the two are like for like.
    if (compareOut) {
      const bPreset = data.hardware.presets.find((p) => p.id === compareSel?.value);
      const c = bPreset
        ? compareMachines(
            m,
            { platform: bPreset.platform, ramGb: bPreset.ramGb, vramGb: bPreset.vramGb, chipId: bPreset.chipId, presetId: bPreset.id },
            task,
            data,
            ctxOverride,
          )
        : null;
      compareOut.innerHTML = c && bPreset ? compareBlock(c, bPreset.label) : '';
    }

    const url = new URL(location.href);
    url.searchParams.set('platform', m.platform);
    url.searchParams.set('ram', String(m.ramGb));
    if (m.vramGb) url.searchParams.set('vram', String(m.vramGb));
    else url.searchParams.delete('vram');
    if (m.ceilingOverrideGb) url.searchParams.set('ceiling', String(m.ceilingOverrideGb));
    else url.searchParams.delete('ceiling');
    if (m.chipId) url.searchParams.set('chip', m.chipId);
    else url.searchParams.delete('chip');
    // Same reason as `preset` below: this decides how fast the CPU runs any
    // offloaded layer, so a link without it hands the recipient a different
    // machine's speed. Shipped missing when the control was added.
    if (m.systemMemoryId) url.searchParams.set('sysmem', m.systemMemoryId);
    else url.searchParams.delete('sysmem');
    // Without this a shared link loses which 24GB card it meant, and the speed
    // estimate silently becomes some other machine's.
    if (m.presetId) url.searchParams.set('preset', m.presetId);
    else url.searchParams.delete('preset');
    if (compareSel?.value) url.searchParams.set('vs', compareSel.value);
    else url.searchParams.delete('vs');
    if (ctxOverride) url.searchParams.set('ctx', String(ctxOverride));
    else url.searchParams.delete('ctx');
    url.searchParams.set('task', task);
    history.replaceState(null, '', url);
  };

  document.querySelectorAll<HTMLButtonElement>('[data-task]').forEach((b) =>
    b.addEventListener('click', () => {
      task = b.dataset.task ?? null;
      render();
    }),
  );
  // One machine spread over six vendor dropdowns, so a pick in any row has to
  // empty the other five — otherwise the page claims two machines at once.
  let presetNote: { text: string; url?: string } | null = null;
  let presetId: string | null = null;
  const presetSels = [...document.querySelectorAll<HTMLSelectElement>('[data-preset-group]')];
  const clearPresets = (except?: HTMLSelectElement) => {
    for (const s of presetSels) if (s !== except) s.value = '';
    presetNote = null; // the caveat belongs to a preset, not to the numbers
    presetId = null; // and so does the bandwidth: hand-edited is no longer that machine
  };
  for (const sel of presetSels) {
    sel.addEventListener('change', () => {
      const preset = data.hardware.presets.find((p) => p.id === sel.value);
      if (!preset) return render(); // back to the placeholder
      if (platformSel) platformSel.value = preset.platform;
      if (vramSel && preset.vramGb) vramSel.value = String(preset.vramGb);
      if (ceilingSel) ceilingSel.value = '';
      if (chipSel) chipSel.value = preset.chipId ?? ''; // show the chip the preset implies
      // The era the card shipped into: a Tesla P40 sat beside DDR4 and a 5090
      // sits beside DDR5, and that gap is the speed of every offloaded layer.
      if (sysSel) sysSel.value = preset.systemMemoryId ?? '';
      // Capacity LAST: the chip above decides which capacities exist.
      setRam(preset.ramGb);
      // Assigning .value fires no change event, so none of the above re-enters
      // this handler or trips the manual-edit listener below.
      clearPresets(sel); // clears presetNote, so set it after
      presetNote = preset.note ? { text: preset.note, url: preset.noteUrl } : null;
      presetId = preset.id;
      described = true;
      render();
    });
  }
  for (const el of [platformSel, ramSel, vramSel, ceilingSel, ctxSel, chipSel, sysSel]) {
    el?.addEventListener('change', () => {
      clearPresets(); // hand-edited: no preset describes this machine any more
      described = true; // but it is still a machine they have now described
      render();
    });
  }
  // Deliberately NOT in the list above: choosing something to compare against
  // says nothing about the machine you own, so it must not clear A's preset.
  compareSel?.addEventListener('change', () => render());

  const params = new URLSearchParams(location.search);
  if (platformSel && params.get('platform')) platformSel.value = params.get('platform')!;

  if (vramSel && params.get('vram')) vramSel.value = params.get('vram')!;
  if (ctxSel && params.get('ctx')) ctxSel.value = params.get('ctx')!;
  // Validated against the table, the way `preset` is: an unknown value would
  // otherwise fall back to a different machine's figure without saying so.
  const savedChip = params.get('chip');
  if (chipSel && savedChip && data.hardware.chips.some((c) => c.id === savedChip)) {
    chipSel.value = savedChip;
  }
  const savedSys = params.get('sysmem');
  if (sysSel && savedSys && data.hardware.systemMemory.some((x) => x.id === savedSys)) {
    sysSel.value = savedSys;
  }
  const savedPreset = params.get('preset');
  if (savedPreset && data.hardware.presets.some((p) => p.id === savedPreset)) {
    presetId = savedPreset;
    const sel = presetSels.find((x) => [...x.options].some((o) => o.value === savedPreset));
    if (sel) sel.value = savedPreset;
  }
  const savedVs = params.get('vs');
  if (compareSel && savedVs && data.hardware.presets.some((p) => p.id === savedVs)) {
    compareSel.value = savedVs;
  }
  // After platform and chip, for the same reason the preset handler does.
  if (params.get('ram')) setRam(Number(params.get('ram')));
  // A shared link describes a machine just as much as clicking does — the
  // reader arriving on it should see the same answered panel the sender saw.
  described = ['platform', 'ram', 'vram', 'chip', 'preset', 'sysmem', 'ceiling'].some((k) =>
    params.get(k),
  );
  task = params.get('task');
  syncControls();
  if (ceilingSel && params.get('ceiling')) ceilingSel.value = params.get('ceiling')!;
  render();
}
