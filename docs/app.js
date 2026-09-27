import { $, esc, fail, int, kpis, legend, load, num, pct, select, table, xy } from './kit.js';

try {
  const rows = await load();
  const B = rows[0].bins.right.length;
  const total = (r) => r.bins.right.reduce((a, b) => a + b, 0) + r.bins.wrong.reduce((a, b) => a + b, 0);
  const at = (r, i) => {   // flag when P(right) < i / B
    let f = 0, w = 0;
    for (let j = 0; j < i; j++) { f += r.bins.right[j] + r.bins.wrong[j]; w += r.bins.wrong[j]; }
    return { flagged: f / total(r), precision: f ? w / f : null, recall: w / r.wrong };
  };
  const best = rows.reduce((a, b) => (b.auroc > a.auroc ? b : a));
  kpis($('#kpis'), [
    { label: 'Questions per model', value: int(total(rows[0])), note: 'GSM8K, LegalBench, MATH, MedQA, MMLU, OpenBookQA' },
    { label: 'Answering models', value: String(rows.length), note: 'each reviewed by a panel of three others' },
    { label: 'Ranking quality (AUROC)', value: `${num(Math.min(...rows.map((r) => r.auroc)), 2)}–${num(best.auroc, 2)}`, note: 'how well P(right) separates right from wrong' },
    { label: 'Calibration error', value: `${num(Math.min(...rows.map((r) => r.ece)), 3)}–${num(Math.max(...rows.map((r) => r.ece)), 3)}`, note: 'expected calibration error, 10 bins' },
  ]);

  let r = rows[0];
  const slider = $('#t');
  function drawFlag() {
    const i = +slider.value, m = at(r, i);
    $('#tVal').textContent = (i / B).toFixed(3);
    $('#flagOut').innerHTML = `<div>Answers flagged<b>${pct(m.flagged)}</b><span class="muted small">${int(m.flagged * total(r))} of ${int(total(r))}</span></div>` +
      `<div>Flagged that were wrong<b>${pct(m.precision)}</b><span class="muted small">precision; the model is wrong on ${pct(1 - r.accuracy)} overall</span></div>` +
      `<div>Wrong answers caught<b>${pct(m.recall)}</b><span class="muted small">recall, of ${int(r.wrong)} wrong answers</span></div>`;
    const curve = [];
    for (let j = 1; j < B; j++) { const x = at(r, j); if (x.precision != null) curve.push({ x: x.recall, y: x.precision }); }
    const series = [
      { name: 'arbiter at every threshold', color: 'var(--accent)', points: curve },
      { name: 'this threshold', color: 'var(--accent)', line: false, dots: true, points: m.precision != null ? [{ x: m.recall, y: m.precision, r: 7, label: 'arbiter', keep: true }] : [] },
      { name: 'majority of critics object', color: 'var(--c5)', line: false, dots: true, points: [{ x: r.majority.recall, y: r.majority.precision, label: 'majority' }] },
      { name: `best single critic (${r.panel[0]})`, color: 'var(--c3)', line: false, dots: true, points: [{ x: r.best_critic.recall, y: r.best_critic.precision, label: 'one critic' }] },
    ];
    xy($('#pr'), { label: 'Precision against recall', height: 280, series, x: { min: 0, max: 1, label: 'wrong answers caught (recall)', fmt: (v) => `${Math.round(v * 100)}%` }, y: { min: 0, max: 1, fmt: (v) => `${Math.round(v * 100)}%`, label: 'flagged that were wrong (precision)' } });
    legend($('#prKey'), series.filter((s) => s.name !== 'this threshold'));
  }
  function drawPatterns() {
    $('#patSub').innerHTML = `Each critic either passes the answer (✓) or objects (✗). Critics, in order: ${r.panel.map((c) => `<b>${esc(c)}</b>`).join(', ')}. The arbiter's average P(right) for each pattern against how often those answers were actually right.`;
    const mark = (v) => [...v].map((c) => (c === 'v' ? '✓' : '✗')).join(' ');
    const pats = [...r.patterns].sort((a, b) => b.p - a.p);
    table($('#patterns'), [
      { key: 'verdicts', label: 'Critics', fmt: mark },
      { key: 'n', label: 'Answers', num: true, fmt: int },
      { key: 'p', label: 'Arbiter P(right)', num: true, fmt: (v) => pct(v) },
      { key: 'acc', label: 'Actually right', num: true, fmt: (v) => pct(v) },
      { key: 'gap', label: 'Gap', num: true, fmt: (v) => `${v > 0 ? '+' : ''}${num(100 * v, 1)}` },
    ], pats.map((p) => ({ ...p, acc: p.right / p.n, gap: p.right / p.n - p.p })), { cls: (p) => (Math.abs(p.gap) >= 0.1 ? 'bad' : '') });
    const worst = pats.reduce((a, b) => (Math.abs(b.right / b.n - b.p) > Math.abs(a.right / a.n - a.p) ? b : a));
    $('#patNote').textContent = `Highlighted: off by 10 points or more. Naive Bayes treats the critics as independent, but they are models that fail on the same questions, so objections that agree get counted as more evidence than they are. Here the largest gap is ${mark(worst.verdicts)}: ${pct(worst.p)} predicted, ${pct(worst.right / worst.n)} observed, on ${int(worst.n)} answers.`;
  }
  slider.oninput = drawFlag;
  select($('#who'), rows.map((x, i) => [i, `${x.answerer} (${pct(x.accuracy)} right)`]), 0, (i) => { r = rows[+i]; drawFlag(); drawPatterns(); });

  const pr = (x) => `${pct(x.precision, 0)} / ${pct(x.recall, 0)}`;
  table($('#all'), [
    { key: 'answerer', label: 'Answers from' },
    { key: 'accuracy', label: 'Accuracy', num: true, fmt: (v) => pct(v) },
    { key: 'auroc', label: 'AUROC', num: true, fmt: (v) => num(v, 3) },
    { key: 'ece', label: 'ECE', num: true, fmt: (v) => num(v, 3) },
    { key: 'arbiter', label: 'Arbiter P / R', num: true, fmt: pr },
    { key: 'majority', label: 'Majority P / R', num: true, fmt: pr },
    { key: 'best_critic', label: 'One critic P / R', num: true, fmt: pr },
  ], rows);
} catch (err) {
  fail(err);
}
