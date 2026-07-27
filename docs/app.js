/**
 * app.js — the PyMAPF playground.
 *
 * Responsibilities, in order of interest:
 *  1. draw the map, the live search and the plan playback on a canvas;
 *  2. drive two interchangeable solver engines (Pyodide worker / in-page JS);
 *  3. run user-written Python and surface whatever it produced;
 *  4. run a benchmark sweep and chart it as inline SVG.
 *
 * Colors come from the CSS custom properties, which mirror pymapf/viz/theme.py,
 * so the canvas, the charts and the matplotlib gallery agree.
 */

import {
  SCENARIOS, buildScenario, problemFromState, solveGenerator, gridFromMatrix, isFree,
} from './mapf.js';

/* ------------------------------------------------------------- helpers --- */

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = () => [1, 2, 3, 4, 5, 6, 7, 8].map((i) => css(`--s${i}`));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmtMs = (seconds) => (seconds >= 1 ? `${seconds.toFixed(2)} s` : `${(seconds * 1000).toFixed(1)} ms`);

const ALGORITHM_BLURBS = {
  cbs: 'Provably optimal. Exponential in the number of conflicts — give it a budget.',
  wcbs: 'Focal search: cost stays within w × optimal, usually far faster.',
  prioritized: 'One agent at a time. Fastest, but can fail on solvable instances.',
};

/* --------------------------------------------------------------- state --- */

const state = {
  scenario: 'warehouse',
  seed: 0,
  agents: 6,
  algorithm: 'wcbs',
  weight: 1.5,
  heuristic: 'manhattan',
  engine: 'python',
  matrix: [],
  agentList: [],
  solution: null,
  // live search view
  searchPaths: null,
  conflicts: [],
  conflictCount: 0,
  expansions: 0,
  running: false,
  // playback
  t: 0,
  playing: false,
  lastFrame: 0,
};

/* ------------------------------------------------------------- drawing --- */

const canvas = $('board');
const ctx = canvas.getContext('2d');

function layout() {
  const rows = state.matrix.length;
  const cols = state.matrix[0]?.length || 1;
  const dpr = window.devicePixelRatio || 1;
  const maxWidth = canvas.parentElement.clientWidth - 24;
  const cell = clamp(Math.floor(Math.min(maxWidth / cols, 560 / rows)), 8, 46);
  const width = cell * cols;
  const height = cell * rows;

  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { cell, rows, cols, width, height };
}

const ease = (f) => f * f * (3 - 2 * f);

function positionAt(path, t) {
  // Total in t: a clock jump (tab restore, a resize mid-capture) must never
  // index outside the path.
  const last = path.length - 1;
  const clamped = Number.isFinite(t) ? Math.max(0, Math.min(t, last)) : 0;
  const whole = Math.floor(clamped);
  const a = path[whole];
  const b = path[Math.min(whole + 1, last)];
  const k = ease(clamped - whole);
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
}

function draw() {
  const { cell, rows, cols, width, height } = layout();
  const colors = SERIES();
  const center = (v) => (v + 0.5) * cell;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = css('--surface-1');
  ctx.fillRect(0, 0, width, height);

  // obstacles
  ctx.fillStyle = css('--obstacle');
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      if (state.matrix[r][c]) ctx.fillRect(c * cell, r * cell, cell, cell);
    }
  }

  // lattice
  ctx.strokeStyle = css('--grid');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let c = 0; c <= cols; c += 1) { ctx.moveTo(c * cell + 0.5, 0); ctx.lineTo(c * cell + 0.5, height); }
  for (let r = 0; r <= rows; r += 1) { ctx.moveTo(0, r * cell + 0.5); ctx.lineTo(width, r * cell + 0.5); }
  ctx.stroke();

  const paths = state.solution?.paths || state.searchPaths;
  const names = paths ? Object.keys(paths) : state.agentList.map((a) => a.name);
  const colorOf = (name) => colors[names.indexOf(name) % colors.length];

  // goal rings, always visible so intent is readable before a plan exists
  state.agentList.forEach((agent, index) => {
    const color = colors[index % colors.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.6, cell * 0.09);
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.arc(center(agent.goal[1]), center(agent.goal[0]), cell * 0.3, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
  });

  if (paths) {
    const railSpan = cell * 0.26;
    names.forEach((name, index) => {
      const path = paths[name];
      if (!path?.length) return;
      const offset = names.length > 1
        ? (index / (names.length - 1) - 0.5) * railSpan
        : 0;
      const color = colorOf(name);

      // underlay in the surface color keeps crossing rails legible
      for (const [stroke, lineWidth, alpha] of [
        [css('--surface-1'), Math.max(3, cell * 0.22), 1],
        [color, Math.max(1.8, cell * 0.11), state.solution ? 0.95 : 0.6],
      ]) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = lineWidth;
        ctx.globalAlpha = alpha;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        path.forEach(([r, c], i) => {
          const x = center(c) + offset;
          const y = center(r) + offset;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });

    // agents at the current timestep
    if (state.solution) {
      names.forEach((name) => {
        const path = paths[name];
        if (!path?.length) return;
        const [r, c] = positionAt(path, state.t);
        const color = colorOf(name);
        const arrived = state.t >= path.length - 1;

        ctx.beginPath();
        ctx.arc(center(c), center(r), cell * 0.31, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = Math.max(1.5, cell * 0.07);
        ctx.strokeStyle = css('--surface-1');
        ctx.stroke();

        if (arrived) {   // a subtle halo marks "done", not color alone
          ctx.beginPath();
          ctx.arc(center(c), center(r), cell * 0.44, 0, Math.PI * 2);
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.35;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }

        if (cell >= 16) {
          ctx.fillStyle = css('--surface-1');
          ctx.font = `600 ${Math.round(cell * 0.36)}px ${css('--sans') || 'sans-serif'}`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(name, center(c), center(r) + 0.5);
        }
      });
    }
  }

  // conflicts found by the live search
  ctx.strokeStyle = css('--critical');
  ctx.lineWidth = Math.max(2, cell * 0.12);
  ctx.lineCap = 'round';
  for (const [r, c] of state.conflicts.slice(-4)) {
    const x = center(c); const y = center(r); const s = cell * 0.24;
    ctx.beginPath();
    ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
    ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s);
    ctx.stroke();
  }
}

/* ------------------------------------------------------------ playback --- */

function tick(now) {
  if (!state.playing) return;
  const horizon = state.solution ? state.solution.makespan : 0;
  // Clamped so a backwards or very large frame delta cannot throw the playhead
  // out of the plan.
  const dt = Math.min(0.1, Math.max(0, (now - state.lastFrame) / 1000));
  state.lastFrame = now;
  state.t = Math.max(0, state.t + dt * 3.2);   // ~3 timesteps per second
  if (state.t >= horizon) { state.t = horizon; state.playing = false; $('play').textContent = '⏵ Play'; }
  $('timeline').value = String(Math.floor(state.t));
  $('clock').textContent = `t = ${Math.floor(state.t)} / ${horizon}`;
  draw();
  if (state.playing) requestAnimationFrame(tick);
}

function play() {
  if (!state.solution) return;
  if (state.t >= state.solution.makespan) state.t = 0;
  state.playing = true;
  state.lastFrame = performance.now();
  $('play').textContent = '⏸ Pause';
  requestAnimationFrame(tick);
}

/* ------------------------------------------------------------- logging --- */

function log(text, className = '') {
  const element = $('log');
  const line = document.createElement('div');
  if (className) line.className = className;
  line.textContent = text;
  element.appendChild(line);
  element.scrollTop = element.scrollHeight;
  while (element.childElementCount > 260) element.removeChild(element.firstChild);
}

function handleEvent(event) {
  if (event.kind === 'expand') {
    state.expansions += 1;
    if (event.paths) state.searchPaths = event.paths;
    if (state.expansions % 3 === 1 || event.node < 6) {
      log(`node ${event.node}  cost ${event.cost}  open ${event.open}`);
    }
  } else if (event.kind === 'conflict') {
    state.conflictCount += 1;
    if (event.cell) state.conflicts.push(event.cell);
    log(`✕ ${event.type} conflict: ${event.a} ↔ ${event.b} at t=${event.t}`, 'conflict');
  } else if (event.kind === 'branch') {
    if (state.conflictCount < 12) log(`  ↳ constrain ${event.agent} (cost ${event.cost})`, 'branch');
  } else if (event.kind === 'agent_planned') {
    log(`planned ${event.agent}: ${event.cost} steps`);
    if (event.path) {
      state.searchPaths = { ...(state.searchPaths || {}), [event.agent]: event.path };
    }
  } else if (event.kind === 'root') {
    log(`root node: ${event.agents.join(', ')}`);
  } else if (event.kind === 'failed') {
    log(`failed — ${event.reason}`, 'conflict');
  }
}

/* ------------------------------------------------------------- metrics --- */

function showMetrics(solution, extra = {}) {
  const set = (id, value, tone = '') => {
    const element = $(id);
    element.textContent = value;
    element.className = `value ${tone}`;
  };
  if (!solution) {
    ['m-cost', 'm-makespan', 'm-valid'].forEach((id) => set(id, '—'));
    set('m-expansions', String(extra.expansions ?? state.expansions));
    set('m-conflicts', String(extra.conflicts ?? state.conflictCount));
    set('m-runtime', extra.runtime != null ? fmtMs(extra.runtime) : '—');
    return;
  }
  set('m-cost', String(solution.sumOfCosts));
  set('m-makespan', String(solution.makespan));
  set('m-expansions', String(solution.expansions));
  set('m-conflicts', String(state.conflictCount));
  set('m-runtime', fmtMs(solution.runtime));
  set('m-valid', solution.valid ? 'yes' : 'no', solution.valid ? 'good' : 'bad');
}

/* -------------------------------------------------------------- engine --- */

const engine = {
  worker: null,
  ready: false,
  failed: false,
  pendingSolve: null,
  onEvent: handleEvent,
};

function setEngineBadge(status, label) {
  $('engine-dot').className = `dot ${status}`;
  $('engine-label').textContent = label;
}

function bootWorker() {
  let worker;
  try {
    worker = new Worker('worker.js', { type: 'module' });
  } catch (error) {
    engine.failed = true;
    setEngineBadge('error', 'JS engine (workers unavailable)');
    selectEngine('js');
    return;
  }
  engine.worker = worker;

  worker.onmessage = ({ data }) => {
    if (data.type === 'status') {
      setEngineBadge('loading', `${data.detail}…`);
    } else if (data.type === 'ready') {
      engine.ready = true;
      setEngineBadge('live', `pymapf ${data.version} · Python ${data.python} (WASM)`);
      $('code-output').textContent =
        `PyMAPF ${data.version} on Python ${data.python}\nSolvers: ${data.solvers.join(', ')}\n\nReady — hit “Run Python”.`;
      if (engine.pendingSolve) { const spec = engine.pendingSolve; engine.pendingSolve = null; runPython(spec); }
    } else if (data.type === 'event') {
      engine.onEvent(data.event);
      draw();
    } else if (data.type === 'result') {
      finishSolve(data.result && data.result.solved ? {
        paths: data.result.paths,
        sumOfCosts: data.result.sum_of_costs,
        makespan: data.result.makespan,
        expansions: data.result.expansions,
        runtime: data.result.runtime,
        valid: data.result.valid,
        algorithm: data.result.algorithm,
      } : null, data.result?.runtime);
    } else if (data.type === 'bench-row') {
      benchmarkRow(data.row);
    } else if (data.type === 'bench-done') {
      benchmarkDone(data.rows);
    } else if (data.type === 'exec-done') {
      showExecResult(data);
    } else if (data.type === 'error') {
      if (data.context === 'init') {
        engine.failed = true;
        setEngineBadge('error', 'JS engine (Python runtime unavailable)');
        selectEngine('js');
        $('code-output').textContent =
          `The Python runtime could not be loaded:\n  ${data.message}\n\n` +
          'The Visual and Benchmark tabs still work — they fall back to the ' +
          'JavaScript port of the same solvers.';
      } else {
        log(`engine error: ${data.message}`, 'conflict');
        state.running = false;
        $('run').disabled = false;
      }
    }
  };

  setEngineBadge('loading', 'loading Python runtime…');
  worker.postMessage({ type: 'init' });
}

function currentSpec() {
  return {
    matrix: state.matrix,
    agents: state.agentList,
    algorithm: state.algorithm,
    options: {
      heuristic: state.heuristic,
      weight: state.weight,
      time_limit: 8.0,
    },
  };
}

function beginSolve() {
  state.running = true;
  state.solution = null;
  state.searchPaths = null;
  state.conflicts = [];
  state.conflictCount = 0;
  state.expansions = 0;
  state.t = 0;
  state.playing = false;
  $('log').innerHTML = '';
  $('run').disabled = true;
  $('play').disabled = true;
  $('timeline').disabled = true;
  showMetrics(null);
  draw();
}

function finishSolve(solution, runtime) {
  state.running = false;
  state.solution = solution;
  state.searchPaths = null;
  state.conflicts = [];
  $('run').disabled = false;

  if (!solution) {
    log('no solution returned', 'conflict');
    showMetrics(null, { runtime });
    draw();
    return;
  }
  log(`solved — cost ${solution.sumOfCosts}, makespan ${solution.makespan}, ${fmtMs(solution.runtime)}`, 'solved');
  showMetrics(solution);
  $('timeline').max = String(solution.makespan);
  $('timeline').value = '0';
  $('timeline').disabled = false;
  $('play').disabled = false;
  updateSnippet();
  play();
}

/** In-page engine: pump the generator with a frame budget so the UI stays live. */
function runJs() {
  const generator = solveGenerator(problemFromState({ matrix: state.matrix, agents: state.agentList }),
    state.algorithm, {
      heuristic: state.heuristic,
      weight: state.weight,
      timeLimit: 8,
    });
  let solution = null;

  const pump = () => {
    const deadline = performance.now() + 10;   // ~10 ms of work per frame
    for (;;) {
      const step = generator.next();
      if (step.done) {
        finishSolve(solution ? {
          paths: solution.paths,
          sumOfCosts: solution.sumOfCosts,
          makespan: solution.makespan,
          expansions: solution.expansions,
          runtime: solution.runtime,
          valid: solution.valid,
          algorithm: solution.algorithm,
        } : null);
        return;
      }
      const event = step.value;
      if (event.kind === 'solved') solution = event;
      handleEvent(event);
      if (performance.now() > deadline) break;
    }
    draw();
    requestAnimationFrame(pump);
  };
  requestAnimationFrame(pump);
}

function runPython(spec) {
  if (!engine.ready) { engine.pendingSolve = spec; return; }
  engine.worker.postMessage({ type: 'solve', payload: spec });
}

function run() {
  if (state.running) return;
  beginSolve();
  if (state.engine === 'python' && engine.ready) runPython(currentSpec());
  else runJs();
}

/* ------------------------------------------------------------ scenario --- */

function regenerate() {
  const built = buildScenario(state.scenario, { seed: state.seed, agents: state.agents });
  state.matrix = built.matrix;
  state.agentList = built.agents;
  state.solution = null;
  state.searchPaths = null;
  state.conflicts = [];
  state.t = 0;
  $('scenario-blurb').textContent = built.blurb;
  $('play').disabled = true;
  $('timeline').disabled = true;
  showMetrics(null, { expansions: 0, conflicts: 0 });
  $('log').innerHTML = '';
  updateSnippet();
  draw();
}

function updateSnippet() {
  const options = state.algorithm === 'wcbs' ? `, weight=${state.weight}` : '';
  const heuristic = state.heuristic === 'manhattan' ? '' : `, heuristic="${state.heuristic}"`;
  $('equivalent-code').innerHTML = `<span class="kw">import</span> pymapf

scenario = pymapf.build_scenario(<span class="s">"${state.scenario}"</span>, n_agents=${state.agentList.length}, seed=${state.seed})
trace = pymapf.SearchTrace()
solution = pymapf.solve(scenario.to_problem(), <span class="s">"${state.algorithm}"</span>, observer=trace${options}${heuristic})

<span class="kw">print</span>(solution.sum_of_costs, solution.makespan, trace.summary())`;
}

/* ------------------------------------------------------- map painting --- */

let painting = null;

function cellFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const cell = rect.width / state.matrix[0].length;
  const c = Math.floor((event.clientX - rect.left) / cell);
  const r = Math.floor((event.clientY - rect.top) / cell);
  if (r < 0 || c < 0 || r >= state.matrix.length || c >= state.matrix[0].length) return null;
  return [r, c];
}

function paint(event) {
  const cell = cellFromEvent(event);
  if (!cell) return;
  const [r, c] = cell;
  // Never wall in a start or a goal: that would make the instance invalid.
  if (state.agentList.some((a) => (a.start[0] === r && a.start[1] === c)
    || (a.goal[0] === r && a.goal[1] === c))) return;
  if (painting === null) painting = state.matrix[r][c] ? 0 : 1;
  if (state.matrix[r][c] === painting) return;
  state.matrix[r][c] = painting;
  state.solution = null;
  state.searchPaths = null;
  draw();
}

canvas.addEventListener('pointerdown', (event) => {
  canvas.setPointerCapture(event.pointerId);
  painting = null;
  paint(event);
});
canvas.addEventListener('pointermove', (event) => { if (event.buttons) paint(event); });
canvas.addEventListener('pointerup', () => { painting = null; });

/* ------------------------------------------------------------- charts --- */

function svgElement(name, attributes) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

function lineChart(target, series, { xLabel, yLabel, log = false, formatY = String }) {
  const svg = $(target);
  svg.innerHTML = '';
  const W = 420; const H = 260;
  const pad = { l: 52, r: 16, t: 14, b: 38 };
  const names = Object.keys(series).filter((k) => series[k].length);
  if (!names.length) return;

  const xs = names.flatMap((n) => series[n].map((p) => p[0]));
  const ys = names.flatMap((n) => series[n].map((p) => p[1])).filter((v) => v > 0 || !log);
  const xMin = Math.min(...xs); const xMax = Math.max(...xs);
  let yMin = Math.min(...ys); let yMax = Math.max(...ys);
  if (log) { yMin = Math.max(yMin, 1e-6); yMax = Math.max(yMax, yMin * 10); }
  else { yMin = 0; yMax *= 1.12; }

  const px = (v) => pad.l + ((v - xMin) / Math.max(1e-9, xMax - xMin)) * (W - pad.l - pad.r);
  const py = (v) => {
    const t = log
      ? (Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))
      : (v - yMin) / (yMax - yMin);
    return H - pad.b - t * (H - pad.t - pad.b);
  };

  for (let i = 0; i <= 4; i += 1) {
    const value = log
      ? yMin * (yMax / yMin) ** (i / 4)
      : yMin + (i / 4) * (yMax - yMin);
    const y = py(value);
    svg.appendChild(svgElement('line', {
      x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: css('--grid'), 'stroke-width': 1,
    }));
    const label = svgElement('text', {
      x: pad.l - 8, y: y + 4, 'text-anchor': 'end', fill: css('--muted'), 'font-size': 10,
    });
    label.textContent = formatY(value);
    svg.appendChild(label);
  }

  const colors = SERIES();
  names.forEach((name, index) => {
    const points = [...series[name]].sort((a, b) => a[0] - b[0]);
    const color = colors[index % colors.length];
    const path = points.map(([x, y], i) => `${i ? 'L' : 'M'}${px(x).toFixed(1)},${py(y).toFixed(1)}`).join(' ');
    svg.appendChild(svgElement('path', {
      d: path, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    }));
    points.forEach(([x, y]) => {
      svg.appendChild(svgElement('circle', {
        cx: px(x), cy: py(y), r: 3.6, fill: color, stroke: css('--surface-1'), 'stroke-width': 1.4,
      }));
    });
    const last = points[points.length - 1];
    const label = svgElement('text', {
      x: Math.min(px(last[0]) + 7, W - 4), y: py(last[1]) + 3.5,
      fill: css('--ink-2'), 'font-size': 10.5, 'text-anchor': px(last[0]) > W - 60 ? 'end' : 'start',
    });
    label.textContent = name;
    svg.appendChild(label);
  });

  const xTicks = [...new Set(xs)].sort((a, b) => a - b);
  xTicks.forEach((value) => {
    const label = svgElement('text', {
      x: px(value), y: H - pad.b + 15, 'text-anchor': 'middle', fill: css('--muted'), 'font-size': 10,
    });
    label.textContent = String(value);
    svg.appendChild(label);
  });

  const xTitle = svgElement('text', {
    x: (W + pad.l) / 2, y: H - 6, 'text-anchor': 'middle', fill: css('--ink-2'), 'font-size': 11,
  });
  xTitle.textContent = xLabel;
  svg.appendChild(xTitle);

  const yTitle = svgElement('text', {
    x: 12, y: (H - pad.b + pad.t) / 2, fill: css('--ink-2'), 'font-size': 11,
    transform: `rotate(-90 12 ${(H - pad.b + pad.t) / 2})`, 'text-anchor': 'middle',
  });
  yTitle.textContent = yLabel;
  svg.appendChild(yTitle);
}

function barChart(target, entries) {
  const svg = $(target);
  svg.innerHTML = '';
  const W = 420; const H = 200;
  const pad = { l: 96, r: 52, t: 12, b: 26 };
  const colors = SERIES();
  const rowHeight = (H - pad.t - pad.b) / Math.max(1, entries.length);

  entries.forEach(([name, value], index) => {
    const y = pad.t + index * rowHeight;
    const barHeight = Math.min(26, rowHeight * 0.56);
    const width = (value / 100) * (W - pad.l - pad.r);
    svg.appendChild(svgElement('rect', {
      x: pad.l, y: y + (rowHeight - barHeight) / 2, width: Math.max(2, width), height: barHeight,
      rx: 4, fill: colors[index % colors.length],
    }));
    const label = svgElement('text', {
      x: pad.l - 10, y: y + rowHeight / 2 + 4, 'text-anchor': 'end', fill: css('--ink-2'), 'font-size': 11.5,
    });
    label.textContent = name;
    svg.appendChild(label);
    const value_ = svgElement('text', {
      x: pad.l + Math.max(2, width) + 8, y: y + rowHeight / 2 + 4, fill: css('--ink-2'), 'font-size': 11.5,
    });
    value_.textContent = `${value.toFixed(0)}%`;
    svg.appendChild(value_);
  });
}

/* ---------------------------------------------------------- benchmark --- */

const bench = { rows: [], running: false };

function benchmarkRow(row) {
  bench.rows.push(row);
  $('bench-status').textContent =
    `${bench.rows.length} runs — latest: ${row.algorithm} on ${row.n_agents} agents ` +
    `(${row.solved ? `${(1000 * row.runtime).toFixed(0)} ms` : 'no solution'})`;
  renderBenchmark();
}

function benchmarkDone(rows) {
  if (rows?.length) bench.rows = rows;
  bench.running = false;
  $('run-bench').disabled = false;
  $('bench-status').textContent = `${bench.rows.length} runs complete`;
  renderBenchmark();
}

function meanBy(rows, key) {
  const buckets = {};
  for (const row of rows) {
    if (!row.solved || row[key] == null) continue;
    (buckets[row.algorithm] ||= {});
    (buckets[row.algorithm][row.n_agents] ||= []).push(row[key]);
  }
  const out = {};
  for (const [algorithm, byCount] of Object.entries(buckets)) {
    out[algorithm] = Object.entries(byCount)
      .map(([count, values]) => [Number(count), values.reduce((a, b) => a + b, 0) / values.length])
      .sort((a, b) => a[0] - b[0]);
  }
  return out;
}

function renderBenchmark() {
  lineChart('chart-runtime', meanBy(bench.rows, 'runtime'), {
    xLabel: 'agents', yLabel: 'runtime (s)', log: true,
    formatY: (v) => (v >= 1 ? v.toFixed(1) : v >= 0.001 ? `${(v * 1000).toFixed(0)}ms` : v.toExponential(0)),
  });
  lineChart('chart-cost', meanBy(bench.rows, 'sum_of_costs'), {
    xLabel: 'agents', yLabel: 'sum of costs', formatY: (v) => v.toFixed(0),
  });

  const algorithms = [...new Set(bench.rows.map((r) => r.algorithm))];
  barChart('chart-success', algorithms.map((algorithm) => {
    const rows = bench.rows.filter((r) => r.algorithm === algorithm);
    return [algorithm, rows.length ? (100 * rows.filter((r) => r.solved).length) / rows.length : 0];
  }));

  const table = $('bench-table');
  table.innerHTML = '<thead><tr><th>agents</th><th>solver</th><th>cost</th><th>makespan</th><th>ms</th></tr></thead>';
  const body = document.createElement('tbody');
  for (const row of bench.rows.slice(-40).reverse()) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.n_agents}</td><td>${row.algorithm}</td>`
      + (row.solved
        ? `<td>${row.sum_of_costs}</td><td>${row.makespan}</td><td>${(1000 * row.runtime).toFixed(1)}</td>`
        : '<td class="fail" colspan="3">no solution within the budget</td>');
    body.appendChild(tr);
  }
  table.appendChild(body);
}

function runBenchmark() {
  if (bench.running) return;
  bench.running = true;
  bench.rows = [];
  $('run-bench').disabled = true;
  const scenario = $('bench-scenario').value;
  const spec = {
    scenario,
    agent_counts: [2, 4, 6, 8, 10, 12],
    algorithms: ['cbs', 'wcbs', 'prioritized'],
    seeds: [0, 1, 2],
    solver_kwargs: { time_limit: 2.0 },
  };

  if (state.engine === 'python' && engine.ready) {
    $('bench-status').textContent = 'running in Python (WASM)…';
    engine.worker.postMessage({ type: 'benchmark', payload: spec });
    return;
  }

  // JS fallback: yield to the browser between instances so the page stays alive.
  $('bench-status').textContent = 'running in the JS engine…';
  const jobs = [];
  for (const count of spec.agent_counts) {
    for (const seed of spec.seeds) {
      for (const algorithm of spec.algorithms) jobs.push({ count, seed, algorithm });
    }
  }
  let index = 0;
  const step = () => {
    if (index >= jobs.length) { benchmarkDone(); return; }
    const job = jobs[index++];
    try {
      const built = buildScenario(scenario, { seed: job.seed, agents: job.count });
      const problem = problemFromState(built);
      const started = performance.now();
      const generator = solveGenerator(problem, job.algorithm, { weight: 1.5, timeLimit: 2 });
      let solution = null;
      let step_ = generator.next();
      while (!step_.done) { if (step_.value.kind === 'solved') solution = step_.value; step_ = generator.next(); }
      benchmarkRow({
        n_agents: job.count,
        algorithm: job.algorithm,
        solved: !!solution,
        runtime: (performance.now() - started) / 1000,
        sum_of_costs: solution?.sumOfCosts ?? null,
        makespan: solution?.makespan ?? null,
        expansions: solution?.expansions ?? 0,
      });
    } catch (error) {
      benchmarkRow({ n_agents: job.count, algorithm: job.algorithm, solved: false, runtime: 0 });
    }
    setTimeout(step, 0);
  };
  step();
}

/* --------------------------------------------------------- code panel --- */

const EXAMPLES = {
  'Solve and inspect': `import pymapf

scenario = pymapf.build_scenario("warehouse", n_agents=8, seed=3)
trace = pymapf.SearchTrace()
solution = pymapf.solve(scenario.to_problem(), "wcbs", observer=trace, weight=1.5)

print("cost     ", solution.sum_of_costs)
print("makespan ", solution.makespan)
print("valid    ", solution.is_valid())
print("search   ", trace.summary())
`,
  'Compare every solver': `import pymapf
from pymapf.benchmark import compare_algorithms

report = compare_algorithms(
    ["corner_swap", "warehouse", "random_obstacles"],
    ["cbs", "wcbs", "prioritized"],
    time_limit=3.0,
)
print(report.table())
`,
  'Where prioritized planning breaks': `import pymapf

# Prioritized planning is fast but incomplete: with a bad order, an agent can
# be locked out of a corridor that a joint plan would have shared.
scenario = pymapf.build_scenario("bottleneck", n_agents=6, seed=4)
problem = scenario.to_problem()

for algorithm in ("prioritized", "wcbs"):
    trace = pymapf.SearchTrace()
    solution = pymapf.solve(problem, algorithm, observer=trace, time_limit=5.0)
    print(algorithm, "->", "cost %d" % solution.sum_of_costs if solution else "NO SOLUTION")
    if not solution:
        print("   ", trace.of_kind("failed")[-1]["reason"])
`,
  'Design a map by hand': `import pymapf
from pymapf.scenarios import from_ascii, to_ascii

scenario = from_ascii("""
##########
#a......A#
#.######.#
#B......b#
##########
""")
print(to_ascii(scenario))

solution = pymapf.solve(scenario.to_problem(), "cbs")
for name, path in solution.paths.items():
    print(name, "->", len(path) - 1, "steps")
`,
  'Register your own solver': `import pymapf
from pymapf.core import MAPFSolver, Solution, register_solver
from pymapf.algorithms import space_time_astar


@register_solver("selfish")
class Selfish(MAPFSolver):
    """Every agent takes its own shortest path and ignores the others."""

    def solve(self, problem, observer=None):
        paths = {}
        for agent in problem.agents:
            paths[agent.name] = space_time_astar(problem.grid, agent.start, agent.goal)
        return Solution(paths=paths, algorithm=self.name)


scenario = pymapf.build_scenario("corner_swap", n_agents=4)
solution = pymapf.solve(scenario.to_problem(), "selfish")
print("registered solvers:", pymapf.available_solvers())
print("cost", solution.sum_of_costs, "— valid?", solution.is_valid())
print("first conflict:", solution.first_conflict())
`,
};

function showExecResult(data) {
  const output = $('code-output');
  output.textContent = data.output || '(no output)';
  if (data.error) {
    const block = document.createElement('span');
    block.className = 'err';
    block.textContent = `\n${data.error}`;
    output.appendChild(block);
  }
  $('run-code').disabled = false;

  if (data.solution?.paths) {
    if (data.solution.matrix) {
      state.matrix = data.solution.matrix;
      state.agentList = data.solution.agents;
    }
    state.conflictCount = 0;
    finishSolve({
      paths: data.solution.paths,
      sumOfCosts: data.solution.sum_of_costs,
      makespan: data.solution.makespan,
      expansions: data.solution.expansions,
      runtime: data.solution.runtime,
      valid: true,
      algorithm: data.solution.algorithm,
    });
    log('plan loaded from the Python tab — switch to Visual to watch it', 'solved');
  }
}

function runCode() {
  if (!engine.ready) {
    $('code-output').textContent =
      'The Python runtime is not available in this browser session, so the Python '
      + 'tab cannot run. The Visual and Benchmark tabs still work.';
    return;
  }
  $('run-code').disabled = true;
  $('code-output').textContent = 'running…';
  engine.worker.postMessage({ type: 'exec', payload: $('editor').value });
}

/* ----------------------------------------------------------------- ui --- */

function selectEngine(name) {
  state.engine = name;
  document.querySelectorAll('#engine button').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.engine === name));
  });
}

function selectTab(id) {
  for (const tab of ['visual', 'code', 'bench']) {
    const selected = tab === id;
    $(`tab-${tab}`).setAttribute('aria-selected', String(selected));
    $(`panel-${tab}`).hidden = !selected;
  }
  if (id === 'visual') draw();
}

function wire() {
  const scenarioSelect = $('scenario');
  const benchSelect = $('bench-scenario');
  for (const [name, spec] of Object.entries(SCENARIOS)) {
    scenarioSelect.appendChild(new Option(spec.label, name));
    benchSelect.appendChild(new Option(spec.label, name));
  }
  scenarioSelect.value = state.scenario;
  benchSelect.value = 'random_obstacles';

  const examples = $('examples');
  for (const name of Object.keys(EXAMPLES)) examples.appendChild(new Option(name, name));
  $('editor').value = EXAMPLES[Object.keys(EXAMPLES)[0]];
  examples.addEventListener('change', () => { $('editor').value = EXAMPLES[examples.value]; });

  scenarioSelect.addEventListener('change', () => { state.scenario = scenarioSelect.value; regenerate(); });
  $('agents').addEventListener('input', (event) => {
    state.agents = Number(event.target.value);
    $('agents-out').value = event.target.value;
    regenerate();
  });
  $('seed').addEventListener('input', (event) => {
    state.seed = Number(event.target.value);
    $('seed-out').value = event.target.value;
    regenerate();
  });
  $('algorithm').addEventListener('change', (event) => {
    state.algorithm = event.target.value;
    $('algorithm-blurb').textContent = ALGORITHM_BLURBS[state.algorithm];
    $('weight-group').style.display = state.algorithm === 'wcbs' ? '' : 'none';
    updateSnippet();
  });
  $('weight').addEventListener('input', (event) => {
    state.weight = Number(event.target.value);
    $('weight-out').value = event.target.value;
    updateSnippet();
  });
  $('heuristic').addEventListener('change', (event) => { state.heuristic = event.target.value; updateSnippet(); });

  document.querySelectorAll('#engine button').forEach((button) => {
    button.addEventListener('click', () => {
      if (button.dataset.engine === 'python' && !engine.ready && engine.failed) return;
      selectEngine(button.dataset.engine);
    });
  });

  $('run').addEventListener('click', run);
  $('regenerate').addEventListener('click', () => {
    $('seed').value = String((state.seed + 1) % 41);
    state.seed = Number($('seed').value);
    $('seed-out').value = $('seed').value;
    regenerate();
  });
  $('clear-walls').addEventListener('click', () => {
    const rows = state.matrix.length; const cols = state.matrix[0].length;
    state.matrix = state.matrix.map((row, r) => row.map((v, c) =>
      ((r === 0 || c === 0 || r === rows - 1 || c === cols - 1) ? 1 : 0)));
    state.solution = null;
    draw();
  });

  $('play').addEventListener('click', () => {
    if (state.playing) { state.playing = false; $('play').textContent = '⏵ Play'; } else play();
  });
  $('timeline').addEventListener('input', (event) => {
    state.playing = false;
    $('play').textContent = '⏵ Play';
    state.t = Number(event.target.value);
    $('clock').textContent = `t = ${state.t} / ${state.solution?.makespan ?? 0}`;
    draw();
  });

  $('run-code').addEventListener('click', runCode);
  $('run-bench').addEventListener('click', runBenchmark);

  for (const tab of ['visual', 'code', 'bench']) {
    $(`tab-${tab}`).addEventListener('click', () => selectTab(tab));
  }

  $('theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('pymapf-theme', next); } catch { /* private mode */ }
    draw();
    renderBenchmark();
  });

  window.addEventListener('resize', draw);
  window.addEventListener('keydown', (event) => {
    if (event.target.matches('input, textarea, select')) return;
    if (event.key === 'r') run();
    if (event.key === ' ') { event.preventDefault(); $('play').click(); }
  });
}

function boot() {
  try {
    const stored = localStorage.getItem('pymapf-theme');
    if (stored) document.documentElement.dataset.theme = stored;
  } catch { /* ignore */ }

  wire();
  $('algorithm-blurb').textContent = ALGORITHM_BLURBS[state.algorithm];
  regenerate();
  bootWorker();
}

boot();
