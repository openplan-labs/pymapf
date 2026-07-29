/**
 * mapf.js — the playground's instant-response engine.
 *
 * This is a faithful port of pymapf's core search (space-time A*, prioritized
 * planning, CBS and weighted CBS) so the page is fully interactive from the
 * first paint, before the ~10 MB Pyodide runtime has finished downloading —
 * and so it still works if that download is blocked.
 *
 * The solvers are written as generators that yield SearchEvents, exactly the
 * vocabulary pymapf.core.trace defines. The driver pumps them with a per-frame
 * time budget, which is what makes the search animate live instead of freezing
 * the tab. When Pyodide is ready the real Python library takes over and emits
 * the same events, so the UI never has to care which engine ran.
 *
 * Cells are [row, col]; a solution's paths[name][t] is the cell at time t.
 */

const ORTHOGONAL = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const DIAGONAL = [[-1, -1], [-1, 1], [1, -1], [1, 1]];

export const key = (r, c) => r + ',' + c;

/* ---------------------------------------------------------------- grid --- */

export function makeGrid(height, width, blocked = []) {
  return { height, width, blocked: new Set(blocked.map(([r, c]) => key(r, c))) };
}

export function gridFromMatrix(matrix) {
  const blocked = [];
  matrix.forEach((row, r) => row.forEach((v, c) => { if (v) blocked.push([r, c]); }));
  return makeGrid(matrix.length, matrix[0].length, blocked);
}

export function toMatrix(grid) {
  return Array.from({ length: grid.height }, (_, r) =>
    Array.from({ length: grid.width }, (_, c) => (isFree(grid, r, c) ? 0 : 1)));
}

export const inBounds = (grid, r, c) => r >= 0 && r < grid.height && c >= 0 && c < grid.width;
export const isFree = (grid, r, c) => inBounds(grid, r, c) && !grid.blocked.has(key(r, c));

export function freeCellCount(grid) {
  return grid.height * grid.width - grid.blocked.size;
}

export function neighbors(grid, [r, c], allowDiagonals = false) {
  const out = [];
  for (const [dr, dc] of ORTHOGONAL) {
    if (isFree(grid, r + dr, c + dc)) out.push([r + dr, c + dc]);
  }
  if (allowDiagonals) {
    for (const [dr, dc] of DIAGONAL) {
      // No corner cutting: both orthogonal cells must be free too.
      if (isFree(grid, r + dr, c + dc) && isFree(grid, r + dr, c) && isFree(grid, r, c + dc)) {
        out.push([r + dr, c + dc]);
      }
    }
  }
  return out;
}

/* ---------------------------------------------------------- heuristics --- */

export const HEURISTICS = {
  manhattan: (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]),
  euclidean: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]),
  chebyshev: (a, b) => Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])),
  octile: (a, b) => {
    const dr = Math.abs(a[0] - b[0]); const dc = Math.abs(a[1] - b[1]);
    return (dr + dc) + (Math.SQRT2 - 2) * Math.min(dr, dc);
  },
};

/* --------------------------------------------------------- constraints --- */

export function newConstraints() {
  return { vertex: new Set(), edge: new Set(), lastVertex: new Map() };
}

export function addVertex(constraints, [r, c], t) {
  constraints.vertex.add(r + ',' + c + ',' + t);
  const k = key(r, c);
  constraints.lastVertex.set(k, Math.max(constraints.lastVertex.get(k) ?? -1, t));
}

export function addEdge(constraints, [ur, uc], [vr, vc], t) {
  constraints.edge.add(ur + ',' + uc + '>' + vr + ',' + vc + ',' + t);
}

export function copyConstraints(constraints) {
  return {
    vertex: new Set(constraints.vertex),
    edge: new Set(constraints.edge),
    lastVertex: new Map(constraints.lastVertex),
  };
}

const blocksVertex = (constraints, [r, c], t) => constraints.vertex.has(r + ',' + c + ',' + t);
const blocksEdge = (constraints, [ur, uc], [vr, vc], t) =>
  constraints.edge.has(ur + ',' + uc + '>' + vr + ',' + vc + ',' + t);

/* -------------------------------------------------------- binary heap --- */

class Heap {
  constructor(compare) { this.items = []; this.compare = compare; }
  get size() { return this.items.length; }
  push(item) {
    const items = this.items;
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(items[i], items[parent]) >= 0) break;
      [items[i], items[parent]] = [items[parent], items[i]];
      i = parent;
    }
  }
  pop() {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1; const r = l + 1;
        let best = i;
        if (l < items.length && this.compare(items[l], items[best]) < 0) best = l;
        if (r < items.length && this.compare(items[r], items[best]) < 0) best = r;
        if (best === i) break;
        [items[i], items[best]] = [items[best], items[i]];
        i = best;
      }
    }
    return top;
  }
  peek() { return this.items[0]; }
}

const byFirst = (a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]);

/* ------------------------------------------------- space-time A* (low) --- */

export function spaceTimeAStar(grid, start, goal, constraints = null, options = {}) {
  const { heuristic = 'manhattan', allowDiagonals = false } = options;
  const h = typeof heuristic === 'function' ? heuristic : HEURISTICS[heuristic];
  const cons = constraints || newConstraints();

  const settleTime = cons.lastVertex.get(key(goal[0], goal[1])) ?? -1;
  let lastConstraintT = 0;
  for (const entry of cons.vertex) lastConstraintT = Math.max(lastConstraintT, +entry.split(',')[2]);
  for (const entry of cons.edge) lastConstraintT = Math.max(lastConstraintT, +entry.split(',').pop());
  const maxTimestep = options.maxTimestep ?? (freeCellCount(grid) + lastConstraintT + 1);

  const open = new Heap(byFirst);
  let tie = 0;
  open.push([h(start, goal), tie++, 0, start]);
  const visited = new Set();
  const parent = new Map([[key(start[0], start[1]) + ',0', null]]);

  while (open.size) {
    const [, , t, cell] = open.pop();
    const state = key(cell[0], cell[1]) + ',' + t;
    if (visited.has(state)) continue;
    visited.add(state);

    if (cell[0] === goal[0] && cell[1] === goal[1] && t > settleTime) {
      const path = [];
      let cursor = state;
      while (cursor != null) {
        const [r, c] = cursor.split(',');
        path.push([+r, +c]);
        cursor = parent.get(cursor);
      }
      return path.reverse();
    }
    if (t >= maxTimestep) continue;

    for (const next of [...neighbors(grid, cell, allowDiagonals), cell]) {
      const nt = t + 1;
      const nstate = key(next[0], next[1]) + ',' + nt;
      if (visited.has(nstate)) continue;
      if (blocksVertex(cons, next, nt)) continue;
      if (blocksEdge(cons, cell, next, nt)) continue;
      if (!parent.has(nstate)) parent.set(nstate, state);
      open.push([nt + h(next, goal), tie++, nt, next]);
    }
  }
  return null;
}

/* ---------------------------------------------------------- conflicts --- */

const cellAt = (path, t) => (t < path.length ? path[t] : path[path.length - 1]);
const same = (a, b) => a[0] === b[0] && a[1] === b[1];

export function findFirstConflict(paths) {
  const names = Object.keys(paths);
  const horizon = Math.max(0, ...names.map((n) => paths[n].length));
  for (let t = 0; t < horizon; t += 1) {
    for (let i = 0; i < names.length; i += 1) {
      for (let j = i + 1; j < names.length; j += 1) {
        const pa = paths[names[i]]; const pb = paths[names[j]];
        const aNow = cellAt(pa, t); const bNow = cellAt(pb, t);
        if (same(aNow, bNow)) {
          return { kind: 'vertex', a: names[i], b: names[j], t, cellA: aNow, cellB: bNow };
        }
        const aNext = cellAt(pa, t + 1); const bNext = cellAt(pb, t + 1);
        if (same(aNow, bNext) && same(bNow, aNext)) {
          return { kind: 'edge', a: names[i], b: names[j], t: t + 1, cellA: aNext, cellB: bNext };
        }
      }
    }
  }
  return null;
}

export function countConflicts(paths) {
  const names = Object.keys(paths);
  const horizon = Math.max(0, ...names.map((n) => paths[n].length));
  let total = 0;
  for (let t = 0; t < horizon; t += 1) {
    for (let i = 0; i < names.length; i += 1) {
      for (let j = i + 1; j < names.length; j += 1) {
        const pa = paths[names[i]]; const pb = paths[names[j]];
        const aNow = cellAt(pa, t); const bNow = cellAt(pb, t);
        if (same(aNow, bNow)) { total += 1; continue; }
        if (same(aNow, cellAt(pb, t + 1)) && same(bNow, cellAt(pa, t + 1))) total += 1;
      }
    }
  }
  return total;
}

export function solutionMetrics(paths, algorithm, expansions, runtime) {
  const lengths = Object.values(paths).map((p) => p.length - 1);
  return {
    paths,
    algorithm,
    expansions,
    runtime,
    makespan: Math.max(0, ...lengths),
    sumOfCosts: lengths.reduce((a, b) => a + b, 0),
    valid: findFirstConflict(paths) === null,
  };
}

/* -------------------------------------------- solvers (as generators) --- */

function reservationsToConstraints(reservations, horizon) {
  const cons = newConstraints();
  for (const path of Object.values(reservations)) {
    path.forEach((cell, t) => addVertex(cons, cell, t));
    for (let t = path.length; t <= horizon; t += 1) addVertex(cons, path[path.length - 1], t);
    for (let t = 0; t < path.length - 1; t += 1) addEdge(cons, path[t + 1], path[t], t + 1);
  }
  return cons;
}

/** Prioritized planning: agents in order, each avoiding the ones before it. */
export function* prioritizedPlanning(problem, options = {}) {
  const started = performance.now();
  const { grid, agents, allowDiagonals = false } = problem;
  const order = options.priority || agents.map((a) => a.name);
  const lookup = Object.fromEntries(agents.map((a) => [a.name, a]));

  yield { kind: 'root', agents: order, cost: 0 };
  const reservations = {};
  let expansions = 0;

  for (const name of order) {
    const agent = lookup[name];
    const reserved = Object.values(reservations).reduce((n, p) => n + p.length, 0);
    const horizon = freeCellCount(grid) + reserved + order.length + 1;
    const path = spaceTimeAStar(grid, agent.start, agent.goal,
      reservationsToConstraints(reservations, horizon),
      { heuristic: options.heuristic, allowDiagonals, maxTimestep: horizon });
    if (!path) {
      yield {
        kind: 'failed',
        reason: `agent ${name} has no path under the reservations of the ` +
                `${Object.keys(reservations).length} higher-priority agents`,
      };
      return null;
    }
    reservations[name] = path;
    expansions += 1;
    yield { kind: 'agent_planned', agent: name, path, cost: path.length - 1 };
    yield {
      kind: 'expand',
      node: expansions,
      cost: Object.values(reservations).reduce((n, p) => n + p.length - 1, 0),
      open: order.length - expansions,
      paths: { ...reservations },
    };
  }

  const solution = solutionMetrics(reservations, 'prioritized', expansions, (performance.now() - started) / 1000);
  yield { kind: 'solved', ...solution };
  return solution;
}

/**
 * Conflict-based search. `weight === 1` is optimal CBS; a larger weight runs
 * the focal (ECBS) variant, whose cost is bounded by weight × optimal.
 */
export function* conflictBasedSearch(problem, options = {}) {
  const started = performance.now();
  const { grid, agents, allowDiagonals = false } = problem;
  const weight = options.weight ?? 1;
  const maxExpansions = options.maxExpansions ?? 10000;
  const timeLimit = options.timeLimit ?? null;
  const searchOptions = { heuristic: options.heuristic, allowDiagonals };

  const lowLevel = (agent, cons) => spaceTimeAStar(grid, agent.start, agent.goal, cons, searchOptions);

  const rootConstraints = {};
  const rootPaths = {};
  for (const agent of agents) {
    rootConstraints[agent.name] = newConstraints();
    const path = lowLevel(agent, rootConstraints[agent.name]);
    if (!path) {
      yield { kind: 'failed', reason: `agent ${agent.name} has no individual path` };
      return null;
    }
    rootPaths[agent.name] = path;
    yield { kind: 'agent_planned', agent: agent.name, path, cost: path.length - 1 };
  }

  const makeNode = (constraints, paths) => ({
    constraints,
    paths,
    cost: Object.values(paths).reduce((n, p) => n + p.length - 1, 0),
    conflicts: countConflicts(paths),
    expanded: false,
  });

  const root = makeNode(rootConstraints, rootPaths);
  yield { kind: 'root', agents: agents.map((a) => a.name), cost: root.cost };

  let tie = 0;
  // OPEN orders by cost (the lower bound); FOCAL orders by conflict count and
  // holds the nodes within weight × that bound.
  const open = new Heap((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const pending = new Heap((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const focal = new Heap((a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]));
  const tag = tie++;
  open.push([root.cost, tag, root]);
  if (weight > 1) pending.push([root.cost, tag, root]);
  else focal.push([root.conflicts, root.cost, tag, root]);

  let expansions = 0;
  let timedOut = false;

  while (open.size && expansions < maxExpansions) {
    if (timeLimit != null && (performance.now() - started) / 1000 > timeLimit) { timedOut = true; break; }
    while (open.size && open.peek()[2].expanded) open.pop();
    if (!open.size) break;

    const bound = weight * open.peek()[0];
    while (pending.size && pending.peek()[0] <= bound) {
      const [cost, nodeTag, candidate] = pending.pop();
      focal.push([candidate.conflicts, cost, nodeTag, candidate]);
    }

    let node = null;
    while (focal.size) {
      const candidate = focal.pop()[3];
      if (candidate.expanded) continue;
      node = candidate;
      break;
    }
    if (!node) node = open.pop()[2];
    node.expanded = true;
    expansions += 1;

    yield {
      kind: 'expand',
      node: expansions,
      cost: node.cost,
      open: open.size,
      bound,
      paths: node.paths,
    };

    const conflict = findFirstConflict(node.paths);
    if (!conflict) {
      const solution = solutionMetrics(node.paths, weight > 1 ? 'wcbs' : 'cbs',
        expansions, (performance.now() - started) / 1000);
      yield { kind: 'solved', ...solution };
      return solution;
    }

    yield {
      kind: 'conflict',
      type: conflict.kind,
      a: conflict.a,
      b: conflict.b,
      t: conflict.t,
      cell: conflict.cellA,
    };

    for (const name of [conflict.a, conflict.b]) {
      const constraints = {};
      for (const [other, cons] of Object.entries(node.constraints)) {
        constraints[other] = other === name ? copyConstraints(cons) : cons;
      }
      const mine = constraints[name];
      if (conflict.kind === 'vertex') {
        addVertex(mine, conflict.cellA, conflict.t);
      } else if (name === conflict.a) {
        addEdge(mine, conflict.cellB, conflict.cellA, conflict.t);
      } else {
        addEdge(mine, conflict.cellA, conflict.cellB, conflict.t);
      }

      const agent = agents.find((a) => a.name === name);
      const path = lowLevel(agent, mine);
      if (!path) continue;
      const child = makeNode(constraints, { ...node.paths, [name]: path });
      yield { kind: 'branch', agent: name, constraint: conflict.kind, cost: child.cost };

      const childTag = tie++;
      open.push([child.cost, childTag, child]);
      if (child.cost <= bound) focal.push([child.conflicts, child.cost, childTag, child]);
      else pending.push([child.cost, childTag, child]);
    }
  }

  const reason = timedOut
    ? `time limit (${timeLimit}s) reached after ${expansions} nodes`
    : (expansions >= maxExpansions
      ? `expansion limit (${maxExpansions}) reached`
      : 'constraint tree exhausted');
  yield { kind: 'failed', reason };
  return null;
}

export function solveGenerator(problem, algorithm, options = {}) {
  if (algorithm === 'prioritized') return prioritizedPlanning(problem, options);
  if (algorithm === 'wcbs') return conflictBasedSearch(problem, { ...options, weight: options.weight ?? 1.5 });
  return conflictBasedSearch(problem, { ...options, weight: 1 });
}

/** Drain a solver generator synchronously (used by the benchmark tab). */
export function solve(problem, algorithm, options = {}) {
  const generator = solveGenerator(problem, algorithm, options);
  let step = generator.next();
  while (!step.done) step = generator.next();
  return step.value;
}

/* ---------------------------------------------------------- scenarios --- */

/** Deterministic 32-bit PRNG, so a seed reproduces an instance exactly. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
export const agentName = (i) => (i < 26 ? ALPHABET[i] : ALPHABET[i % 26] + (Math.floor(i / 26) + 1));

function largestComponent(matrix) {
  const height = matrix.length; const width = matrix[0].length;
  const seen = new Set(); let best = [];
  for (let r = 0; r < height; r += 1) {
    for (let c = 0; c < width; c += 1) {
      if (matrix[r][c] || seen.has(key(r, c))) continue;
      const stack = [[r, c]]; const component = [];
      seen.add(key(r, c));
      while (stack.length) {
        const [cr, cc] = stack.pop();
        component.push([cr, cc]);
        for (const [dr, dc] of ORTHOGONAL) {
          const nr = cr + dr; const nc = cc + dc;
          if (nr >= 0 && nr < height && nc >= 0 && nc < width && !matrix[nr][nc] && !seen.has(key(nr, nc))) {
            seen.add(key(nr, nc));
            stack.push([nr, nc]);
          }
        }
      }
      if (component.length > best.length) best = component;
    }
  }
  return best;
}

function sampleAgents(matrix, count, rng, minSeparation = 3) {
  const cells = largestComponent(matrix);
  if (cells.length < 2 * count) {
    throw new Error(`map has ${cells.length} reachable cells, need ${2 * count} for ${count} agents`);
  }
  const pool = [...cells];
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const starts = pool.slice(0, count);
  const goals = pool.slice(count);
  const distance = (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);

  return starts.map((start, i) => {
    let index = 0;
    for (let k = 0; k < goals.length; k += 1) {
      if (distance(goals[k], start) >= minSeparation) { index = k; break; }
      if (distance(goals[k], start) > distance(goals[index], start)) index = k;
    }
    const [goal] = goals.splice(index, 1);
    return { name: agentName(i), start, goal };
  });
}

const bordered = (height, width, fill = 0) =>
  Array.from({ length: height }, (_, r) => Array.from({ length: width },
    (_, c) => ((r === 0 || r === height - 1 || c === 0 || c === width - 1) ? 1 : fill)));

export const SCENARIOS = {
  empty_room: {
    label: 'Empty room',
    blurb: 'Open floor — every interaction comes from the agents themselves.',
    defaults: { height: 12, width: 12, agents: 4 },
    build({ height, width, agents, seed }) {
      const matrix = bordered(height, width);
      return { matrix, agents: sampleAgents(matrix, agents, mulberry32(seed)) };
    },
  },
  random_obstacles: {
    label: 'Random obstacles',
    blurb: 'Uniform clutter: the standard MAPF stress map.',
    defaults: { height: 16, width: 16, agents: 6, density: 0.18 },
    build({ height, width, agents, seed, density = 0.18 }) {
      const rng = mulberry32(seed);
      const matrix = bordered(height, width);
      for (let r = 1; r < height - 1; r += 1) {
        for (let c = 1; c < width - 1; c += 1) matrix[r][c] = rng() < density ? 1 : 0;
      }
      return { matrix, agents: sampleAgents(matrix, agents, rng) };
    },
  },
  warehouse: {
    label: 'Warehouse',
    blurb: 'Shelf blocks and one-cell aisles — agents meet head-on constantly.',
    defaults: { height: 12, width: 19, agents: 8 },
    build({ agents, seed }) {
      const shelfRows = 3; const shelfCols = 4; const shelfH = 2; const shelfW = 3;
      const height = 2 + shelfRows * (shelfH + 1) + 1;
      const width = 2 + shelfCols * (shelfW + 1) + 1;
      const matrix = bordered(height, width);
      for (let br = 0; br < shelfRows; br += 1) {
        for (let bc = 0; bc < shelfCols; bc += 1) {
          const top = 2 + br * (shelfH + 1); const left = 2 + bc * (shelfW + 1);
          for (let r = top; r < Math.min(top + shelfH, height - 1); r += 1) {
            for (let c = left; c < Math.min(left + shelfW, width - 1); c += 1) matrix[r][c] = 1;
          }
        }
      }
      return { matrix, agents: sampleAgents(matrix, agents, mulberry32(seed), 6) };
    },
  },
  maze: {
    label: 'Maze',
    blurb: 'A perfect maze: exactly one route between any two cells.',
    defaults: { height: 15, width: 15, agents: 4 },
    build({ height, width, agents, seed }) {
      const h = height % 2 ? height : height + 1;
      const w = width % 2 ? width : width + 1;
      const matrix = Array.from({ length: h }, () => Array(w).fill(1));
      const rng = mulberry32(seed);
      matrix[1][1] = 0;
      const stack = [[1, 1]];
      while (stack.length) {
        const [r, c] = stack[stack.length - 1];
        const candidates = [];
        for (const [dr, dc] of [[-2, 0], [2, 0], [0, -2], [0, 2]]) {
          const nr = r + dr; const nc = c + dc;
          if (nr >= 1 && nr < h - 1 && nc >= 1 && nc < w - 1 && matrix[nr][nc]) candidates.push([nr, nc, dr, dc]);
        }
        if (!candidates.length) { stack.pop(); continue; }
        const [nr, nc, dr, dc] = candidates[Math.floor(rng() * candidates.length)];
        matrix[r + dr / 2][c + dc / 2] = 0;
        matrix[nr][nc] = 0;
        stack.push([nr, nc]);
      }
      return { matrix, agents: sampleAgents(matrix, agents, rng, 8) };
    },
  },
  bottleneck: {
    label: 'Bottleneck',
    blurb: 'Two rooms, one corridor. Somebody has to yield.',
    defaults: { agents: 4 },
    build({ agents, seed }) {
      const room = 5; const corridor = 5;
      const height = 2 * room + 1; const width = 2 * room + corridor + 2;
      const matrix = Array.from({ length: height }, () => Array(width).fill(1));
      const mid = Math.floor(height / 2);
      const left = []; const right = [];
      for (let r = 1; r < height - 1; r += 1) {
        for (let c = 1; c <= room; c += 1) { matrix[r][c] = 0; left.push([r, c]); }
        for (let c = room + corridor + 1; c < width - 1; c += 1) { matrix[r][c] = 0; right.push([r, c]); }
      }
      for (let c = room + 1; c <= room + corridor; c += 1) matrix[mid][c] = 0;

      const rng = mulberry32(seed);
      const shuffle = (list) => {
        for (let i = list.length - 1; i > 0; i -= 1) {
          const j = Math.floor(rng() * (i + 1));
          [list[i], list[j]] = [list[j], list[i]];
        }
      };
      shuffle(left); shuffle(right);
      const perSide = Math.max(1, Math.floor(agents / 2));
      const out = [];
      for (let i = 0; i < perSide; i += 1) out.push({ name: agentName(out.length), start: left[i], goal: right[i] });
      for (let i = 0; i < agents - perSide; i += 1) {
        out.push({ name: agentName(out.length), start: right[perSide + i], goal: left[perSide + i] });
      }
      return { matrix, agents: out };
    },
  },
  corner_swap: {
    label: 'Corner swap',
    blurb: 'Symmetric crossings — every route runs through the middle.',
    defaults: { agents: 4 },
    build({ agents }) {
      const size = 9;
      const matrix = bordered(size, size);
      const mid = Math.floor(size / 2); const last = size - 2;
      const ring = [
        [[1, 1], [last, last]], [[last, last], [1, 1]],
        [[1, last], [last, 1]], [[last, 1], [1, last]],
        [[mid, 1], [mid, last]], [[mid, last], [mid, 1]],
        [[1, mid], [last, mid]], [[last, mid], [1, mid]],
      ];
      const count = Math.min(agents, ring.length);
      return {
        matrix,
        agents: ring.slice(0, count).map(([start, goal], i) => ({ name: agentName(i), start, goal })),
      };
    },
  },
};

export function buildScenario(name, overrides = {}) {
  const spec = SCENARIOS[name];
  if (!spec) throw new Error(`unknown scenario ${name}`);
  const params = { seed: 0, ...spec.defaults, ...overrides };
  const { matrix, agents } = spec.build(params);
  return { name, label: spec.label, blurb: spec.blurb, matrix, agents, params };
}

export function problemFromState(state) {
  return {
    grid: gridFromMatrix(state.matrix),
    agents: state.agents.map((a) => ({ name: a.name, start: a.start, goal: a.goal })),
    allowDiagonals: !!state.allowDiagonals,
  };
}
