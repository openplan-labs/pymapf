/**
 * worker.js — runs the real PyMAPF inside Pyodide, off the main thread.
 *
 * The page never freezes during a solve, and search events are streamed back as
 * they are emitted, which is what makes the live view live rather than a replay.
 *
 * The library is not pip-installed: `pymapf-bundle.json` carries the actual
 * source files of the pure-Python modules (built by scripts/build_web_bundle.py)
 * and they are written straight into Pyodide's virtual filesystem. That keeps
 * the download to a few tens of kB and guarantees the page runs the same code
 * that ships in the repository.
 */

/* eslint-env worker */

// Tried in order; the first that loads wins. Pinned versions, newest first.
const PYODIDE_SOURCES = [
  'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide.mjs',
  'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/pyodide.mjs',
  'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs',
];

let pyodide = null;
let bundleVersion = 'unknown';

const post = (message) => self.postMessage(message);

async function loadPyodideRuntime() {
  let lastError = null;
  for (const url of PYODIDE_SOURCES) {
    try {
      const module = await import(/* webpackIgnore: true */ url);
      const indexURL = url.slice(0, url.lastIndexOf('/') + 1);
      return await module.loadPyodide({ indexURL });
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error('no Pyodide build could be loaded');
}

async function init() {
  post({ type: 'status', stage: 'downloading', detail: 'fetching Python runtime' });
  pyodide = await loadPyodideRuntime();

  post({ type: 'status', stage: 'installing', detail: 'installing pymapf' });
  const response = await fetch('pymapf-bundle.json', { cache: 'no-cache' });
  if (!response.ok) throw new Error(`could not fetch pymapf bundle (${response.status})`);
  const bundle = await response.json();
  bundleVersion = bundle.version;

  for (const [path, source] of Object.entries(bundle.files)) {
    const directory = path.slice(0, path.lastIndexOf('/'));
    if (directory) pyodide.FS.mkdirTree(`/lib/${directory}`);
    pyodide.FS.writeFile(`/lib/${path}`, source);
  }

  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/lib")
import pymapf
`);

  // The bridge lives in Python so the solve loop never round-trips through JS
  // except to emit events.
  await pyodide.runPythonAsync(`
import json, time
import pymapf
from pymapf.core import Agent, GridMap, MAPFProblem
from pymapf.benchmark import _supported


def _problem(spec):
    grid = GridMap(spec["matrix"])
    agents = [Agent(a["name"], tuple(a["start"]), tuple(a["goal"])) for a in spec["agents"]]
    return MAPFProblem(grid, agents, allow_diagonals=bool(spec.get("allowDiagonals")))


class _Streamer:
    """Observer that forwards events to JS, throttled so the pipe stays sane."""

    def __init__(self, sink, every=1, max_events=4000):
        self.sink = sink
        self.every = every
        self.max_events = max_events
        self.count = 0

    def __call__(self, event):
        self.count += 1
        if self.count > self.max_events:
            return
        if event.kind in ("expand", "agent_planned") and self.count % self.every:
            return
        payload = dict(event.payload)
        paths = payload.get("paths")
        if paths:
            payload["paths"] = {n: [list(c) for c in p] for n, p in paths.items()}
        cell = payload.get("cell")
        if cell is not None:
            payload["cell"] = list(cell)
        path = payload.get("path")
        if path is not None:
            payload["path"] = [list(c) for c in path]
        self.sink(json.dumps({
            "kind": event.kind,
            "step": event.step,
            "elapsed": event.elapsed,
            **payload,
        }))


def run_solver(spec_json, sink):
    spec = json.loads(spec_json)
    problem = _problem(spec)
    algorithm = spec.get("algorithm", "cbs")
    options = _supported(algorithm, spec.get("options") or {})
    observer = _Streamer(sink, every=spec.get("every", 1))
    started = time.perf_counter()
    solution = pymapf.solve(problem, algorithm, observer=observer, **options)
    elapsed = time.perf_counter() - started
    if solution is None:
        return json.dumps({"solved": False, "runtime": elapsed, "events": observer.count})
    data = solution.as_dict()
    data.update({
        "solved": True,
        "valid": solution.is_valid(),
        "runtime": elapsed,
        "events": observer.count,
    })
    return json.dumps(data)


def run_benchmark(spec_json, progress):
    from pymapf.benchmark import scaling_study
    spec = json.loads(spec_json)
    rows = []

    def _on_result(row):
        rows.append(row.as_dict())
        progress(json.dumps(row.as_dict()))

    scaling_study(
        spec.get("scenario", "random_obstacles"),
        agent_counts=spec.get("agent_counts", [2, 4, 6, 8]),
        algorithms=spec.get("algorithms", ["cbs", "wcbs", "prioritized"]),
        seeds=spec.get("seeds", [0, 1, 2]),
        solver_kwargs=spec.get("solver_kwargs") or {},
        on_result=_on_result,
    )
    return json.dumps(rows)
`);

  const version = pyodide.runPython('pymapf.__version__');
  const solvers = pyodide.runPython('__import__("json").dumps(pymapf.available_solvers())');
  post({
    type: 'ready',
    version,
    bundleVersion,
    solvers: JSON.parse(solvers),
    python: pyodide.runPython('__import__("sys").version.split()[0]'),
  });
}

async function solve(spec) {
  const sink = (payload) => post({ type: 'event', event: JSON.parse(payload) });
  const runner = pyodide.globals.get('run_solver');
  try {
    const result = runner(JSON.stringify(spec), sink);
    post({ type: 'result', id: spec.id, result: JSON.parse(result) });
  } finally {
    runner.destroy?.();
  }
}

async function benchmark(spec) {
  const progress = (payload) => post({ type: 'bench-row', row: JSON.parse(payload) });
  const runner = pyodide.globals.get('run_benchmark');
  try {
    const rows = runner(JSON.stringify(spec), progress);
    post({ type: 'bench-done', rows: JSON.parse(rows) });
  } finally {
    runner.destroy?.();
  }
}

async function exec(code) {
  // stdout/stderr are captured per run so output never leaks between runs.
  await pyodide.runPythonAsync(`
import io, sys
_stdout, _stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = io.StringIO()
`);
  let error = null;
  try {
    await pyodide.runPythonAsync(code);
  } catch (e) {
    error = String(e.message || e);
  }
  const output = await pyodide.runPythonAsync(`
_captured = sys.stdout.getvalue()
sys.stdout, sys.stderr = _stdout, _stderr
_captured
`);

  // If the script left a Solution behind, hand it to the visual tab.
  let solution = null;
  try {
    solution = await pyodide.runPythonAsync(`
import json
_result = None
_solution = globals().get("solution")
if _solution is not None and hasattr(_solution, "as_dict"):
    _scenario = globals().get("scenario")
    _payload = _solution.as_dict()
    if _scenario is not None and hasattr(_scenario, "grid"):
        _grid = _scenario.grid
        _payload["matrix"] = [
            [0 if _grid.is_free((r, c)) else 1 for c in range(_grid.width)]
            for r in range(_grid.height)
        ]
        _payload["agents"] = [
            {"name": a.name, "start": list(a.start), "goal": list(a.goal)}
            for a in _scenario.agents
        ]
    _result = json.dumps(_payload)
_result
`);
  } catch (e) {
    solution = null;
  }

  post({
    type: 'exec-done',
    output,
    error,
    solution: solution ? JSON.parse(solution) : null,
  });
}

self.onmessage = async (message) => {
  const { type, payload } = message.data;
  try {
    if (type === 'init') await init();
    else if (type === 'solve') await solve(payload);
    else if (type === 'benchmark') await benchmark(payload);
    else if (type === 'exec') await exec(payload);
  } catch (error) {
    post({ type: 'error', context: type, message: String(error?.message || error) });
  }
};
