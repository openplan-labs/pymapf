# AGENTS.md

## Cursor Cloud specific instructions

PyMAPF is a self-contained Python multi-agent path finding library (no servers, DBs, or
external services). "Running the app" means executing a planner and rendering a matplotlib
animation, or serving the static playground in `.docs/`.

Layout:
- `pymapf/core` + `pymapf/algorithms` + `pymapf/scenarios.py` + `pymapf/benchmark.py` — the
  current framework. **Pure standard library**: no numpy, no matplotlib, and it must stay that
  way (the browser playground loads these exact files into Pyodide).
- `pymapf/viz` — matplotlib visualisation (optional `[viz]` extra).
- `pymapf/decentralized` — reactive planners (NMPC, Velocity Obstacles); needs numpy/scipy.
- `pymapf/centralized` — the legacy cooperative A* modules, kept for compatibility.
- `.docs/` — the static playground (see below).

### Environment / interpreter
- `requires-python` is `>=3.10`, and CI tests 3.10 through 3.13. `requirements.txt`
  carries three pin sets selected by `python_version` marker, precisely so every
  interpreter in that range gets a NumPy/SciPy/matplotlib release that still supports
  it — there is no longer a single pinned version that constrains you to 3.10.
- A `.venv` at the repository root is the convention here; any interpreter in the
  supported range will do.

### Headless gotchas (important)
- Always run with `MPLBACKEND=Agg` (no display in the VM).
- The centralized `CooperativeAStar.visualize()` (and `Animator.show()`) calls `plt.show()`,
  which **blocks indefinitely headless**. The A* planner itself is fast (~seconds). To render
  the centralized planner without hanging, save via the animator directly and skip `.show()`:
  `Animator(world, paths, agents, max(sim.searches_sim_times)).save("out")`.
- Under `MPLBACKEND=Agg` this is no longer a problem for the test suite:
  `plt.show()` becomes a no-op that warns `FigureCanvasAgg is non-interactive`, and
  all three tests in `tests/test_cooperative_astar_manager.py` pass in about ten
  seconds. No deselection is needed. It still matters for interactive scripts.
- The decentralized NMPC/Velocity-Obstacle `visualize()` do NOT call `plt.show()` and work
  headless (they use a `plt.pause` loop + `FuncAnimation.save`). `ffmpeg` is required for
  `.gif` output and is available in the VM.

### Timing
- NMPC simulations are heavy (scipy optimization per step): ~30-70s per `run_simulation`, so
  `tests/test_nmpc.py` takes ~3 min. Velocity-Obstacle tests take ~20s. Use generous timeouts.

### The playground (`.docs/`)
- Static site, no build step. Serve it with `python -m http.server -d .docs 8000`.
- `.docs/pymapf-bundle.json` is generated — **re-run `python scripts/build_web_bundle.py` after
  touching `pymapf/core`, `pymapf/algorithms`, `pymapf/scenarios.py` or `pymapf/benchmark.py`**,
  or the page will run stale code. CI regenerates it before deploying.
- `.docs/mapf.js` is a hand-maintained JavaScript port of the same solvers, used while Pyodide
  downloads and when it is unavailable. If you change search behaviour in Python, change it
  there too — the two are expected to return identical optimal costs.
- Pyodide loads from a CDN. In a sandbox without egress the page falls back to the JS engine;
  that path is worth testing on its own.

### Commands
- Lint (matches CI): `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv`
- Tests: `MPLBACKEND=Agg pytest`. The full suite is ~2 min; `test_nmpc.py` is most of it.
- Gallery figures: `MPLBACKEND=Agg python scripts/generate_gallery.py` (add `--fast` to skip GIFs).
- Promo film: `MPLBACKEND=Agg python scripts/make_promo.py` (~10 min; `--preview 6` while iterating).
- Run examples: `MPLBACKEND=Agg python scripts/switch_positions_nmpc.py` (NMPC, slow) and the
  library snippets in `README.md`. Note `main.py` is stale (imports a non-existent `src/` package).
