# Contributing to PyMAPF

Thanks for considering a contribution. PyMAPF has two goals, and every change
is judged against both:

1. **A framework you can build on.** Adding a solver, a heuristic, a swarm
   behaviour or a scenario family should take one module and one decorator,
   and get you the benchmark harness, the visualisation, the tests' brute-force
   checks and the browser playground for free.
2. **Reference implementations that explain.** Multi-agent planning papers are
   dense. The code here is meant to be the readable version: every algorithm
   cites the paper it comes from, says where it deviates, and is tested
   against something that is known to be right.

That second goal is why we care more about clarity and correctness than about
raw speed. A solver that is 3x faster but no longer readable alongside its
paper is not an improvement here.

## Ways to contribute

- **Report a bug.** Open an issue with the *bug report* template. A minimal
  reproduction (a grid, a few agents, the solver name) is worth more than any
  description.
- **Add an algorithm.** Open an issue with the *new algorithm* template first,
  so we can agree on scope before you write code. See
  [Adding an algorithm](#adding-an-algorithm) below.
- **Improve an explanation.** Clearer docstrings, a better README example, a
  figure, a correction to [REFERENCES.md](REFERENCES.md) or to the
  [survey](.docs/survey-v2.md). These are first-class contributions.
- **Tell us you used it.** If PyMAPF appears in a paper, thesis, course or
  project, open an issue with the *used in research or teaching* template.
  This matters more than it sounds: it is the evidence a software journal asks
  for (see [Publication](README.md#publication)).

## Development setup

PyMAPF supports Python 3.10 to 3.13.

```bash
git clone https://github.com/openplan-labs/pymapf && cd pymapf
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
```

`pymapf/core`, `pymapf/algorithms`, `pymapf/scenarios.py` and
`pymapf/benchmark.py` need nothing but the standard library. If you only work
there, `pip install -e ".[dev]"` is enough.

## Checks to run before opening a pull request

```bash
# The lint gate CI enforces (syntax errors and undefined names)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv

# The test suite. Always headless: there is no display in CI.
MPLBACKEND=Agg pytest
```

The full suite takes a few minutes, most of it in `tests/test_nmpc.py`. While
iterating, run the file you are touching, e.g. `pytest tests/test_cbs.py`.

Formatting is handled by [black](https://black.readthedocs.io/): a workflow
reformats `main` after every merge, so you do not have to, but running
`black .` locally keeps your diff readable.

## Adding an algorithm

The bar is "a student could read this next to the paper and follow it".
Concretely, a new solver should come with:

1. **One module** in `pymapf/algorithms/` (centralized MAPF) or the matching
   package for other families. Its module docstring cites the paper and states
   plainly anything you simplified or chose yourself.
2. **Registration** with `@register_solver("name")` (or the family's own
   registry), so `pymapf.solve(problem, "name")`, the benchmark harness and the
   playground can find it.
3. **An entry in [REFERENCES.md](REFERENCES.md)** with the citation, and an
   entry under *Implementation notes* if the implementation deviates from the
   paper.
4. **Tests** that check the guarantee the algorithm claims, not just that it
   runs:
   - every returned solution is valid (`solution.is_valid()`, no conflicts);
   - optimal solvers match a brute-force or known-optimal cost on small
     instances; bounded-suboptimal ones stay within their bound;
   - the solver is deterministic for a fixed seed;
   - unsolvable instances fail cleanly instead of looping.
5. **A changelog entry** under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

Some conventions that keep the project usable everywhere:

- **No third-party imports in `core`, `algorithms`, `scenarios.py` or
  `benchmark.py`.** The browser playground loads these exact files into
  Pyodide. numpy and friends belong in the optional extras.
- **No `plt.show()` in library code.** Return figures; let the caller decide.
- **Randomness takes a seed.** Results in this project are meant to be
  reproducible.
- If you change a solver in `core` or `algorithms`, regenerate the playground
  bundle with `python scripts/build_web_bundle.py`. If you change the *search
  behaviour* of a solver that also exists in `.docs/mapf.js`, update the
  JavaScript port too: the two must return the same optimal costs.

The [Adding your own solver](README.md#adding-your-own-solver) section of the
README shows the minimum working example.

## Pull request process

1. Fork the repository and create a branch from `main`.
2. Keep each pull request to one concern. Two small PRs are reviewed faster
   than one large one.
3. Fill in the pull request template. Link the issue it addresses.
4. CI must pass on every supported Python version.
5. A maintainer reviews and merges. We try to give a first response within a
   week. If you have not heard back, a polite ping on the PR is welcome.

## AI-assisted contributions

Using AI coding assistants is fine. In return we ask that you:

- say so in the pull request description, naming the tool;
- understand every line you submit and be able to explain it in review;
- have run the checks above yourself.

Part of this project was written with AI assistance and is disclosed the same
way, both in the git history and in any paper about the software.

## Code of conduct

Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
