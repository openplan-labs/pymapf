## What this changes

<!-- One or two sentences. Link the issue: "Closes #123". -->

## Why

<!-- The problem it solves, or the algorithm and paper it implements. -->

## How it was checked

<!-- Tests added or run, instances tried, figures regenerated. -->

## Checklist

- [ ] `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` passes
- [ ] `MPLBACKEND=Agg pytest` passes (or the relevant test files, listed above)
- [ ] New behaviour is tested against what it claims (validity, optimality or bound, determinism)
- [ ] Commit messages follow Conventional Commits (`fix(scope): ...`, `feat(scope): ...`, `!` for breaking); they decide the version bump
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`
- [ ] New algorithms cite their paper in the module docstring and in `REFERENCES.md`
- [ ] `core`, `algorithms`, `scenarios.py` and `benchmark.py` still import nothing outside the standard library
- [ ] If a solver in `core`/`algorithms` changed: `python scripts/build_web_bundle.py` was re-run, and `.docs/mapf.js` was updated if search behaviour changed

## AI assistance

<!-- Did you use an AI coding assistant? If so, which one, and for what part? See CONTRIBUTING.md. -->
