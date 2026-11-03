---
name: Bug report
about: Something returns a wrong answer, crashes, or hangs
title: ''
labels: 'bug'
assignees: 'guilyx,sampreets3'

---

**What happened**
A clear description of the bug.

**Minimal reproduction**
The smallest script that shows it, ideally a few lines with a small grid:

```python
import pymapf

grid = pymapf.GridMap([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
problem = pymapf.MAPFProblem(grid, [
    pymapf.Agent("a", start=(0, 0), goal=(2, 2)),
])
print(pymapf.solve(problem, "cbs"))
```

**What you expected**
For a wrong answer: the cost or path you expected, and why (a paper, another solver, a hand calculation).

**Environment**
- pymapf version (`python -c "import pymapf; print(pymapf.__version__)"`):
- Python version:
- OS:
- Installed extras (`viz`, `rl`, `decentralized`, ...):

**Anything else**
Tracebacks, figures, links.
