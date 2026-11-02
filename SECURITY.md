# Security policy

PyMAPF is a planning library: it runs no server and handles no credentials. The
realistic risks are in what it loads (pickled RL checkpoints, map files) and in
the browser playground.

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/pymapf/) receives
fixes.

## Reporting a vulnerability

Please do **not** open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/openplan-labs/pymapf/security/advisories/new)
or email **erwinlejeune.pro@gmail.com**. Include what you found, how to
reproduce it, and which version you used.

You should get an acknowledgement within a week. Once a fix is released, we will
credit you in the changelog unless you prefer otherwise.
