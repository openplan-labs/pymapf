import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# The centralized framework (pymapf.core, pymapf.algorithms, pymapf.scenarios,
# pymapf.benchmark) is pure standard library on purpose: that is what lets the
# solvers run in CI, on a robot, and in the browser under Pyodide with nothing
# to build. Everything needing a third-party package is an extra.
EXTRAS = {
    # plots, animations, live views and benchmark charts
    "viz": ["matplotlib>=3.4"],
    # the reactive planners (NMPC, velocity obstacles)
    "decentralized": ["numpy>=1.20", "scipy>=1.6"],
    # the legacy centralized modules kept for backwards compatibility
    "legacy": [
        "numpy>=1.20",
        "scipy>=1.6",
        "matplotlib>=3.4",
        "coloredlogs>=15.0",
        "termcolor>=1.1",
    ],
    "dev": ["pytest>=6.0", "pytest-cov"],
}
EXTRAS["all"] = sorted(
    {dep for name, deps in EXTRAS.items() if name != "dev" for dep in deps}
)

setuptools.setup(
    name="pymapf",
    version="0.6.0",
    author="Erwin Lejeune",
    author_email="erwinlejeune.pro@gmail.com",
    description=(
        "Multi-agent path finding: a solver framework (CBS, weighted CBS, "
        "prioritized planning) with scenarios, benchmarks and visualisation"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apla-toolbox/pymapf",
    project_urls={
        "Playground": "https://apla-toolbox.github.io/pymapf/",
        "Source": "https://github.com/apla-toolbox/pymapf",
        "Changelog": "https://github.com/apla-toolbox/pymapf/blob/main/CHANGELOG.md",
    },
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "scripts"]),
    install_requires=[],
    extras_require=EXTRAS,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: Pytest",
    ],
    python_requires=">=3.8",
)
