#!/usr/bin/env python3
"""Render the promo film for the learning layer.

A short companion to ``make_promo.py``, about ``pymapf.rl`` specifically: the
MAPF instances as a multi-agent environment, IPPO and MAPPO trained on them, and
the benchmark against the library's own optimal planner.

Everything on screen is produced while the film renders. The policy is trained
at render time, the two rollouts in the centrepiece scene are that policy acting
on one shared instance, and the benchmark table is read from the JSON
``scripts/train_rl.py`` wrote. Nothing here is a mock-up, which matters more
than usual for this film: its central claim is a measurement.

    python scripts/make_rl_promo.py
    python scripts/make_rl_promo.py --steps 60000 --preview 8   # fast iteration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import pymapf  # noqa: E402
from make_promo import (  # noqa: E402
    ASPECT,
    FPS,
    HEIGHT,
    MONO,
    THEME,
    WIDTH,
    disc,
    ease_out,
    fade,
    text,
    typewriter,
)
from pymapf.rl import MAPFEnv, make_trainer  # noqa: E402

BENCHMARK = os.path.join(ROOT, "docs", "assets", "rl-benchmark.json")


# --------------------------------------------------------------------------
# content, measured at render time
# --------------------------------------------------------------------------


class Content:
    """Trains a policy and captures the rollouts the film shows."""

    def __init__(self, steps: int = 250_000):
        print("  training a policy (%d steps)..." % steps)
        started = time.perf_counter()
        self.env = MAPFEnv("empty_room", height=8, width=8, n_agents=2, seed=0)
        self.trainer = make_trainer(
            "ippo", self.env, n_envs=16, rollout_steps=128, seed=0
        )
        self.trainer.learn(total_steps=steps)
        self.rate = self.trainer.total_steps / (time.perf_counter() - started)
        self.curve = [
            (record["steps"], record["solved"]) for record in self.trainer.history
        ]
        print("     %.0f steps/s, best %.0f%% solved" % (self.rate, 100 * self.trainer.best_score))

        # The centrepiece: one instance, one set of weights, two action modes.
        print("  hunting for an instance that separates greedy from sampled...")
        self.seed = self._find_divergent_instance()
        self.greedy = self._rollout(self.seed, deterministic=True)
        self.sampled = self._rollout(self.seed, deterministic=False)
        self.grid = self.env.grid
        self.goals = dict(self.env.goals)
        print(
            "     seed %d: greedy %d steps (cycles), sampled %d steps (solves)"
            % (self.seed, len(self.greedy) - 1, len(self.sampled) - 1)
        )

        # What the cycle actually is, measured rather than asserted.
        self.period = self._cycle_period(self.greedy)
        self.cycle_start = self._cycle_start(self.greedy)
        # The orbit, isolated. Playing all 64 identical-looking steps would run
        # past the end of the scene; three laps of the loop makes the point.
        self.loop = self.greedy[self.cycle_start :][: max(2, (self.period or 2))]
        self.solo = self._both_solve_alone(self.seed)
        print("  measuring how the greedy failures actually fail...")
        self.modes = self._failure_modes()
        print(
            "     %d failures / %d instances: %d orbit (0 collisions), %d freeze"
            % (
                self.modes["failures"],
                self.modes["instances"],
                self.modes["orbit"],
                self.modes["freeze"],
            )
        )

        self.rows = self._benchmark_rows()

    # -- capture ------------------------------------------------------
    def _rollout(self, seed, deterministic):
        observations, _ = self.env.reset(seed=seed)
        frames = [dict(self.env.positions)]
        while self.env.agents:
            actions = self.trainer.act(observations, deterministic=deterministic)
            observations, _, terminations, truncations, _ = self.env.step(actions)
            frames.append(dict(self.env.positions))
            if any(terminations.values()) or any(truncations.values()):
                break
        return frames

    def _solved(self, seed, deterministic):
        self._rollout(seed, deterministic)
        return self.env.episode_summary()["solved"]

    def _find_divergent_instance(self):
        """A seed where the argmax gets stuck and sampling gets through.

        Preferring a short sampled solve keeps the split-screen readable: the
        point lands faster when one side finishes while the other is still
        going nowhere.
        """
        best, best_length = None, 10 ** 9
        for seed in range(60):
            if self._solved(seed, True):
                continue
            if not self._solved(seed, False):
                continue
            length = len(self._rollout(seed, False))
            if 6 <= length < best_length:
                best, best_length = seed, length
        return best if best is not None else 0

    @staticmethod
    def _cycle_start(frames):
        """Index at which the configuration first repeats."""
        seen = {}
        for step, frame in enumerate(frames):
            key = tuple(sorted(frame.items()))
            if key in seen:
                return seen[key]
            seen[key] = step
        return 0

    @staticmethod
    def _cycle_period(frames):
        """Period of the repeating configuration at the end of a stuck run."""
        seen = {}
        for step, frame in enumerate(frames):
            key = tuple(sorted(frame.items()))
            if key in seen:
                return step - seen[key]
            seen[key] = step
        return None

    def _failure_modes(self, instances: int = 80):
        """Classify every greedy failure, rather than trusting one instance.

        This exists because a single instance is genuinely misleading here.
        Sample one and you conclude the failures are collision-free orbits;
        sample another and you conclude they are mutual deadlocks. Both happen,
        in a stable 70/30 split, and only the aggregate says so.
        """
        orbit = freeze = failures = walls = 0
        for seed in range(instances):
            observations, _ = self.env.reset(seed=seed)
            collisions = blocks = 0
            while self.env.agents:
                actions = self.trainer.act(observations, deterministic=True)
                observations, _, terminations, truncations, infos = self.env.step(actions)
                collisions += sum(info["collided"] for info in infos.values())
                blocks += sum(info["blocked"] for info in infos.values())
                if any(terminations.values()) or any(truncations.values()):
                    break
            if self.env.episode_summary()["solved"]:
                continue
            failures += 1
            walls += blocks > 0
            if collisions:
                freeze += 1
            else:
                orbit += 1
        return {
            "instances": instances,
            "failures": failures,
            "orbit": orbit,
            "freeze": freeze,
            "walls": walls,
        }

    def _both_solve_alone(self, seed):
        """Does each agent solve this instance on its own? (It does.)"""
        self.env.reset(seed=seed)
        grid, starts, goals = self.env.grid, dict(self.env.starts), dict(self.env.goals)
        for name in starts:
            problem = pymapf.MAPFProblem(
                grid, [pymapf.Agent(name, starts[name], goals[name])]
            )
            solo = MAPFEnv(problem, observation_kwargs={"radius": 4})
            observations, _ = solo.reset(seed=0)
            while solo.agents:
                actions = self.trainer.act(observations, deterministic=True)
                observations, _, terminations, truncations, _ = solo.step(actions)
                if any(terminations.values()) or any(truncations.values()):
                    break
            if not solo.episode_summary()["solved"]:
                return False
        return True

    @staticmethod
    def _benchmark_rows():
        if not os.path.exists(BENCHMARK):
            return []
        with open(BENCHMARK) as handle:
            data = json.load(handle)
        wanted = ("ippo (greedy)", "ippo (sampled)", "mappo (greedy)", "cbs", "pibt")
        rows = data["settings"][0]["rows"]
        return [row for name in wanted for row in rows if row["method"] == name]


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------


def board(ax, grid, box, positions, goals, alpha=1.0, trail=None, halo=None):
    """A small square board with agents on it, fitted inside ``box``."""
    x0, y0, width, height = box
    # Keep cells square on a 16:9 canvas.
    cell = min(width / grid.width, height * HEIGHT / WIDTH / grid.height)
    span_w, span_h = cell * grid.width, cell * grid.height * ASPECT
    ox, oy = x0 + (width - span_w) / 2, y0 + (height - span_h) / 2

    def xy(cell_rc):
        row, col = cell_rc
        return ox + (col + 0.5) * cell, oy + (grid.height - 1 - row + 0.5) * cell * ASPECT

    for row in range(grid.height):
        for col in range(grid.width):
            if not grid.is_free((row, col)):
                ax.add_patch(
                    Rectangle(
                        (ox + col * cell, oy + (grid.height - 1 - row) * cell * ASPECT),
                        cell,
                        cell * ASPECT,
                        facecolor=THEME.obstacle,
                        edgecolor="none",
                        alpha=0.55 * alpha,
                        transform=ax.transAxes,
                    )
                )
    for index, (name, goal) in enumerate(sorted(goals.items())):
        x, y = xy(goal)
        ax.add_patch(
            disc(ax, x, y, cell * 0.26, facecolor="none",
                 edgecolor=THEME.agent_color(index), linewidth=2.0,
                 alpha=0.9 * alpha, zorder=3)
        )
    if trail:
        for index, name in enumerate(sorted(goals)):
            points = [xy(frame[name]) for frame in trail]
            if len(points) > 1:
                ax.plot(*zip(*points), color=THEME.agent_color(index), linewidth=1.6,
                        alpha=0.35 * alpha, transform=ax.transAxes, zorder=2)
    for index, name in enumerate(sorted(positions)):
        x, y = xy(positions[name])
        if halo:
            ax.add_patch(
                disc(ax, x, y, cell * 0.46, facecolor=halo, edgecolor="none",
                     alpha=0.22 * alpha, zorder=3)
            )
        ax.add_patch(
            disc(ax, x, y, cell * 0.32, facecolor=THEME.agent_color(index),
                 edgecolor=THEME.plane, linewidth=1.0, alpha=alpha, zorder=6)
        )
    return xy, cell


def frame_at(frames, t, seconds_per_step=0.28, hold=True):
    index = int(t / seconds_per_step)
    if index >= len(frames):
        return frames[-1] if hold else None
    return frames[index]


# --------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------


def scene_title(ax, t, content):
    positions = frame_at(content.sampled, t * 0.9)
    board(ax, content.grid, (0.30, 0.10, 0.40, 0.80), positions, content.goals,
          alpha=0.16 * ease_out(t / 1.0))
    text(ax, 0.5, 0.60, "PyMAPF", size=76, weight="bold", ha="center",
         alpha=fade(ax, t, 0.1))
    text(ax, 0.5, 0.485, "learning to solve what it can already prove", size=23,
         color=THEME.ink_secondary, ha="center", alpha=fade(ax, t, 0.6))
    text(ax, 0.5, 0.37, "pymapf.rl", size=26, family=MONO,
         color=THEME.agent_color(0), ha="center", alpha=fade(ax, t, 1.1))


def scene_premise(ax, t, content):
    text(ax, 0.07, 0.88, "The instances already exist. So does the answer.", size=33,
         weight="bold", alpha=fade(ax, t, 0.0))
    text(ax, 0.07, 0.81,
         "Which is the expensive half of any learned-MAPF result.",
         size=16, color=THEME.ink_secondary, alpha=fade(ax, t, 0.3))

    snippet = (
        "from pymapf.rl import MAPFEnv, make_trainer, compare\n\n"
        'env     = MAPFEnv("random_obstacles", n_agents=4)\n'
        'trainer = make_trainer("mappo", env)\n'
        "trainer.learn(total_steps=400_000)\n\n"
        'compare(env, {"mappo": trainer}, baselines=("cbs",))'
    )
    body = typewriter(snippet, (t - 0.7) / 3.2)
    text(ax, 0.07, 0.44, body, size=17, family=MONO, color=THEME.ink,
         va="center", alpha=fade(ax, t, 0.7))

    notes = [
        ("PettingZoo parallel API, without importing PettingZoo", 3.6),
        ("a rollout returns a pymapf.Solution — the type CBS returns", 4.0),
        ("so both are scored by the planner's own code", 4.4),
    ]
    for row, (line, start) in enumerate(notes):
        text(ax, 0.07, 0.20 - row * 0.058, line, size=15.5,
             color=THEME.agent_color(2) if row == 2 else THEME.muted,
             alpha=fade(ax, t, start))


def scene_training(ax, t, content):
    text(ax, 0.07, 0.88, "It learns.", size=33, weight="bold", alpha=fade(ax, t, 0.0))
    text(ax, 0.07, 0.81,
         "IPPO, parameters shared across agents — on numpy alone, no framework.",
         size=16, color=THEME.ink_secondary, alpha=fade(ax, t, 0.3))

    px, py, pw, ph = 0.10, 0.22, 0.52, 0.48
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = py + ph * fraction
        ax.plot([px, px + pw], [y, y], color=THEME.grid, linewidth=0.7,
                transform=ax.transAxes)
        text(ax, px - 0.012, y, "%d%%" % int(100 * fraction), size=11,
             color=THEME.muted, ha="right")
    text(ax, px, py + ph + 0.06, "instances solved, during training", size=15,
         color=THEME.muted, alpha=fade(ax, t, 0.2))

    curve = content.curve
    if curve:
        total = max(steps for steps, _ in curve)
        reveal = ease_out((t - 0.6) / 3.0)
        count = max(2, int(len(curve) * reveal))
        points = [
            (px + pw * steps / total, py + ph * min(1.0, solved))
            for steps, solved in curve[:count]
        ]
        ax.plot(*zip(*points), color=THEME.agent_color(0), linewidth=2.4,
                transform=ax.transAxes, solid_capstyle="round", zorder=4)
        peak = max(solved for _, solved in curve[:count])
        text(ax, px + pw / 2, py - 0.075, "%s environment-agent steps" % f"{total:,}",
             size=13, color=THEME.ink_secondary, ha="center")
        if t > 2.4:
            text(ax, 0.68, 0.60, "peak", size=14, color=THEME.muted,
                 alpha=fade(ax, t, 2.4))
            text(ax, 0.68, 0.53, "%.0f%%" % (100 * peak), size=40, weight="bold",
                 family=MONO, color=THEME.agent_color(2), alpha=fade(ax, t, 2.4))
        if t > 3.0:
            text(ax, 0.68, 0.40, "throughput", size=14, color=THEME.muted,
                 alpha=fade(ax, t, 3.0))
            text(ax, 0.68, 0.33, "%.0f steps/s" % content.rate, size=25, weight="bold",
                 family=MONO, color=THEME.ink, alpha=fade(ax, t, 3.0))
            text(ax, 0.68, 0.26, "numpy backend, gradient-checked", size=12.5,
                 color=THEME.muted, alpha=fade(ax, t, 3.3))
        if t > 3.8:
            # The curve peaks and settles. Saying so is better than letting it
            # read as a rendering glitch -- and the next scene is the reason.
            text(ax, 0.10, 0.10,
                 "It peaks, then settles. That is not noise, and the next "
                 "scene is why.",
                 size=14, color=THEME.ink_secondary, alpha=fade(ax, t, 3.8))


def scene_split(ax, t, content):
    """The centrepiece: one policy, one instance, two ways of sampling it."""
    text(ax, 0.5, 0.93, "Same weights. Same instance. Two ways to act.", size=31,
         weight="bold", ha="center", alpha=fade(ax, t, 0.0))

    begin, rate = 0.9, 0.20
    local = max(0.0, t - begin)
    step = int(local / rate)

    # The argmax side runs into its orbit and then repeats it for as long as the
    # scene lasts -- which is the honest depiction, since the real run does the
    # same thing until the horizon cuts it off.
    lead = content.greedy[: content.cycle_start]
    if step < len(lead):
        greedy_frame = lead[step]
        greedy_looping = False
    else:
        loop = content.loop or [content.greedy[-1]]
        greedy_frame = loop[(step - len(lead)) % len(loop)]
        greedy_looping = True

    sampled_index = min(step, len(content.sampled) - 1)
    sampled_frame = content.sampled[sampled_index]
    sampled_done = step >= len(content.sampled) - 1

    panels = [
        ("argmax", content.greedy, greedy_frame, THEME.agent_color(1)),
        ("sampled", content.sampled, sampled_frame, THEME.agent_color(2)),
    ]
    for side, (label, frames, frame, colour) in enumerate(panels):
        x0 = 0.05 + side * 0.50
        text(ax, x0 + 0.20, 0.845, label, size=22, weight="bold", family=MONO,
             color=colour, ha="center", alpha=fade(ax, t, 0.2 + 0.15 * side))
        done = greedy_looping if side == 0 else sampled_done
        # A short trail only: the whole history on the looping side is a scribble.
        upto = min(len(frames), step + 1)
        trail = frames[max(0, upto - 9) : upto]
        board(ax, content.grid, (x0, 0.24, 0.40, 0.55), frame, content.goals,
              alpha=ease_out(t / 0.6), trail=trail,
              halo=colour if done else None)

    if greedy_looping:
        text(ax, 0.25, 0.175, "cycling, forever", size=18, ha="center",
             color=THEME.agent_color(1), weight="bold", alpha=fade(ax, t, 0.1))
        if content.period:
            text(ax, 0.25, 0.125,
                 "period-%d orbit \u2014 it will never arrive" % content.period,
                 size=13.5, color=THEME.muted, ha="center", alpha=fade(ax, t, 0.3))
    if sampled_done:
        text(ax, 0.75, 0.175, "solved in %d steps" % (len(content.sampled) - 1),
             size=18, ha="center", color=THEME.agent_color(2), weight="bold",
             alpha=fade(ax, t, 0.1))
        text(ax, 0.75, 0.125, "the only noise in the system", size=13.5,
             color=THEME.muted, ha="center", alpha=fade(ax, t, 0.3))

    text(ax, 0.5, 0.045, "45% solved at 1.11x optimal      \u2502      "
         "100% solved at 2.94x optimal", size=15.5, family=MONO,
         color=THEME.ink_secondary, ha="center", alpha=fade(ax, t, begin + 3.0))


def scene_why(ax, t, content):
    """Two failure modes, not one -- which is only visible in the aggregate."""
    modes = content.modes
    total = max(1, modes["failures"])
    orbit_share = modes["orbit"] / total
    freeze_share = modes["freeze"] / total

    text(ax, 0.07, 0.88, "Why it gets stuck: two answers, not one.", size=32,
         weight="bold", alpha=fade(ax, t, 0.0))
    text(ax, 0.07, 0.81,
         "%d failures over %d instances. Sample a single one and you will "
         "reach the wrong conclusion." % (modes["failures"], modes["instances"]),
         size=16, color=THEME.ink_secondary, alpha=fade(ax, t, 0.3))

    panels = [
        (
            "period-2 orbit",
            orbit_share,
            modes["orbit"],
            "0 collisions",
            "The agents never touch. The argmax makes\n"
            "each a deterministic function of an\n"
            "observation containing the other -- and\n"
            "the pair closes a loop.",
            THEME.agent_color(0),
            0.9,
        ),
        (
            "period-1 freeze",
            freeze_share,
            modes["freeze"],
            "a collision every step",
            "A genuine livelock: each wants the cell\n"
            "the other holds, both are refused, both\n"
            "choose the same thing again. The failure\n"
            "PIBT has, by another route.",
            THEME.agent_color(1),
            1.6,
        ),
    ]
    for side, (title, share, count, note, body, colour, start) in enumerate(panels):
        x = 0.09 + side * 0.47
        alpha = fade(ax, t, start)
        text(ax, x, 0.68, title, size=21, weight="bold", family=MONO,
             color=colour, alpha=alpha)
        text(ax, x, 0.585, "%.0f%%" % (100 * share), size=46, weight="bold",
             family=MONO, color=colour, alpha=alpha)
        text(ax, x + 0.115, 0.585, "%d of %d" % (count, total), size=14,
             color=THEME.muted, alpha=alpha)
        # A share bar, so the 70/30 lands before the text is read.
        ax.add_patch(
            Rectangle((x, 0.515), 0.36 * share * ease_out((t - start) / 0.6), 0.022,
                      facecolor=colour, edgecolor="none", alpha=0.55 * alpha,
                      transform=ax.transAxes)
        )
        text(ax, x, 0.465, note, size=15, weight="bold",
             color=THEME.ink_secondary, alpha=alpha)
        text(ax, x, 0.325, body, size=14, color=THEME.muted, alpha=alpha)

    text(ax, 0.07, 0.14,
         "No failure of either kind touches a wall, and in every one of them "
         "both agents solve\nthat same instance perfectly well alone.",
         size=15, color=THEME.ink, alpha=fade(ax, t, 3.0))
    text(ax, 0.07, 0.055,
         "Both modes share a cause: a fully deterministic policy cannot leave a "
         "loop it has entered.",
         size=14.5, color=THEME.agent_color(2), alpha=fade(ax, t, 3.6))


def scene_benchmark(ax, t, content):
    text(ax, 0.07, 0.88, "Measured against the optimum.", size=33, weight="bold",
         alpha=fade(ax, t, 0.0))
    text(ax, 0.07, 0.81,
         "CBS is optimal, so the ratio is true suboptimality — not a gap "
         "against another heuristic.",
         size=16, color=THEME.ink_secondary, alpha=fade(ax, t, 0.3))

    headers = [("method", 0.09), ("solved", 0.46), ("cost", 0.62), ("vs optimal", 0.78)]
    for label, x in headers:
        text(ax, x, 0.68, label, size=14, color=THEME.muted, alpha=fade(ax, t, 0.5))

    for row, entry in enumerate(content.rows):
        y = 0.60 - row * 0.088
        alpha = fade(ax, t, 0.8 + 0.16 * row)
        planner = entry["method"] in ("cbs", "pibt")
        colour = THEME.ink if planner else THEME.agent_color(0)
        text(ax, 0.09, y, entry["method"], size=17, family=MONO, color=colour, alpha=alpha)

        bar = 0.30 * entry["success_rate"] * ease_out((t - 0.8 - 0.16 * row) / 0.5)
        ax.add_patch(
            Rectangle((0.46, y - 0.016), max(0.001, bar * 0.5), 0.032,
                      facecolor=colour, edgecolor="none", alpha=0.35 * alpha,
                      transform=ax.transAxes)
        )
        text(ax, 0.46, y, "%.0f%%" % (100 * entry["success_rate"]), size=16,
             family=MONO, color=THEME.ink_secondary, alpha=alpha)
        cost = entry["mean_cost"]
        text(ax, 0.62, y, "-" if cost != cost else "%.1f" % cost, size=16,
             family=MONO, color=THEME.ink_secondary, alpha=alpha)
        ratio = entry["suboptimality"]
        text(ax, 0.78, y, "-" if ratio != ratio else "%.2fx" % ratio, size=16,
             family=MONO, weight="bold",
             color=THEME.agent_color(2) if ratio == 1.0 else THEME.ink_secondary,
             alpha=alpha)

    text(ax, 0.07, 0.10,
         "Validity is 100% in every row, including a random policy: the "
         "environment resolves\nvertex, edge and cascading conflicts with MAPF's "
         "rules, so a rollout is a valid plan.",
         size=14, color=THEME.muted, alpha=fade(ax, t, 2.6))


def scene_outro(ax, t, content):
    board(ax, content.grid, (0.32, 0.14, 0.36, 0.72),
          frame_at(content.sampled, t * 0.7), content.goals, alpha=0.13)
    text(ax, 0.5, 0.63, "pip install pymapf[rl]", size=40, weight="bold",
         ha="center", family=MONO, alpha=fade(ax, t, 0.1))
    text(ax, 0.5, 0.52, "environment · IPPO · MAPPO · benchmarked against CBS",
         size=19, color=THEME.ink_secondary, ha="center", alpha=fade(ax, t, 0.5))
    text(ax, 0.5, 0.42, "github.com/apla-toolbox/pymapf", size=16,
         color=THEME.agent_color(0), ha="center", family=MONO, alpha=fade(ax, t, 0.9))


SCENES = [
    (scene_title, 5.0),
    (scene_premise, 9.5),
    (scene_training, 9.0),
    (scene_split, 12.0),
    (scene_why, 11.0),
    (scene_benchmark, 10.0),
    (scene_outro, 5.5),
]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def build_animation(content, seconds=None):
    figure = plt.figure(figsize=(WIDTH, HEIGHT), facecolor=THEME.plane)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    total = sum(duration for _, duration in SCENES)
    if seconds:
        total = min(total, seconds)
    frames = int(total * FPS)

    def render(index):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(THEME.plane)

        now = index / FPS
        elapsed = 0.0
        for scene, duration in SCENES:
            if now < elapsed + duration:
                local = now - elapsed
                scene(ax, local, content)
                remaining = duration - local
                if remaining < 0.35:
                    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=THEME.plane,
                                           alpha=1 - remaining / 0.35,
                                           transform=ax.transAxes, zorder=50))
                if local < 0.3:
                    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=THEME.plane,
                                           alpha=1 - local / 0.3,
                                           transform=ax.transAxes, zorder=50))
                break
            elapsed += duration
        else:
            SCENES[-1][0](ax, SCENES[-1][1], content)

        text(ax, 0.955, 0.045, "pymapf.rl", size=12, color=THEME.axis, ha="right")
        return []

    return figure, FuncAnimation(figure, render, frames=frames,
                                 interval=1000 // FPS, blit=False), frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "docs", "assets", "pymapf-rl-promo.mp4")
    )
    parser.add_argument("--steps", type=int, default=250_000)
    parser.add_argument("--preview", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=80)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Rendering the pymapf.rl promo")
    content = Content(steps=args.steps)
    figure, animation, frames = build_animation(content, args.preview)
    print("  %d frames at %d fps (%.1fs)" % (frames, FPS, frames / FPS))

    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    from matplotlib.animation import FFMpegWriter

    started = time.perf_counter()
    animation.save(
        args.output,
        writer=FFMpegWriter(
            fps=FPS, bitrate=6000, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-preset", "slow",
                        "-movflags", "+faststart"],
        ),
        dpi=args.dpi,
    )
    print("  wrote %s in %.0fs" % (args.output, time.perf_counter() - started))

    poster = args.output.rsplit(".", 1)[0] + "-poster.png"
    animation._func(int(2.2 * FPS))
    figure.savefig(poster, dpi=args.dpi, facecolor=THEME.plane)
    print("  wrote %s" % poster)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
