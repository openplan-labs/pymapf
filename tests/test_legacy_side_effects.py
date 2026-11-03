"""Importing and constructing the legacy planners must not touch global state."""

import json
import os
import subprocess
import sys
from os import path

ROOT = path.dirname(path.dirname(path.abspath(__file__)))

SCRIPT = """
import json, logging, os
import matplotlib
from pymapf.decentralized.velocity_obstacle.velocity_obstacle import (
    MultiAgentVelocityObstacle,
)
from pymapf.decentralized.nmpc.nmpc import MultiAgentNMPC
from pymapf.centralized.cooperative_astar.cooperative_astar import CooperativeAStar
from pymapf.centralized.world import World

MultiAgentVelocityObstacle()
MultiAgentNMPC()
CooperativeAStar(World(5, 5, 0))
print(json.dumps({
    "backend": matplotlib.get_backend().lower(),
    "root_handlers": len(logging.getLogger().handlers),
    "logs_dir": os.path.exists("logs"),
}))
"""


def test_legacy_planners_leave_backend_logging_and_cwd_alone(tmp_path):
    # With a display around, the modules used to force the TkAgg backend at
    # import (overriding MPLBACKEND), and every constructor created ./logs
    # and installed handlers on the root logger.
    env = dict(os.environ, DISPLAY=":0", MPLBACKEND="Agg")
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    report = json.loads(out.strip().splitlines()[-1])
    assert report == {"backend": "agg", "root_handlers": 0, "logs_dir": False}
