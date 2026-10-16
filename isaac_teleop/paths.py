"""Where the simulation env and the Allegro hand URDFs come from.

The teleop drives the dual-arm handover env. It is resolved in this order:
  1. $THESIS_WS                      (any checkout that has task_registry.py, tasks/, trl_utils/, assets/)
  2. ../thesis_ws next to this repo  (the original development layout)
  3. ./sim                           (the env subset bundled with this repo; assets via scripts/download_assets.sh)
No Isaac import here, so the unit tests can use it.
"""
from __future__ import annotations

import os

QWS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sim_root() -> str:
    env = os.environ.get("THESIS_WS")
    if env:
        return os.path.abspath(env)
    sibling = os.path.join(os.path.dirname(QWS), "thesis_ws")
    if os.path.isfile(os.path.join(sibling, "task_registry.py")):
        return sibling
    return os.path.join(QWS, "sim")


def allegro_assets() -> str:
    """Directory holding allegro_{left,right}/allegro_hand_{left,right}.urdf (+ meshes)."""
    return os.environ.get("ALLEGRO_ASSETS", os.path.join(QWS, "assets", "allegro"))
