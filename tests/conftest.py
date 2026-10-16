import os
import sys

QWS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if QWS not in sys.path:
    sys.path.insert(0, QWS)

from isaac_teleop.paths import sim_root  # noqa: E402

# objects/, trl_utils/ of the handover env ($THESIS_WS, ../thesis_ws or the bundled ./sim)
if sim_root() not in sys.path:
    sys.path.insert(1, sim_root())
