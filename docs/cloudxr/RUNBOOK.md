# CloudXR path — runbook

Three things run: (1) the **CloudXR runtime** on the PC, (2) the **Isaac XR app** on the PC, (3) the
**CloudXR.js client** in the Quest browser. The runtime outlives the app: you can stop/restart the Isaac
app as often as you like and the headset stays connected.

## Start (fresh)
```bash
# Terminal 1 -- CloudXR runtime + WSS proxy (leave it running all session)
scripts/cloudxr_service.sh
#   first run: accept the NVIDIA CloudXR EULA (y). It prints the cloudxr.env path and "WSS proxy listening on 48322".

# Headset -- Quest browser: https://nvidia.github.io/IsaacTeleop/client/release-1.3.x
#   IP <PC_IP> -> "accept cert" link (https://<PC_IP>:48322/, Advanced -> Proceed) -> back -> Connect.
#   Hand tracking: put the controllers down (or off); Quest Settings -> Movement tracking -> Hand tracking ON.

# Terminal 2 -- Isaac XR teleop (headless; the scene is in the headset). ~60-90 s to start.
GPU=1 DEVICE=cpu scripts/run_cloudxr_teleop.sh --object rod
#   --object rod | rod_slim | tumbler picks the catalog object (one object per session).
#   wait for "[teleop] running"; keys: A=start demo, F=finish, S/N/D=save OK / save FAIL / discard,
#   SPACE=engage, C=re-center, Q=quit. Full protocol: docs/cloudxr/DEMO_COLLECTION.md
#   (scripts/run_cloudxr_smoke.sh is the view-only smoke; --anchor_pos/--anchor_yaw apply to both.)
```
Optional check without Isaac: `.venv_cloudxr/bin/python scripts/cloudxr_hand_probe.py --seconds 60`
prints head/wrist positions from the headset (run it instead of Terminal 2, not at the same time).

## Reset / recover
| Situation | Do |
|---|---|
| Want to restart the Isaac app (new scene, different anchor) | Terminal 2: `Q` (or Ctrl-C), wait for "bye"/exit, relaunch `scripts/run_cloudxr_smoke.sh …`. Headset stays connected. |
| Object fell / want a fresh episode | Terminal 2: `R` |
| Wrong object on the table | `Q`, relaunch with the right `--object` (the object is fixed per session). |
| Headset shows black / "disconnected" | In the CloudXR.js client press **Connect** again (the runtime keeps running). If it will not reconnect: Ctrl-C Terminal 1, restart `scripts/cloudxr_service.sh`, reconnect headset, restart Terminal 2. |
| `XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35)` in Terminal 2 | No headset connected yet -> Connect in the client; the app/probe retry by themselves. |
| Terminal 2 hangs on exit | kill it: `ps -eo pid,args \| grep cloudxr_render_smoke \| grep -v grep \| awk '{print $1}' \| xargs -r kill` |
| Everything wedged | Ctrl-C Terminal 1 and 2; `ps -eo pid,args \| grep -E "isaacteleop.cloudxr\|cloudxr_render_smoke" \| grep -v grep \| awk '{print $1}' \| xargs -r kill -9`; on the headset close the browser tab; start again from "Start (fresh)". EULA is remembered (`~/.cloudxr/run/eula_accepted`). |
| View is in the wrong place | relaunch with `--anchor_pos X Y 0` (where your feet should be in the cell) / `--anchor_yaw` (180 = face +y/table, 0 = face -y). In the headset, long-press the Meta button to recentre your own origin. |

## Logs
CloudXR runtime: `~/.cloudxr/logs/` (`cxr_server.*.log`, `wss.*.log`); Isaac app: stdout of Terminal 2.
