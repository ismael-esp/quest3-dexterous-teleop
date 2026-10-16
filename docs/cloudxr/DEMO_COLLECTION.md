# Demo collection protocol

## Objects and counts
| `--object` | what | target |
|---|---|---|
| `rod` | the yellow rod (r 2.7 cm, 30 cm) | ≥ 12 saved-OK demos |
| `rod_slim` | same rod, r 1.8 cm | ≥ 12 |
| `tumbler` | BODex tumbler lying on its side (Ø 8 cm, 25 cm) | ≥ 12 |

One session = one object (`session.json` records it). Object pose is randomized per episode by the env.

## Start
```bash
scripts/cloudxr_service.sh                                   # terminal 1 (leave running)
GPU=1 DEVICE=cpu scripts/run_cloudxr_teleop.sh --object rod --session_note "operator X, day 1"   # terminal 2
```
Wait for `[teleop] palm check right: ...` and `[teleop] running`. Connect the headset (RUNBOOK.md).

## Per episode
1. `A` — env reset, recording starts (front-view mp4 too). Object is at a new random pose.
2. `SPACE` — engage both arms. Do the task: left hand grasps the LEFT end, hands over, right hand
   carries the object to the goal marker and holds it there.
3. **Release, then bring BOTH hands back to their start pose and OPEN them; hold ~2 s.** The HUD
   shows `SUCCESS` once the env latches. Without this tail the env never marks success and the
   converter cannot cut the demo at the success step.
4. `F` — finish. The terminal shows `env success=YES/no` and the stage each arm reached.
5. Label: `S` = save as **OK** (the motion was a valid, complete handover — even if imperfect or the
   env did not latch) · `N` = save as **FAIL** (kept for analysis) · `D` = discard (mistake).
   Files: `ep_NNN_ok.pickle` / `ep_NNN_fail.pickle` (+ `.mp4`), labels also inside the pickle.

Vary approach direction, where along the object you grasp, and pace between demos — the policy
should learn the task, not one trajectory. Grasp naturally.
Episode cap is 40 s (`--max_episode_seconds`); the converter cuts at env success + 0.5 s anyway.

## After a session
```bash
python scripts/convert_vr_demos.py --tag vrA --sessions demos/vr_cloudxr/handover_tshape/2026*/ --dry_run
```
Check `env-success rate` (should be > 0.8 — otherwise the home-return tail is being skipped) and the
chosen `T`. Drop `--dry_run` to write `demos/converted/Demo_success_handover_tshape_vrA.pickle` (+ `Demo_Traj_...`).
