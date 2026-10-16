"""Vuer WSS server for the Quest 3: hand/head tracking -> ZMQ PUB, ZMQ SUB stereo JPEG -> headset.

Runs in quest3_ws/.venv (see scripts/setup_bridge_env.sh). No ROS.
Headset URL:  https://<PC_IP>:<port>?ws=wss://<PC_IP>:<port>
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import socket
import time

import zmq
import zmq.asyncio
from vuer import Vuer
from vuer.schemas import Hands, HemisphereLightStage, ImageBackground, Scene

from quest_bridge.events import HEAD_AXES, head_axes_matrix, parse_hand_value, parse_head_value
from quest_bridge.messages import StereoFrame, VRFrame

QWS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# certs copied out of the robotis container by scripts/setup_bridge_env.sh (git-ignored)
DEFAULT_CERT_DIR = os.path.join(QWS, "certs")


def _host_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("1.1.1.1", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except OSError:
        return socket.gethostbyname(socket.gethostname())


class Bridge:
    def __init__(self, args):
        self.args = args
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.CONFLATE, 1); self.pub.setsockopt(zmq.SNDHWM, 2)
        self.pub.bind(args.pub)
        self.sub = None            # zmq.asyncio socket; created inside main() on the running loop
        self.head = None
        self.seq = 0
        self.n_hand = 0; self.n_img = 0; self.n_bad = 0
        self.last_err = ""; self.last_err_t = 0.0
        self.head_src = ""
        self.head_axes = args.head_axes if args.head_axes != "auto" else None   # decided from the first samples
        self._axes_votes = {k: 0.0 for k in HEAD_AXES}; self._axes_n = 0
        self.head_up_fit = float("nan")   # dot(head up axis, gravity up) -- ~1 when the convention is right
        self.session = None
        self.use_body = not bool(args.camera_head)
        # The per-eye image plane is head-locked at `distance`; size it so it subtends exactly the render
        # FOV (teleop --hfov, aspect = eye_res) -> the world appears at 1:1 scale, not as a small window.
        if args.height is not None:
            self.panel_height = float(args.height)
        else:
            vfov = 2.0 * math.atan(math.tan(math.radians(args.hfov) / 2.0) / float(args.aspect))
            self.panel_height = 2.0 * float(args.distance) * math.tan(vfov / 2.0)

    # ---- vuer handlers ----
    async def on_hand(self, event, session):
        try:
            left, right, ls, rs = parse_hand_value(event.value)
            self.seq += 1
            f = VRFrame(t_wall=time.time(), seq=self.seq, head=self.head, left=left, right=right,
                        left_state=ls, right_state=rs)
            self.pub.send(f.to_bytes(), flags=zmq.NOBLOCK)
            self.n_hand += 1
        except Exception as e:  # never kill the server on a bad packet
            self._err(f"hand: {e!r}")

    async def on_head(self, event, session):
        try:
            if event.etype == "BODY_MOVE" and self.head_axes is None:      # --head_axes auto
                raw = parse_head_value(event.etype, event.value, head_axes="camera")
                if raw is None:
                    return
                for k in HEAD_AXES:      # which convention makes the head's up axis point along gravity (+y)?
                    self._axes_votes[k] += float((raw[:3, :3] @ head_axes_matrix(k))[1, 1])
                self._axes_n += 1
                if self._axes_n >= 60:
                    self.head_axes = max(self._axes_votes, key=self._axes_votes.get)
                    fits = {k: round(v / self._axes_n, 2) for k, v in self._axes_votes.items()}
                    print(f"[bridge] head axes auto-detected: {self.head_axes!r} (up-fit per convention {fits}); "
                          f"pass --head_axes to override", flush=True)
                return
            m = parse_head_value(event.etype, event.value, head_axes=self.head_axes or "camera")
            if m is not None:
                self.head = m
                self.head_src = "body" if event.etype == "BODY_MOVE" else "cam"
                self.head_up_fit = float(m[1, 1])
        except Exception as e:
            self._err(f"head: {e!r}")

    def _err(self, msg: str):
        self.n_bad += 1
        self.last_err = msg; self.last_err_t = time.time()

    # ---- image + status loop ----
    async def main(self, session):
        self.session = session
        children = [HemisphereLightStage(key="light-stage", hide=False),
                    Hands(fps=self.args.hz, stream=True, key="hands",
                          hideLeft=not self.args.show_hands, hideRight=not self.args.show_hands)]
        if self.use_body:
            from vuer.schemas import Body
            children.append(Body(key="body_tracking", stream=True, fps=self.args.hz, leftHand=False,
                                 rightHand=False, hideIndicate=True, showFrame=False, showBody=False))
        session.set @ Scene(bgChildren=children)
        actx = zmq.asyncio.Context.instance()
        self.sub = actx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1); self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.connect(self.args.sub)
        period = 1.0 / float(self.args.image_fps)
        last_status = time.time(); last_hand = self.n_hand; last_img = self.n_img
        while True:
            t0 = time.time()
            try:
                b = await asyncio.wait_for(self.sub.recv(), timeout=period)
                frame = StereoFrame.from_bytes(b)
                await self._show(session, frame)
                self.n_img += 1
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                self._err(f"img: {e!r}")
            if time.time() - last_status >= 3.0:
                now = time.time(); dt = now - last_status
                if self.head is not None:
                    head = f"ok({self.head_src}, up-fit {self.head_up_fit:+.2f})"
                    if self.head_up_fit < 0.7:
                        head += " <-- head axes look wrong (try --head_axes camera/body)"
                elif self.use_body and self.head_axes is None:
                    head = f"detecting axes ({self._axes_n}/60)" if self._axes_n else "none (body tracking off in Quest settings?)"
                else:
                    head = "none"
                # only show an error that happened within the last status window (not sticky)
                err = f" | last error {now - self.last_err_t:.0f}s ago: {self.last_err}" if now - self.last_err_t < dt else ""
                print(f"[bridge] hands {(self.n_hand - last_hand) / dt:5.1f} Hz | images {(self.n_img - last_img) / dt:5.1f} Hz"
                      f" | head {head}{err}", flush=True)
                last_status = now; last_hand = self.n_hand; last_img = self.n_img
            await asyncio.sleep(max(0.0, period - (time.time() - t0)))

    async def _show(self, session, frame: StereoFrame):
        a = self.args
        # session.upsert(...) is synchronous in vuer 0.1.6 (returns None) -- do not await it.
        for key, layer, jpg in (("bg_left", 1, frame.left_jpeg), ("bg_right", 2, frame.right_jpeg)):
            session.upsert(
                # NB: the client places the plane at camera + (position.z - distanceToCamera); passing both
                # position=[0,0,-d] and distanceToCamera=d (as robotis does) puts it at 2d = half the angular size.
                ImageBackground(src=jpg, key=key, layers=layer, distanceToCamera=a.distance, aspect=a.aspect,
                                height=self.panel_height, position=[0, 0, 0], format="jpeg", interpolate=True),
                to="bgChildren",
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--cert", default=os.path.join(DEFAULT_CERT_DIR, "cert.pem"))
    ap.add_argument("--key", default=os.path.join(DEFAULT_CERT_DIR, "key.pem"))
    ap.add_argument("--pub", default="tcp://127.0.0.1:5555", help="ZMQ PUB address for VRFrame")
    ap.add_argument("--sub", default="tcp://127.0.0.1:5556", help="ZMQ SUB address for StereoFrame")
    ap.add_argument("--hz", type=int, default=60, help="hand tracking stream fps requested from the client")
    ap.add_argument("--image_fps", type=float, default=30.0)
    ap.add_argument("--distance", type=float, default=2.0, help="head-locked image plane distance [m]")
    ap.add_argument("--aspect", type=float, default=4.0 / 3.0, help="image aspect (= teleop --eye_res)")
    ap.add_argument("--hfov", type=float, default=105.0,
                    help="horizontal FOV [deg] of the Isaac eye cameras (= teleop --hfov); sets the plane size")
    ap.add_argument("--height", type=float, default=None, help="override the plane height [m] (default: from --hfov)")
    ap.add_argument("--show_hands", action="store_true",
                    help="draw the tracked hand skeletons in the headset (default hidden: they never coincide with "
                         "the robot hands in the image and clutter the view; the HUD's 'track L+ R+' shows tracking)")
    ap.add_argument("--head_axes", choices=["auto"] + sorted(HEAD_AXES), default="auto",
                    help="axis convention of the body-tracking head joint (auto = pick the one whose up axis "
                         "matches gravity during the first second)")
    ap.add_argument("--body_head", action="store_true",
                    help="(default, kept for compatibility) head from WebXR body tracking 'head' joint (BODY_MOVE)")
    ap.add_argument("--camera_head", action="store_true",
                    help="do not request body tracking; rely on CAMERA_MOVE only (this vuer client never sends "
                         "CAMERA_MOVE from the XR headset pose, so the head stays 'none' in VR)")
    args = ap.parse_args()

    for p in (args.cert, args.key):
        if not os.path.exists(p):
            raise SystemExit(f"missing TLS file {p} (run scripts/setup_bridge_env.sh, or pass --cert/--key)")
    ip = _host_ip()
    app = Vuer(host="0.0.0.0", port=args.port, cert=args.cert, key=args.key,
               queries={"grid": False, "reconnect": True}, queue_len=3)
    bridge = Bridge(args)
    app.add_handler("HAND_MOVE")(bridge.on_hand)
    app.add_handler("BODY_MOVE")(bridge.on_head)     # XR head pose comes from body tracking (Body component)
    app.add_handler("CAMERA_MOVE")(bridge.on_head)   # desktop OrbitControls fallback
    print(f"[bridge] open on the Quest browser:  https://{ip}:{args.port}?ws=wss://{ip}:{args.port}", flush=True)
    print("[bridge] on the page press the 'Virtual Reality' button (NOT 'Pass-through' and NOT the browser's own "
          "'Enter AR' prompt) for a fully immersive view", flush=True)
    print(f"[bridge] eye image plane: {bridge.panel_height:.2f} m high x {bridge.panel_height * args.aspect:.2f} m wide "
          f"at {args.distance:.1f} m (hfov {args.hfov:.0f} deg)", flush=True)
    print(f"[bridge] VRFrame PUB {args.pub} | StereoFrame SUB {args.sub}", flush=True)
    app.spawn(start=True)(bridge.main)


if __name__ == "__main__":
    main()
