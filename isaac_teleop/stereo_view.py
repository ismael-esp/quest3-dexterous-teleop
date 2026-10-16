"""Two Isaac Lab Camera sensors that follow the operator's head, encoded as JPEG (+ HUD) for the headset."""
from __future__ import annotations

import math
import time
from typing import List

import cv2
import numpy as np
import torch

from quest_bridge.messages import StereoFrame
from .frames import mat_to_quat_wxyz


class StereoView:
    APERTURE = 20.955   # horizontal aperture [mm]; hfov = 2 atan(APERTURE / (2 focal_length))

    @classmethod
    def focal_length_for_hfov(cls, hfov_deg: float) -> float:
        return cls.APERTURE / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))

    def __init__(self, sim, width: int = 640, height: int = 480, ipd: float = 0.063,
                 focal_length: float = 10.5, near: float = 0.05, far: float = 20.0, jpeg_quality: int = 80):
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        self.sim = sim
        self.hfov_deg = math.degrees(2.0 * math.atan(self.APERTURE / (2.0 * float(focal_length))))
        self.ipd = float(ipd)
        self.jpeg_quality = int(jpeg_quality)
        base = CameraCfg(
            prim_path="/World/VRCamL", update_period=0.0, height=int(height), width=int(width),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=float(focal_length), horizontal_aperture=self.APERTURE,
                                             clipping_range=(float(near), float(far))),
        )
        self.cam_l = Camera(base)
        self.cam_r = Camera(base.replace(prim_path="/World/VRCamR"))
        for cam in (self.cam_l, self.cam_r):
            if not cam.is_initialized:
                cam._initialize_callback(None)      # timeline is already playing -> no PLAY event will come
            if not cam.is_initialized:
                raise RuntimeError("VR camera failed to initialise; try running with --enable_cameras")
        self.seq = 0
        self._device = torch.device(str(sim.device)) if hasattr(sim, "device") else torch.device("cuda:0")

    def set_head_pose(self, T_world_head: np.ndarray) -> None:
        """T_world_head: (4,4), WebXR camera convention (looks -z, up +y) expressed in Isaac world."""
        R = T_world_head[:3, :3]; p = T_world_head[:3, 3]
        x_axis = R[:, 0]
        q = mat_to_quat_wxyz(R)
        for cam, sgn in ((self.cam_l, -0.5), (self.cam_r, +0.5)):
            pos = torch.as_tensor(p + sgn * self.ipd * x_axis, dtype=torch.float32, device=self._device).view(1, 3)
            quat = torch.as_tensor(q, dtype=torch.float32, device=self._device).view(1, 4)
            cam.set_world_poses(pos, quat, convention="opengl")

    def capture(self, dt: float, hud_lines: List[str]) -> StereoFrame:
        imgs = []
        for cam in (self.cam_l, self.cam_r):
            cam.update(dt, force_recompute=True)
            rgb = cam.data.output["rgb"][0]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            y = 28
            for line in hud_lines:
                cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y += 30
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise RuntimeError("JPEG encode failed")
            imgs.append(buf.tobytes())
        self.seq += 1
        return StereoFrame(seq=self.seq, t_wall=time.time(), left_jpeg=imgs[0], right_jpeg=imgs[1],
                           hud="\n".join(hud_lines))

    def mean_intensity(self) -> float:
        rgb = self.cam_l.data.output["rgb"][0]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        return float(np.asarray(rgb[..., :3]).mean())
