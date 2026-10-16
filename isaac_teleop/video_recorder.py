"""Stream RGB frames to ffmpeg and save an .mp4 (pure numpy + subprocess, no Isaac deps).

Frames arrive at the sim's render rate (fps_in) and the container is written at fps_out
(ffmpeg duplicates frames when fps_out > fps_in, so playback speed stays sim-time correct).
Defaults to NVENC for real-time 4K; falls back to libx264 if the encoder dies on startup.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

FALLBACK_ENCODER = "libx264"


class VideoRecorder:
    def __init__(self, path, width: int, height: int, fps_in: float, fps_out: float = 60.0,
                 encoder: str = "h264_nvenc"):
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        self.fps_in, self.fps_out = float(fps_in), float(fps_out)
        self.frames = 0
        self._proc: Optional[subprocess.Popen] = None
        self._encoder = encoder

    def _spawn(self, encoder: str) -> subprocess.Popen:
        probe = subprocess.run(["ffmpeg", "-hide_banner", "-h", f"encoder={encoder}"],
                               capture_output=True, text=True)
        if probe.returncode != 0 or "is not recognized" in probe.stdout + probe.stderr:
            if encoder == FALLBACK_ENCODER:
                raise RuntimeError(f"encoder {encoder!r} unavailable in ffmpeg")
            print(f"[video] encoder {encoder!r} unavailable; falling back to {FALLBACK_ENCODER}", flush=True)
            self._encoder = encoder = FALLBACK_ENCODER
        quality = ["-cq", "23", "-b:v", "0"] if "nvenc" in encoder else ["-crf", "20"]
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
               "-framerate", f"{self.fps_in:g}", "-i", "-",
               "-r", f"{self.fps_out:g}", "-c:v", encoder, *quality,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def add(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        assert rgb.shape == (self.height, self.width, 3), rgb.shape
        if self._proc is None:
            self._proc = self._spawn(self._encoder)
        try:
            self._proc.stdin.write(rgb.tobytes())
        except BrokenPipeError:
            err = (self._proc.stderr.read() or b"").decode(errors="ignore").strip()
            if self._encoder != FALLBACK_ENCODER and self.frames < 5:
                print(f"[video] {self._encoder} failed ({err.splitlines()[-1] if err else 'died'}); "
                      f"falling back to {FALLBACK_ENCODER}", flush=True)
                self._encoder = FALLBACK_ENCODER
                self._proc = self._spawn(self._encoder)
                self._proc.stdin.write(rgb.tobytes())
            else:
                raise RuntimeError(f"ffmpeg died: {err}") from None
        self.frames += 1

    def close(self) -> Optional[Path]:
        """Finalize the file. Returns the path if any frames were written, else None."""
        if self._proc is None:
            return None
        self._proc.stdin.close()
        ret = self._proc.wait(timeout=120)
        err = (self._proc.stderr.read() or b"").decode(errors="ignore").strip()
        self._proc = None
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with {ret}: {err}")
        return self.path
