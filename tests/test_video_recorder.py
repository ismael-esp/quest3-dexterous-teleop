import json
import subprocess

import numpy as np

from isaac_teleop.video_recorder import VideoRecorder


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def _frames(n, w=64, h=48):
    for i in range(n):
        f = np.full((h, w, 3), i * 8 % 255, dtype=np.uint8)
        f[:, : w // 2, 0] = 255
        yield f


def test_records_mp4_with_requested_output_rate(tmp_path):
    p = tmp_path / "vid.mp4"
    rec = VideoRecorder(p, 64, 48, fps_in=30, fps_out=60, encoder="libx264")
    for f in _frames(30):
        rec.add(f)
    assert rec.close() == p and p.exists() and p.stat().st_size > 0
    s = _probe(p)
    assert (s["width"], s["height"]) == (64, 48)
    assert s["r_frame_rate"] == "60/1"
    assert abs(float(s["duration"]) - 1.0) < 0.2          # 30 frames at 30 in-fps = 1 s
    assert rec.frames == 30


def test_bad_encoder_falls_back_to_libx264(tmp_path):
    p = tmp_path / "vid.mp4"
    rec = VideoRecorder(p, 64, 48, fps_in=30, fps_out=60, encoder="no_such_encoder_xyz")
    for f in _frames(12):
        rec.add(f)
    assert rec.close() == p and p.exists()
    assert _probe(p)["codec_name"] == "h264"


def test_close_without_frames_returns_none(tmp_path):
    p = tmp_path / "vid.mp4"
    rec = VideoRecorder(p, 64, 48, fps_in=30)
    assert rec.close() is None
    assert not p.exists()
