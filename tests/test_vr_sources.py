import time

import numpy as np
import zmq

from quest_bridge.messages import VRFrame, StereoFrame
from isaac_teleop.fake_vr import FakeVRSource, synth_hand_xr25
from isaac_teleop.vr_source import ZmqVRSource, StereoPublisher, ZmqStereoSink
from isaac_teleop.xr_hand import xr25_to_mp21


def test_synth_hand_geometry():
    T = np.eye(4); T[:3, 3] = [0.2, 1.0, -0.4]
    h_open = synth_hand_xr25(T, close=0.0, side="right")
    h_fist = synth_hand_xr25(T, close=1.0, side="right")
    assert h_open.shape == (25, 4, 4)
    np.testing.assert_allclose(h_open[0], T)
    # tips further from the wrist when open than in a fist
    d = lambda h: np.linalg.norm(h[[9, 14, 19, 24], :3, 3] - h[0, :3, 3], axis=1).mean()
    assert d(h_open) > d(h_fist) + 0.03
    assert np.all(np.isfinite(xr25_to_mp21(h_open)))


def test_fake_source_frames():
    src = FakeVRSource(hz=60.0)
    f0 = src.poll(t=0.0)
    assert isinstance(f0, VRFrame) and f0.head.shape == (4, 4)
    assert f0.left.shape == (25, 4, 4) and f0.right.shape == (25, 4, 4)
    f1 = src.poll(t=0.5)
    assert f1.seq > f0.seq
    assert not np.allclose(f0.right[0, :3, 3], f1.right[0, :3, 3])   # moves
    assert src.stale(t=0.5) is False
    assert 1.4 < f0.head[1, 3] < 1.9                                  # head at standing height (VR y-up)


def test_zmq_source_receives_newest_and_reports_stale():
    port = 5599
    src = ZmqVRSource(f"tcp://127.0.0.1:{port}", stale_after=0.2)
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB); pub.setsockopt(zmq.CONFLATE, 1); pub.bind(f"tcp://127.0.0.1:{port}")
    time.sleep(0.3)                                                    # slow joiner
    for seq in range(3):
        pub.send(VRFrame(t_wall=time.time(), seq=seq, head=np.eye(4)).to_bytes())
        time.sleep(0.02)
    got = None
    for _ in range(50):
        got = src.poll(t=time.time()) or got
        if got is not None and got.seq == 2:
            break
        time.sleep(0.02)
    assert got is not None and got.seq == 2
    assert src.stale(t=time.time()) is False
    assert src.stale(t=time.time() + 1.0) is True
    src.close(); pub.close()


def test_stereo_publisher_to_sink():
    port = 5598
    sink = ZmqStereoSink(f"tcp://127.0.0.1:{port}")
    pub = StereoPublisher(f"tcp://127.0.0.1:{port}")
    time.sleep(0.3)
    got = None
    for i in range(20):
        pub.publish(StereoFrame(seq=i, t_wall=time.time(), left_jpeg=b"L", right_jpeg=b"R", hud="h"))
        time.sleep(0.02)
        got = sink.poll() or got
        if got is not None:
            break
    assert got is not None and got.left_jpeg == b"L"
    pub.close(); sink.close()
