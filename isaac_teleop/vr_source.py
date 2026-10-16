"""ZMQ transport on the Isaac side: newest-frame VR subscriber and stereo publisher (+ a sink for tests)."""
from __future__ import annotations

from typing import Optional

import zmq

from quest_bridge.messages import StereoFrame, VRFrame


class ZmqVRSource:
    def __init__(self, addr: str = "tcp://127.0.0.1:5555", stale_after: float = 0.5):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(addr)
        self.stale_after = float(stale_after)
        self._last_rx_t: Optional[float] = None
        self.last: Optional[VRFrame] = None

    def poll(self, t: float) -> Optional[VRFrame]:
        """Newest VRFrame received since the last call, or None."""
        newest = None
        while True:
            try:
                b = self.sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            newest = b
        if newest is None:
            return None
        self.last = VRFrame.from_bytes(newest)
        self._last_rx_t = t
        return self.last

    def stale(self, t: float) -> bool:
        return self._last_rx_t is None or (t - self._last_rx_t) > self.stale_after

    def close(self) -> None:
        self.sock.close(linger=0)


class StereoPublisher:
    def __init__(self, addr: str = "tcp://127.0.0.1:5556"):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.SNDHWM, 2)
        self.sock.bind(addr)

    def publish(self, frame: StereoFrame) -> None:
        try:
            self.sock.send(frame.to_bytes(), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        self.sock.close(linger=0)


class ZmqStereoSink:
    """SUB side of StereoPublisher (used by the bridge and by tests)."""

    def __init__(self, addr: str = "tcp://127.0.0.1:5556"):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(addr)

    def poll(self) -> Optional[StereoFrame]:
        newest = None
        while True:
            try:
                newest = self.sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        return StereoFrame.from_bytes(newest) if newest is not None else None

    def close(self) -> None:
        self.sock.close(linger=0)
