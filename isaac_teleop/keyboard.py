"""Non-blocking single-key reads from the launching terminal (works headless, no window)."""
from __future__ import annotations

import atexit
import os
import select
import signal
import sys
import termios
import tty
from typing import Optional

KEY_HELP = (
    "keys: A=start demo (records data+video) | F=finish demo -> then S=save / D=discard | "
    "SPACE=engage/disengage both arms | 1=left arm | 2=right arm | C=re-center (eyes -> robot head) | "
    "K=hold view | arrows=move held view (fwd/back/strafe) | O/L=held view up/down | "
    "3=video record start/stop (outside demos) | R=reset env / cancel demo | Q=quit"
)

_CSI_TOKENS = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}


class TerminalKeys:
    def __init__(self, fd: Optional[int] = None):
        self.fd = sys.stdin.fileno() if fd is None else fd
        self._saved = None
        self._prev_handlers: dict = {}

    def __enter__(self) -> "TerminalKeys":
        if os.isatty(self.fd):
            self._saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            # Isaac's SIGINT handler tears the process down without unwinding the Python stack,
            # so __exit__ never runs on Ctrl-C: restore cbreak in our own handler first, then chain.
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    self._prev_handlers[sig] = signal.signal(sig, self._on_signal)
                except (ValueError, OSError):          # not the main thread
                    pass
            atexit.register(self._restore)
        return self

    def __exit__(self, *exc) -> None:
        self._restore()
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._prev_handlers = {}
        atexit.unregister(self._restore)

    def _restore(self) -> None:
        if self._saved is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            except (OSError, termios.error):
                pass
            self._saved = None

    def _on_signal(self, signum, frame) -> None:
        self._restore()
        prev = self._prev_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
        elif prev == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def poll(self) -> Optional[str]:
        """Return one pending key or None; never blocks. Arrow keys (ESC [ A..D) come back as the
        tokens "UP"/"DOWN"/"RIGHT"/"LEFT"; other CSI sequences are swallowed; a lone ESC passes through."""
        while True:
            b = self._read_byte(0)
            if b is None:
                return None
            if b != "\x1b":
                return b
            nxt = self._read_byte(0)                    # terminals deliver the sequence atomically
            if nxt is None:
                return "\x1b"
            if nxt != "[":
                return nxt
            final = self._read_byte(0)
            while final is not None and not ("@" <= final <= "~"):  # skip CSI parameter bytes (e.g. "5" of PgUp)
                final = self._read_byte(0)
            tok = _CSI_TOKENS.get(final)
            if tok is not None:
                return tok                              # unknown sequence: swallowed, try the next key

    def _read_byte(self, timeout: float) -> Optional[str]:
        try:
            r, _, _ = select.select([self.fd], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not r:
            return None
        data = os.read(self.fd, 1)
        return data.decode(errors="ignore") if data else None
