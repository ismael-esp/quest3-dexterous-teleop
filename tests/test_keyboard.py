import os

from isaac_teleop.keyboard import TerminalKeys, KEY_HELP


def test_terminal_keys_reads_from_pty_nonblocking():
    master, slave = os.openpty()
    try:
        with TerminalKeys(fd=slave) as keys:
            assert keys.poll() is None
            os.write(master, b" ")
            os.write(master, b"s")
            assert keys.poll() == " "
            assert keys.poll() == "s"
            assert keys.poll() is None
    finally:
        os.close(master); os.close(slave)


def test_key_help_mentions_all_bindings():
    for k in ("A", "F", "SPACE", "1", "2", "3", "R", "S", "D", "Q"):
        assert k in KEY_HELP


def test_arrow_keys_decoded_as_tokens():
    master, slave = os.openpty()
    try:
        with TerminalKeys(fd=slave) as keys:
            os.write(master, b"\x1b[A\x1b[B\x1b[C\x1b[D")
            assert [keys.poll() for _ in range(4)] == ["UP", "DOWN", "RIGHT", "LEFT"]
            assert keys.poll() is None
    finally:
        os.close(master); os.close(slave)


def test_lone_escape_passes_through():
    master, slave = os.openpty()
    try:
        with TerminalKeys(fd=slave) as keys:
            os.write(master, b"\x1b")
            assert keys.poll() == "\x1b"
            os.write(master, b"\x1b[Zq")          # unknown CSI sequence is swallowed, next key survives
            assert keys.poll() == "q"
            assert keys.poll() is None
    finally:
        os.close(master); os.close(slave)


def test_key_help_mentions_view_nudge_keys():
    for k in ("K", "arrows", "O/L"):
        assert k in KEY_HELP


def test_ctrl_c_restores_terminal_even_when_python_never_unwinds():
    """Isaac's SIGINT handler kills the process without unwinding the stack, so the
    restore must happen inside TerminalKeys' own signal handler, chaining to the previous one."""
    import signal
    import termios
    calls = []
    prev = signal.signal(signal.SIGINT, lambda s, f: calls.append(s))   # stand-in for Isaac's handler
    master, slave = os.openpty()
    try:
        before = termios.tcgetattr(slave)
        assert before[3] & termios.ECHO
        with TerminalKeys(fd=slave) as keys:
            assert not (termios.tcgetattr(slave)[3] & termios.ECHO)     # cbreak: echo off
            os.kill(os.getpid(), signal.SIGINT)
            assert termios.tcgetattr(slave)[3] & termios.ECHO           # restored by our handler
            assert calls == [signal.SIGINT]                             # previous handler still ran
            assert keys.poll() is None                                  # still usable afterwards
        assert signal.getsignal(signal.SIGINT) is prev or callable(signal.getsignal(signal.SIGINT))
    finally:
        signal.signal(signal.SIGINT, prev)
        os.close(master); os.close(slave)


def test_exit_reinstates_previous_signal_handler():
    import signal
    prev = signal.getsignal(signal.SIGINT)
    master, slave = os.openpty()
    try:
        with TerminalKeys(fd=slave):
            assert signal.getsignal(signal.SIGINT) is not prev
        assert signal.getsignal(signal.SIGINT) is prev
    finally:
        signal.signal(signal.SIGINT, prev)
        os.close(master); os.close(slave)
