import subprocess
import sys
import textwrap


def _run(code: str, *, path: str | None = None) -> str:
    """Run ``code`` in a fresh interpreter so imports and signal handlers start clean."""
    prefix = f"import sys; sys.path.insert(0, {path!r})\n" if path else ""
    result = subprocess.run(
        [sys.executable, "-c", prefix + textwrap.dedent(code)], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_interfaces_loads_no_interface_module():
    out = _run(
        """
        import sys
        import rio_hw.interfaces
        print(sorted(m for m in sys.modules if m.startswith("rio_hw.interfaces.")))
        """
    )
    assert out == "[]"


def test_an_interface_is_loaded_on_first_access():
    out = _run(
        """
        import sys
        from rio_hw import interfaces
        client = interfaces.SshKeyboardClient
        print(client.__module__, "rio_hw.interfaces.avp_stream" in sys.modules)
        """
    )
    assert out == "rio_hw.interfaces.ssh_keyboard False"


def test_an_unknown_name_raises_attribute_error():
    out = _run(
        """
        from rio_hw import interfaces
        print(getattr(interfaces, "NopeClient", "missing"), "SshKeyboardServer" in dir(interfaces))
        """
    )
    assert out == "missing True"


def test_avp_stream_import_keeps_the_process_signal_handlers(tmp_path):
    # A stand-in for the real package: it hijacks SIGINT/SIGTERM at import, as avp_stream does.
    fake = tmp_path / "avp_stream"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        textwrap.dedent(
            """
            import signal

            def _hijack(signum, frame):
                raise KeyboardInterrupt

            signal.signal(signal.SIGINT, _hijack)
            signal.signal(signal.SIGTERM, _hijack)
            """
        )
    )
    out = _run(
        """
        import signal
        before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        import rio_hw.interfaces.avp_stream as node
        after = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        print(node.avp_stream is not None, before == after)
        """,
        path=str(tmp_path),
    )
    assert out == "True True"
