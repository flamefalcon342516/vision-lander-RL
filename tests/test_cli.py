"""Skeleton tests: the CLI parses and placeholders fail loudly, not silently."""
import pytest

from lander.cli import build_parser, main


def test_parser_knows_all_commands() -> None:
    for cmd in ["doctor", "sim", "train", "eval", "deploy"]:
        assert build_parser().parse_args([cmd]).command == cmd


def test_unimplemented_command_returns_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["train"]) == 2
    assert "not implemented yet" in capsys.readouterr().out


def test_protobuf_backend_forced_to_python() -> None:
    import os

    import lander  # noqa: F401  (import has the side effect under test)

    assert os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] == "python"
