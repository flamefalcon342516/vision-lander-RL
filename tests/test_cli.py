"""CLI: every subcommand parses; the protobuf backend is forced on import."""
import os

from lander.cli import build_parser


def test_parser_knows_all_commands() -> None:
    p = build_parser()
    assert p.parse_args(["doctor"]).command == "doctor"
    assert p.parse_args(["train", "--run", "x"]).run == "x"
    assert p.parse_args(["sim", "fly-check"]).action == "fly-check"


def test_protobuf_backend_forced_to_python() -> None:
    import lander  # noqa: F401  (import has the side effect under test)

    assert os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] == "python"


def test_deploy_and_eval_parse() -> None:
    p = build_parser()
    assert p.parse_args(["deploy", "rehearse", "--kind", "circle"]).kind == "circle"
    assert p.parse_args(["eval", "--policy", "pid", "--out", "x"]).stage == "fast"
