from __future__ import annotations

import argparse
from unittest.mock import patch

from tabular_polygraph.cli.listinfo import cmd_list


def test_cmd_list(capsys):
    args = argparse.Namespace(vertical=None)
    cmd_list(args)
    captured = capsys.readouterr()
    assert "Available datasets" in captured.out


def test_list_command(capsys):
    from tabular_polygraph.cli.main import main

    with patch("sys.argv", ["tabular-polygraph", "list"]):
        main()
    captured = capsys.readouterr()
    assert "Available datasets" in captured.out
    assert "adult" in captured.out
