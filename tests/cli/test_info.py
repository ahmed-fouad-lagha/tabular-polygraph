from __future__ import annotations

import argparse
from unittest.mock import patch

from tabular_polygraph.cli.listinfo import cmd_info


def test_cmd_info(capsys):
    args = argparse.Namespace(dataset="adult")
    cmd_info(args)
    captured = capsys.readouterr()
    assert "Adult Census Income" in captured.out


def test_info_command(capsys):
    from tabular_polygraph.cli.main import main

    with patch("sys.argv", ["tabular-polygraph", "info", "adult"]):
        main()
    captured = capsys.readouterr()
    assert "Adult Census Income" in captured.out
