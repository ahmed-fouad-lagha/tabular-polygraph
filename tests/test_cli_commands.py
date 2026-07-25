from __future__ import annotations

import argparse

import pytest

from tabular_polygraph.cli import _positive_int, cmd_info, cmd_list, cmd_validate


def test_positive_int_argparse():
    assert _positive_int("10") == 10
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("-5")


def test_cmd_list(capsys):
    args = argparse.Namespace(vertical=None)
    cmd_list(args)
    captured = capsys.readouterr()
    assert "Available datasets" in captured.out


def test_cmd_info(capsys):
    args = argparse.Namespace(dataset="adult")
    cmd_info(args)
    captured = capsys.readouterr()
    assert "Adult Census Income" in captured.out


def test_cmd_validate(tmp_path, capsys):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n5,6\n" * 20)

    args = argparse.Namespace(
        file=str(csv_file),
        null_threshold=0.3,
        dup_threshold=0.5,
        max_cardinality=500,
        min_rows=10,
    )
    cmd_validate(args)
    captured = capsys.readouterr()
    assert "Validating:" in captured.out
