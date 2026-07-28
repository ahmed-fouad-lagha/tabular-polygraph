from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from tabular_polygraph.cli import (
    _parse_drop_cols,
    _parse_filters,
    _positive_int,
    cmd_info,
    cmd_list,
    cmd_validate,
    main,
)


class TestParseFilters:
    def test_empty_returns_dict(self):
        assert _parse_filters(None) == {}
        assert _parse_filters([]) == {}

    def test_min_max_filters(self):
        result = _parse_filters(["dti_min:45", "income_max:100000"])
        assert result["dti_min"] == 45.0
        assert result["income_max"] == 100000.0

    def test_exact_match_filter(self):
        result = _parse_filters(["state:CA"])
        assert result["state"] == "CA"

    def test_multi_value_filter(self):
        result = _parse_filters(["state:CA,TX,NY"])
        assert result["state"] == ["CA", "TX", "NY"]

    def test_malformed_filter_warns(self):
        with patch("tabular_polygraph.cli.warn") as mock_warn:
            _parse_filters(["no_colon_here"])
            mock_warn.assert_called_once()


class TestParseDropCols:
    def test_none_returns_empty(self):
        assert _parse_drop_cols(None) == []

    def test_deduplicates(self):
        result = _parse_drop_cols("a,b,a,c")
        assert result == ["a", "b", "c"]


def test_positive_int_argparse():
    assert _positive_int("10") == 10
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("-5")


class TestCLIMain:
    def test_no_command_prints_help(self, capsys):
        with patch("sys.argv", ["tabular-polygraph"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "list" in captured.out

    def test_list_command(self, capsys):
        with patch("sys.argv", ["tabular-polygraph", "list"]):
            main()
        captured = capsys.readouterr()
        assert "Available datasets" in captured.out
        assert "adult" in captured.out

    def test_info_command(self, capsys):
        with patch("sys.argv", ["tabular-polygraph", "info", "adult"]):
            main()
        captured = capsys.readouterr()
        assert "Adult Census Income" in captured.out


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
