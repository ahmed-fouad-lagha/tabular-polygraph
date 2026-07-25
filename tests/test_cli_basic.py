from unittest.mock import patch

import pytest

from tabular_polygraph.cli import (
    _parse_drop_cols,
    _parse_filters,
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
