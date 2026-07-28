from __future__ import annotations

import argparse

import pytest

from tabular_polygraph.cli.helpers import (
    _parse_drop_cols,
    _parse_filters,
    _positive_int,
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

    def test_malformed_filter_warns(self, capsys):
        _parse_filters(["no_colon_here"])
        captured = capsys.readouterr()
        assert "Skipping malformed filter" in captured.out


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
