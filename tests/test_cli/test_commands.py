from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from tabular_polygraph.cli.main import main
from tabular_polygraph.cli.listinfo import cmd_info, cmd_list
from tabular_polygraph.cli.validate import cmd_validate
from tabular_polygraph.cli.download import cmd_download


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


@patch("tabular_polygraph.cli.generate._prepare_generate_request")
@patch("tabular_polygraph.cli.generate._fit_generate_generator")
@patch("tabular_polygraph.cli.generate._compute_generate_report")
@patch("tabular_polygraph.cli.generate._print_generate_report")
@patch("tabular_polygraph.cli.generate._save_generated_output")
def test_cmd_generate(
    mock_save, mock_print, mock_compute, mock_fit, mock_prepare, capsys
):
    import pandas as pd

    from tabular_polygraph.cli.generate import cmd_generate

    mock_prepare.return_value = ("test.csv", "test_id", {}, [])

    class MockGenerator:
        def generate(self, rows, filters=None, seed=None):
            return pd.DataFrame({"a": [1] * rows})

    mock_fit.return_value = (MockGenerator(), pd.DataFrame(), "mock_gen")
    mock_compute.return_value = {"mock": "report"}

    args = argparse.Namespace(
        seed=42,
        rows=10,
        output="out.csv",
        hif_epochs=1,
        hif_hubs=1,
        hif_depth=1,
        target=None,
        verbose=False,
    )
    cmd_generate(args)

    mock_prepare.assert_called_once_with(args)
    mock_fit.assert_called_once()
    mock_compute.assert_called_once()
    mock_print.assert_called_once_with({"mock": "report"})
    mock_save.assert_called_once()


@patch("tabular_polygraph.cli.helpers._load_eval_frames")
@patch("tabular_polygraph.cli.helpers._apply_eval_drop_cols")
@patch("tabular_polygraph.fidelity.fidelity_report")
@patch("tabular_polygraph.fidelity.format_report")
def test_cmd_evaluate(mock_format, mock_report, mock_drop, mock_load, capsys):
    import pandas as pd

    from tabular_polygraph.cli.evaluate import cmd_evaluate

    mock_load.return_value = (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 2]}))
    mock_drop.return_value = (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 2]}))
    mock_report.return_value = {"mock": "report"}
    mock_format.return_value = "Formatted Mock Report"

    args = argparse.Namespace(
        seed=42,
        real="real.csv",
        synthetic="syn.csv",
        drop_cols=None,
        type="cross_sectional",
        target=None,
        hif_epochs=1,
        hif_hubs=1,
        hif_depth=1,
        json=False,
        output=None,
        rule_coverage=True,
        rule_threshold=0.1,
    )
    cmd_evaluate(args)

    mock_load.assert_called_once_with("real.csv", "syn.csv")
    mock_report.assert_called_once()
    mock_format.assert_called_once_with({"mock": "report"})

    captured = capsys.readouterr()
    assert "Formatted Mock Report" in captured.out


@patch("tabular_polygraph.dataset.downloader.is_cached")
@patch("tabular_polygraph.dataset.downloader.download")
@patch("tabular_polygraph.dataset.list_datasets")
def test_cmd_download(mock_list, mock_download, mock_is_cached, capsys):
    mock_list.return_value = ["adult", "credit"]
    mock_is_cached.return_value = False
    args = argparse.Namespace(dataset="adult", output=None, sample=100, force=False)
    cmd_download(args)

    mock_download.assert_called_once_with("adult", force=False, n_sample=100)
