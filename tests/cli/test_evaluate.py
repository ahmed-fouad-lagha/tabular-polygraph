from __future__ import annotations

import argparse
from unittest.mock import patch

import pandas as pd


@patch("tabular_polygraph.cli.helpers._load_eval_frames")
@patch("tabular_polygraph.cli.helpers._apply_eval_drop_cols")
@patch("tabular_polygraph.fidelity.fidelity_report")
@patch("tabular_polygraph.fidelity.format_report")
def test_cmd_evaluate(mock_format, mock_report, mock_drop, mock_load, capsys):
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
