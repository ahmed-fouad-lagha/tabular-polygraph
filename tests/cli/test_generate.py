from __future__ import annotations

from unittest.mock import patch

import pandas as pd


@patch("tabular_polygraph.cli.generate._prepare_generate_request")
@patch("tabular_polygraph.cli.generate._fit_generate_generator")
@patch("tabular_polygraph.cli.generate._compute_generate_report")
@patch("tabular_polygraph.cli.generate._print_generate_report")
@patch("tabular_polygraph.cli.generate._save_generated_output")
def test_cmd_generate(mock_save, mock_print, mock_compute, mock_fit, mock_prepare):
    import argparse

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
