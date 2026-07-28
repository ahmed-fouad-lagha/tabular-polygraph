from __future__ import annotations

import argparse
from unittest.mock import patch


@patch("tabular_polygraph.dataset.downloader.is_cached")
@patch("tabular_polygraph.dataset.downloader.download")
@patch("tabular_polygraph.dataset.list_datasets")
def test_cmd_download(mock_list, mock_download, mock_is_cached):
    from tabular_polygraph.cli.download import cmd_download

    mock_list.return_value = ["adult", "credit"]
    mock_is_cached.return_value = False
    args = argparse.Namespace(dataset="adult", output=None, sample=100, force=False)
    cmd_download(args)

    mock_download.assert_called_once_with("adult", force=False, n_sample=100)
