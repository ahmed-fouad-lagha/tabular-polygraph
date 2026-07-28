from __future__ import annotations

from unittest.mock import patch

import pytest


class TestCLIMain:
    def test_no_command_prints_help(self, capsys):
        from tabular_polygraph.cli.main import main

        with patch("sys.argv", ["tabular-polygraph"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "list" in captured.out
