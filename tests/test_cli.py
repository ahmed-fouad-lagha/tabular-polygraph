import sys
from unittest.mock import patch

import pytest

from tabular_polygraph.cli import main


def test_cli_help():
    """Smoke test for the CLI help."""
    with patch.object(sys, "argv", ["tabular-polygraph", "--help"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0


def test_cli_list():
    """Smoke test for the list command."""
    with patch.object(sys, "argv", ["tabular-polygraph", "list"]):
        main()  # Should not raise
