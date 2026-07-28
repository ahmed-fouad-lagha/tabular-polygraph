"""CLI package — dispatches to subcommands."""

from tabular_polygraph.cli.main import main  # noqa: F401

from .download import cmd_download  # noqa: F401
from .evaluate import cmd_evaluate  # noqa: F401
from .generate import cmd_generate  # noqa: F401
from .helpers import (  # noqa: F401
    _parse_drop_cols,
    _parse_filters,
    _positive_int,
)
from .listinfo import cmd_info, cmd_list  # noqa: F401
from .validate import cmd_validate  # noqa: F401
