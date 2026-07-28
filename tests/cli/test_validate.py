import argparse

from tabular_polygraph.cli.validate import cmd_validate


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
