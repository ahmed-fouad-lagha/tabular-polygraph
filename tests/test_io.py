import os
import tempfile

import pandas as pd
import pytest

class TestIO:
    def test_csv_roundtrip(self, syn_hmda):
        from src.io import write, read
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            write(syn_hmda, path)
            reloaded = read(path)
            assert len(reloaded) == len(syn_hmda)
        finally:
            os.unlink(path)

    def test_json_roundtrip(self, syn_hmda):
        from src.io import write, read
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write(syn_hmda, path)
            reloaded = read(path)
            assert len(reloaded) == len(syn_hmda)
        finally:
            os.unlink(path)

    def test_stata_roundtrip(self, syn_hmda):
        from src.io import write, read
        with tempfile.NamedTemporaryFile(suffix=".dta", delete=False) as f:
            path = f.name
        try:
            write(syn_hmda, path)
            reloaded = read(path)
            assert len(reloaded) == len(syn_hmda)
        finally:
            os.unlink(path)

    def test_unsupported_format_raises(self, syn_hmda):
        from src.io import write
        with pytest.raises(ValueError):
            write(syn_hmda, "/tmp/test.xyz")

    def test_validate_passes_clean_data(self, hmda):
        from src.io import validate
        result = validate(hmda)
        assert result.passed
        assert len(result.errors) == 0

    def test_validate_catches_nulls(self):
        from src.io import validate
        df = pd.DataFrame({"a": [1, None, None, None, None], "b": [1, 2, 3, 4, 5]})
        # 80% nulls in 'a' — should fail with default threshold 0.3
        result = validate(df, min_rows=3)
        assert not result.passed
        assert any("a" in e for e in result.errors)

    def test_validate_warns_constant_column(self):
        from src.io import validate
        df = pd.DataFrame({"a": [1]*100, "b": range(100)})
        result = validate(df)
        assert any("constant" in w.lower() for w in result.warnings)

    def test_validate_warns_high_cardinality(self):
        from src.io import validate
        df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(200)],
            "val": range(200),
        })
        result = validate(df, max_cardinality=50, min_rows=100)
        assert any("cardinality" in w.lower() for w in result.warnings)

    def test_supported_formats_list(self):
        from src.io import supported_formats
        fmts = supported_formats()
        assert "csv"   in fmts
        assert "json"  in fmts
        assert "stata" in fmts


