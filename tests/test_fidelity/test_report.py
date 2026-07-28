import numpy as np
import pandas as pd


def test_fidelity_report_smoke():
    from tabular_polygraph.fidelity import fidelity_report, format_report

    real = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": ["x", "y"] * 50,
            "C": np.random.randint(0, 2, 100),
        }
    )
    syn = real.copy()
    report = fidelity_report(real, syn, target_col="C", hif_epochs=2, hif_hubs=2, hif_depth=4)
    assert "summary" in report
    assert "moment_matching_score" in report["summary"]
    assert "logic_score" in report["summary"]

    formatted = format_report(report)
    assert "FIDELITY REPORT" in formatted
    assert "Moment matching" in formatted


def test_format_report_with_errors():
    from tabular_polygraph.fidelity import format_report

    report = {
        "summary": {
            "rows_real": 100,
            "rows_synthetic": 100,
            "moment_matching_score": 90.0,
            "ks_score": 90.0,
            "joint_score": 90.0,
            "logic_score": 90.0,
            "elapsed_seconds": 1.5,
        },
        "logical": {"error": "Test error message"},
        "stylized_facts": {"_summary": {"applicable": False, "note": "Not applicable"}},
    }
    formatted = format_report(report)
    assert "ERROR (Test error message)" in formatted
    assert "Not applicable" in formatted


def test_format_report_with_rules():
    from tabular_polygraph.fidelity import format_report

    report = {
        "summary": {
            "rows_real": 100,
            "rows_synthetic": 100,
            "moment_matching_score": 90.0,
            "ks_score": 90.0,
            "joint_score": 90.0,
            "logic_score": 90.0,
            "elapsed_seconds": 1.5,
        },
        "logical": {
            "hif_violation_rate_pct": 5.0,
            "nic_violation_rate_pct": 2.0,
            "rule_violation_rate_pct": 3.0,
            "top_violated_rules": [
                {
                    "antecedent_repr": "A=1",
                    "consequent_feature": "B",
                    "consequent_value": "2",
                    "confidence": 0.9,
                    "violation_count": 10,
                }
            ],
            "violation_examples": [
                {
                    "row_index": 0,
                    "antecedent": "A=1",
                    "expected": "B=2",
                    "actual": "B=3",
                }
            ],
        },
    }
    formatted = format_report(report)
    assert "Top violated rules:" in formatted
    assert "IF A=1 THEN B=2" in formatted
    assert "Example violating rows:" in formatted
