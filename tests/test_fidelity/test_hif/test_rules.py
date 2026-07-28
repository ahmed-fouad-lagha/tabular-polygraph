import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.hif import mine_implication_rules, rule_violation_score


def test_mine_rules_numeric_quantization():
    df = pd.DataFrame({"A": np.linspace(0, 100, 100), "B": ["x", "y"] * 50})
    rules = mine_implication_rules(
        df, columns=["A", "B"], min_confidence=0.1, min_support=0.01
    )
    assert any(
        "A" in r["antecedent_repr"] or r["consequent_feature"] == "A" for r in rules
    )


def test_mine_implication_rules():
    df = pd.DataFrame({"A": ["x", "x", "y", "y"] * 25, "B": ["1", "1", "2", "2"] * 25})
    rules = mine_implication_rules(
        df, columns=["A", "B"], min_confidence=0.9, min_support=0.1
    )
    assert len(rules) > 0
    assert rules[0]["confidence"] >= 0.9


def test_rule_violation_score_penalizes_corruption():
    real = pd.DataFrame(
        {
            "state": ["CA", "CA", "TX", "TX", "NY", "NY"] * 40,
            "county": ["001", "001", "005", "005", "003", "003"] * 40,
            "segment": ["urban", "urban", "rural", "rural", "urban", "urban"] * 40,
        }
    )

    clean = real.sample(frac=1.0, random_state=10).reset_index(drop=True)
    bad = clean.copy()
    bad.loc[:79, "county"] = "999"

    clean_rules = rule_violation_score(
        real, clean, columns=["state", "county", "segment"]
    )
    bad_rules = rule_violation_score(
        real, bad, columns=["state", "county", "segment"]
    )

    assert clean_rules["num_rules_mined"] > 0
    assert clean_rules["rule_violation_rate"] < bad_rules["rule_violation_rate"]
    assert clean_rules["total_rule_hits"] < bad_rules["total_rule_hits"]
    assert (
        clean_rules["num_rows_with_violations"]
        < bad_rules["num_rows_with_violations"]
    )
