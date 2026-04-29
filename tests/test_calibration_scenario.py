import numpy as np
import pandas as pd
import pytest

from tabular_polygraph.calibration.scenario import apply_scenario, list_scenarios


def test_apply_scenario_builtin():
    # recession affects gdp_growth_yoy, unemployment_rate, etc.
    df = pd.DataFrame(
        {
            "gdp_growth_yoy": [1.0, 1.5, 2.0, 2.5, 3.0] * 20,
            "unemployment_rate": [4.0, 4.5, 5.0, 5.5, 6.0] * 20,
            "other": [1, 2, 3, 4, 5] * 20,
        }
    )

    # Recession target mean for gdp is -2.5
    res = apply_scenario(df, "recession", intensity=1.0)
    assert res["gdp_growth_yoy"].mean() == pytest.approx(-2.5)
    assert res["unemployment_rate"].mean() == pytest.approx(8.5)
    assert (res["other"] == df["other"]).all()


def test_apply_scenario_custom():
    # Use enough rows to stabilize std across ddof differences
    df = pd.DataFrame({"val": np.linspace(10, 30, 200)})
    custom = {"val": {"target_mean": 50.0, "target_std": 2.0}}

    res = apply_scenario(df, custom, intensity=1.0)
    assert res["val"].mean() == pytest.approx(50.0)
    # std in apply_scenario uses ddof=0, pandas uses ddof=1. With 200 rows it should be close.
    assert res["val"].std() == pytest.approx(2.0, rel=0.05)


def test_apply_scenario_intensity():
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0] * 20})
    custom = {"val": {"target_mean": 50.0}}

    # 0.5 intensity should be halfway between 20.0 and 50.0 -> 35.0
    res = apply_scenario(df, custom, intensity=0.5)
    assert res["val"].mean() == pytest.approx(35.0)


def test_apply_scenario_scale_factor():
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0] * 20})
    custom = {"val": {"scale_factor": 1.1}}  # +10%

    res = apply_scenario(df, custom, intensity=1.0)
    assert res["val"].iloc[0] == pytest.approx(11.0)


def test_apply_scenario_target_rate():
    # Use continuous data so percentile is smooth
    df = pd.DataFrame({"default": np.linspace(0, 1, 100)})
    custom = {"default": {"target_rate": 0.8}}

    res = apply_scenario(df, custom, intensity=1.0)
    # Threshold at 20th percentile of [0, 1] is 0.2.
    # arr >= 0.2 on [0, 1] gives 80% ones.
    assert res["default"].mean() == pytest.approx(0.8)


def test_apply_scenario_unknown():
    df = pd.DataFrame({"a": [1] * 60})
    with pytest.raises(ValueError, match="Unknown scenario"):
        apply_scenario(df, "non_existent")


def test_list_scenarios():
    info = list_scenarios()
    assert isinstance(info, pd.DataFrame)
    assert "recession" in info["name"].values
