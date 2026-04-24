import numpy as np
import pytest


class TestPriors:
    def test_prior_map_mean_normal(self):
        from tabular_polygraph.calibration.priors import Prior

        p = Prior("normal", mu=10.0, sigma=2.0)
        assert np.isclose(p.map_mean(12.0, 0), 10.0)
        map_val = p.map_mean(12.0, 1000000)
        assert np.isclose(map_val, 12.0, atol=0.1)

    def test_prior_map_std_lognormal(self):
        from tabular_polygraph.calibration.priors import Prior

        p = Prior("lognormal", mu=0.0, sigma=1.0)
        std = p.map_std(1.0, 10)
        assert std > 0

    def test_prior_all_means_and_stds(self):
        from tabular_polygraph.calibration.priors import Prior

        dists = {
            "normal": {"mu": 10, "sigma": 2},
            "lognormal": {"mu": 0, "sigma": 0.5},
            "beta": {"alpha": 2, "beta": 5},
            "gamma": {"alpha": 2, "beta": 1},
            "fixed": {"value": 42},
        }
        for d, params in dists.items():
            p = Prior(d, **params)
            assert p._prior_mean() is not None
            if d != "fixed":
                assert p._prior_std() is not None

    def test_prior_invalid_distribution(self):
        from tabular_polygraph.calibration.priors import Prior

        with pytest.raises(ValueError, match="Unknown distribution"):
            Prior("unsupported", mu=0)

    def test_prior_set_management(self):
        from tabular_polygraph.calibration.priors import Prior, PriorSet

        p1 = Prior("normal", mu=0, sigma=1)
        p2 = Prior("fixed", value=10)
        ps = PriorSet({"col1": p1, "col2": p2})
        assert "col1" in ps._priors
        assert ps.get("col1").distribution == "normal"
        assert ps.get("missing") is None

    def test_prior_sample_n(self):
        from tabular_polygraph.calibration.priors import Prior

        p = Prior("normal", mu=0, sigma=1)
        samples = p.sample(100, seed=42)
        assert len(samples) == 100
        assert np.abs(np.mean(samples)) < 0.5

    def test_gamma_prior_calibration(self):
        from tabular_polygraph.calibration.priors import Prior

        p = Prior("gamma", alpha=2.0, beta=0.5)
        mean = p._prior_mean()
        assert np.isclose(mean, 4.0)

    def test_beta_prior_calibration(self):
        from tabular_polygraph.calibration.priors import Prior

        p = Prior("beta", alpha=2.0, beta=2.0)
        mean = p._prior_mean()
        assert np.isclose(mean, 0.5)

    def test_fixed_prior_blending(self):
        from tabular_polygraph.calibration.priors import Prior

        p = Prior("fixed", value=3.14, strength=10.0)
        assert np.isclose(p.map_mean(99.0, 0), 3.14)
        assert p.map_mean(100.0, 1000) > 70

    def test_prior_set_load_builtin(self):
        from tabular_polygraph.calibration.priors import get_priors

        ps = get_priors("fred_macro")
        assert len(ps._priors) > 0
        assert any("cpi" in k.lower() for k in ps._priors)

    def test_prior_sampling_consistency(self):
        from tabular_polygraph.calibration.priors import Prior

        p = Prior("lognormal", mu=0.5, sigma=0.1)
        arr = p.sample(100, seed=123)
        assert len(arr) == 100
        assert np.isfinite(arr).all()

    def test_prior_set_summary(self):
        from tabular_polygraph.calibration.priors import get_priors

        ps = get_priors("world_bank")
        summary = ps.summary()
        assert isinstance(summary, list)
        cols = [r["column"] for r in summary]
        assert "gdp_per_capita" in cols

    def test_prior_regularises_small_dataset(self, all_seeds):
        from tabular_polygraph.calibration.priors import get_priors

        ps = get_priors("fred_macro")
        tiny_df = all_seeds["fred_macro"].iloc[:5]
        for col in tiny_df.columns:
            prior = ps.get(col)
            if prior:
                m = prior.map_mean(tiny_df[col].mean(), len(tiny_df))
                assert np.isfinite(m)
