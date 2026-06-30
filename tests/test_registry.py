"""Tests for the Observable dataclass and Registry."""

import pytest

from sdgft.registry import Observable, Registry, REGISTRY


def _make_obs(name: str = "test_obs", **kwargs) -> Observable:
    """Helper to create a test observable with defaults."""
    defaults = dict(
        name=name,
        symbol="x",
        formula="test formula",
        predicted=1.0,
        observed=1.0,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=0,
        d_star_variant="none",
    )
    defaults.update(kwargs)
    return Observable(**defaults)


class TestObservable:

    def test_frozen(self):
        obs = _make_obs()
        with pytest.raises(AttributeError):
            obs.name = "changed"  # type: ignore[misc]

    def test_deviation_abs(self):
        obs = _make_obs(predicted=10.0, observed=9.5)
        assert abs(obs.deviation_abs - 0.5) < 1e-12

    def test_deviation_percent(self):
        obs = _make_obs(predicted=10.0, observed=9.5)
        assert abs(obs.deviation_percent - 100 * 0.5 / 9.5) < 1e-10

    def test_deviation_percent_zero_observed(self):
        obs = _make_obs(predicted=1.0, observed=0.0)
        assert obs.deviation_percent == 0.0

    def test_sigma_tension(self):
        obs = _make_obs(predicted=10.0, observed=9.5, observed_uncertainty=0.1)
        assert obs.sigma_tension is not None
        assert abs(obs.sigma_tension - 5.0) < 1e-10

    def test_sigma_tension_no_error(self):
        obs = _make_obs(predicted=10.0, observed=9.5, observed_uncertainty=0.0)
        assert obs.sigma_tension is None

    def test_sigma_tension_upper_limit(self):
        obs = _make_obs(predicted=0.01, observed=0.036, is_upper_limit=True,
                        observed_uncertainty=0.01)
        assert obs.sigma_tension is None

    def test_status_exact(self):
        obs = _make_obs(predicted=3.0, observed=3.0)
        assert obs.status == "exact"

    def test_status_excellent(self):
        obs = _make_obs(predicted=137.0, observed=137.036)
        assert obs.status == "excellent"

    def test_status_good(self):
        obs = _make_obs(predicted=0.050, observed=0.0493)
        assert obs.status == "good"

    def test_status_fair(self):
        obs = _make_obs(predicted=6.54e-10, observed=6.1e-10)
        assert obs.status == "fair"

    def test_status_upper_limit_compatible(self):
        obs = _make_obs(predicted=0.013, observed=0.036, is_upper_limit=True)
        assert obs.status == "compatible"

    def test_status_upper_limit_excluded(self):
        obs = _make_obs(predicted=0.05, observed=0.036, is_upper_limit=True)
        assert obs.status == "excluded"

    def test_invalid_d_star_variant(self):
        with pytest.raises(ValueError, match="d_star_variant"):
            _make_obs(d_star_variant="invalid")


class TestRegistry:

    def test_register_and_get(self):
        reg = Registry()
        obs = _make_obs(name="a")
        reg.register(obs)
        assert reg.get("a") is obs

    def test_getitem(self):
        reg = Registry()
        obs = _make_obs(name="a")
        reg.register(obs)
        assert reg["a"] is obs

    def test_contains(self):
        reg = Registry()
        reg.register(_make_obs(name="a"))
        assert "a" in reg
        assert "b" not in reg

    def test_len(self):
        reg = Registry()
        assert len(reg) == 0
        reg.register(_make_obs(name="a"))
        assert len(reg) == 1

    def test_duplicate_raises(self):
        reg = Registry()
        reg.register(_make_obs(name="a"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_make_obs(name="a"))

    def test_get_missing_raises(self):
        reg = Registry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_all_sorted_by_level(self):
        reg = Registry()
        reg.register(_make_obs(name="c", level=2))
        reg.register(_make_obs(name="a", level=0))
        reg.register(_make_obs(name="b", level=1))
        names = [o.name for o in reg.all()]
        assert names == ["a", "b", "c"]

    def test_by_level(self):
        reg = Registry()
        reg.register(_make_obs(name="a", level=0))
        reg.register(_make_obs(name="b", level=1))
        reg.register(_make_obs(name="c", level=1))
        assert len(reg.by_level(1)) == 2

    def test_by_d_star(self):
        reg = Registry()
        reg.register(_make_obs(name="a", d_star_variant="tree"))
        reg.register(_make_obs(name="b", d_star_variant="none"))
        reg.register(_make_obs(name="c", d_star_variant="tree"))
        assert len(reg.by_d_star("tree")) == 2

    def test_scorecard(self):
        reg = Registry()
        reg.register(_make_obs(name="a", predicted=3.0, observed=3.0))  # exact
        reg.register(_make_obs(name="b", predicted=137.0, observed=137.036))  # excellent
        sc = reg.scorecard()
        assert sc["exact"] == 1
        assert sc["excellent"] == 1

    def test_summary_table_nonempty(self):
        reg = Registry()
        reg.register(_make_obs(name="a"))
        table = reg.summary_table()
        assert "a" in table
        assert "Total: 1" in table

    def test_iter(self):
        reg = Registry()
        reg.register(_make_obs(name="a"))
        reg.register(_make_obs(name="b"))
        names = [o.name for o in reg]
        assert len(names) == 2

    def test_global_registry_exists(self):
        assert isinstance(REGISTRY, Registry)
