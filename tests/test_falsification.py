"""Tests for the falsification module."""

from sdgft.falsification import (
    FalsificationBet, BETS,
    BET_W0, BET_R_TENSOR, BET_BETA_ISO,
    BET_SIGMA_8, BET_SUM_MNU,
    summary_table,
)


class TestBetDefinitions:

    def test_five_bets(self):
        """Exactly 5 scientific bets."""
        assert len(BETS) == 5

    def test_all_frozen(self):
        """Bets are immutable."""
        for bet in BETS:
            assert isinstance(bet, FalsificationBet)

    def test_all_pending(self):
        """All bets start as pending."""
        for bet in BETS:
            assert bet.confirmed is None


class TestBetW0:

    def test_predicted_value(self):
        """w_0 = -D*/3 = -67/72."""
        assert abs(BET_W0.predicted - (-0.9306)) < 0.001

    def test_uncertainty(self):
        assert BET_W0.uncertainty == 0.010

    def test_experiment(self):
        assert BET_W0.experiment == "Euclid"

    def test_falsify_lcdm(self):
        """w = -1.00 (LCDM) should falsify SDGFT."""
        assert BET_W0.is_falsified(-1.00, 0.01)

    def test_consistent_with_sdgft(self):
        """w = -0.93 should be consistent."""
        assert not BET_W0.is_falsified(-0.93, 0.01)


class TestBetRTensor:

    def test_predicted_value(self):
        assert abs(BET_R_TENSOR.predicted - 0.013) < 0.001

    def test_falsify_low(self):
        """r < 0.004 falsifies SDGFT."""
        assert BET_R_TENSOR.is_falsified(0.003, 0.001)

    def test_falsify_high(self):
        """r > 0.022 falsifies SDGFT."""
        assert BET_R_TENSOR.is_falsified(0.030, 0.001)

    def test_consistent(self):
        assert not BET_R_TENSOR.is_falsified(0.014, 0.001)


class TestBetBetaIso:

    def test_predicted_value(self):
        """beta_iso = 1/36 ~ 0.028."""
        assert abs(BET_BETA_ISO.predicted - 0.0278) < 0.001

    def test_near_zero_falsifies(self):
        """beta_iso < 0.002 falsifies SDGFT."""
        assert BET_BETA_ISO.is_falsified(0.001, 0.001)


class TestBetSigma8:

    def test_predicted_value(self):
        assert BET_SIGMA_8.predicted == 0.775

    def test_lcdm_value_falsifies(self):
        """sigma_8 = 0.811 (Planck CMB) falsifies SDGFT."""
        assert BET_SIGMA_8.is_falsified(0.811, 0.006)


class TestBetSumMnu:

    def test_predicted_value(self):
        assert abs(BET_SUM_MNU.predicted - 0.058) < 0.001

    def test_consistent_with_upper_limit(self):
        """sum_mnu < 0.12 eV: consistent."""
        assert not BET_SUM_MNU.is_falsified(0.08, 0.02)


class TestFalsificationLogic:

    def test_falsify_bounds(self):
        """Falsification bounds are +-3sigma."""
        bet = BET_W0
        assert abs(bet.falsify_low - (bet.predicted - 3 * bet.uncertainty)) < 1e-10
        assert abs(bet.falsify_high - (bet.predicted + 3 * bet.uncertainty)) < 1e-10

    def test_status_pending(self):
        """Status is PENDING when no measurement."""
        assert "PENDING" in BET_W0.status_string()

    def test_status_consistent(self):
        assert "CONSISTENT" in BET_W0.status_string(-0.93, 0.01)

    def test_status_falsified(self):
        assert "FALSIFIED" in BET_W0.status_string(-1.00, 0.01)

    def test_summary_table(self):
        """Summary table is non-empty string."""
        table = summary_table()
        assert isinstance(table, str)
        assert "w_0" in table
        assert "sigma_8" in table
        assert len(table) > 100
