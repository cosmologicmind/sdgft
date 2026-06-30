"""Tests for the effective dimension D* and related quantities."""

import math
from fractions import Fraction

import pytest
from sdgft.constants import DELTA, DELTA_G, DELTA_F, DELTA_G_F, PHI, SIN2_30
from sdgft.dimension import (
    D_STAR_TREE, D_STAR_TREE_F, N_TREE, N_TREE_F,
    TWO_N_MINUS_1_TREE, TWO_N_MINUS_1_TREE_F,
    D_STAR_FP, N_FP, TWO_N_MINUS_1_FP,
    compute_d_star_fp,
    fixed_point_map,
    fixed_point_derivative,
)


class TestTreeLevel:

    def test_d_star_tree_fraction(self):
        assert D_STAR_TREE == Fraction(67, 24)

    def test_d_star_tree_float(self):
        assert abs(D_STAR_TREE_F - 67.0 / 24.0) < 1e-15

    def test_d_star_tree_from_sin2(self):
        """D* = 3 - sin^2(30) + delta."""
        assert D_STAR_TREE == 3 - SIN2_30 + DELTA_G

    def test_d_star_tree_from_delta(self):
        """D* = 3 - Delta (equivalent form)."""
        assert D_STAR_TREE == 3 - DELTA

    def test_n_tree_fraction(self):
        assert N_TREE == Fraction(67, 48)

    def test_n_tree_is_half_d_star(self):
        assert N_TREE == D_STAR_TREE / 2

    def test_two_n_minus_1_tree(self):
        assert TWO_N_MINUS_1_TREE == Fraction(43, 24)

    def test_two_n_minus_1_tree_relation(self):
        assert TWO_N_MINUS_1_TREE == 2 * N_TREE - 1


class TestFixedPoint:

    def test_d_star_fp_value(self):
        """D*_fp should be around 2.797."""
        assert abs(D_STAR_FP - 2.797) < 0.001

    def test_d_star_fp_is_fixed_point(self):
        """f(D*) should equal D* to high precision."""
        correction = DELTA_F ** (DELTA_F * DELTA_G_F)
        f_d = DELTA_F ** (-1.0 / D_STAR_FP) * PHI * correction
        assert abs(f_d - D_STAR_FP) < 1e-12

    def test_d_star_fp_from_various_starts(self):
        """Iteration converges from different starting values."""
        for d0 in [1.5, 2.0, 3.0, 5.0, 10.0]:
            result, _ = compute_d_star_fp(d0=d0)
            assert abs(result - D_STAR_FP) < 1e-10, (
                f"Failed to converge from d0={d0}: got {result}"
            )

    def test_d_star_fp_history_decreasing_error(self):
        """Iteration errors should decrease monotonically (eventually)."""
        _, history = compute_d_star_fp(d0=3.0)
        final = history[-1]
        errors = [abs(h - final) for h in history[5:]]  # skip transient
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-14

    def test_n_fp(self):
        assert abs(N_FP - D_STAR_FP / 2.0) < 1e-15

    def test_two_n_minus_1_fp(self):
        assert abs(TWO_N_MINUS_1_FP - (2.0 * N_FP - 1.0)) < 1e-15


class TestTreeVsFp:

    def test_differ_by_less_than_one_percent(self):
        """Tree and FP D* values should differ by < 1%."""
        pct = abs(D_STAR_TREE_F - D_STAR_FP) / D_STAR_FP * 100
        assert pct < 1.0

    def test_fp_greater_than_tree(self):
        """The fixed-point correction increases D*."""
        assert D_STAR_FP > D_STAR_TREE_F


class TestBanach:
    """Verifies the Banach contraction mapping theorem for the D* fixed-point flow."""

    def test_banach_self_mapping(self):
        """Prüft die Selbstabbildungseigenschaft: T([2.4, 3.6]) in [2.4, 3.6]."""
        steps = 100
        for i in range(steps + 1):
            d = 2.4 + (i / steps) * 1.2
            mapped_d = fixed_point_map(d)
            assert 2.4 <= mapped_d <= 3.6, f"Selbstabbildung verletzt bei D = {d}: T(D) = {mapped_d}"

    def test_banach_lipschitz_bound(self):
        """Verifiziert, dass die Lipschitz-Schranke q < 0.84 auf dem gesamten Intervall hält."""
        steps = 100
        for i in range(steps + 1):
            d = 2.4 + (i / steps) * 1.2
            derivative = fixed_point_derivative(d)
            assert abs(derivative) < 0.84, f"Lipschitz-Schranke verletzt bei D = {d}: |T'(D)| = {abs(derivative)}"

    def test_fixpoint_convergence_speed(self):
        """Stellt sicher, dass der Fixpunkt D* in unter 65 Iterationen Maschinengenauigkeit erreicht."""
        current_d = 3.0
        tolerance = 1e-15
        iterations = 0
        
        while iterations < 65:
            next_d = fixed_point_map(current_d)
            iterations += 1
            if abs(next_d - current_d) < tolerance:
                break
            current_d = next_d
            
        assert iterations < 65, f"Konvergenz zu langsam! Benötigte {iterations} Iterationen."
        
        # Gegenprobe mit der Core-Engine-Funktion (entpackt das Rückgabe-Tupel)
        d_star, _ = compute_d_star_fp()
        assert d_star == pytest.approx(next_d, abs=1e-15)


