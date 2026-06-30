"""Tests for the QFT bridge: gauge groups from 24-cell geometry.

Sections:
    1.  TestVertices24Cell        — 24 vertices in canonical form
    2.  TestD4RootSystem          — 24 roots, root-system axioms
    3.  TestCartanMatrix          — D₄ Cartan matrix verification
    4.  TestPositiveRoots         — 12 positive roots
    5.  TestSMDecomposition       — D₄ → A₂ ⊕ A₁ ⊕ U(1)
    6.  TestSubsystemVerification — A₂ and A₁ sub-root-systems
    7.  TestTriality              — Z₃ outer automorphism
    8.  TestSymmetryGroups        — Weyl, outer, full automorphism
    9.  TestSDGFTConstants        — δ = 1/24, Δ = 5/24, sin²(30°)
    10. TestIsomorphism           — 24-cell ↔ D₄ proof
    11. TestSMContent             — gauge boson & matter counts
    12. TestCosetPairs            — 8 particle/antiparticle pairs
    13. TestRegistry              — observable registration
    14. TestSummary               — print_summary smoke test
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from sdgft.qft_bridge.gauge_groups import (
    # §1 24-cell
    VERTICES_24CELL,
    N_VERTICES,
    # §2 D₄ roots
    D4_ROOTS,
    N_D4_ROOTS,
    D4_SIMPLE_ROOTS,
    ALPHA_1,
    ALPHA_2,
    ALPHA_3,
    ALPHA_4,
    # §3 algebra
    inner,
    norm_sq,
    vec_add,
    vec_neg,
    cartan_matrix,
    # §4 Cartan
    D4_CARTAN_MATRIX,
    D4_CARTAN_EXPECTED,
    # §5 positive roots
    D4_POSITIVE_ROOTS,
    # §6 verification
    verify_root_system,
    # §7 decomposition
    SM_DECOMPOSITION,
    decompose_d4_to_sm,
    N_GLUONS,
    N_WEAK_BOSONS,
    N_PHOTON,
    N_GAUGE_BOSONS,
    GaugeDecomposition,
    # §8 subsystem
    verify_a2_subsystem,
    verify_a1_subsystem,
    # §9 triality
    triality_permutation,
    verify_triality,
    TRIALITY_REPS,
    # §10 symmetry
    WEYL_D4_ORDER,
    OUTER_AUT_ORDER,
    AUT_24CELL_ORDER,
    # §11 SDGFT constants
    DELTA_G_FROM_ROOTS,
    DELTA_FROM_GEOMETRY,
    DELTA_G_CONSISTENT,
    DELTA_CONSISTENT,
    EDGE_ANGLE_DEG,
    COS_EDGE_ANGLE,
    SIN2_COMPLEMENT,
    SIN2_30_CONSISTENT,
    # §12 SM content
    SM_CONTENT,
    SMContent,
    # §13 isomorphism
    verify_24cell_d4_isomorphism,
    # §14 coset
    coset_pairs,
    # §15 registry & summary
    register_all,
    print_summary,
)


# ── 1. 24-Cell Vertices ──────────────────────────────────────────

class TestVertices24Cell:
    """Test 24-cell vertex construction."""

    def test_vertex_count(self):
        assert N_VERTICES == 24

    def test_vertices_unique(self):
        assert len(set(VERTICES_24CELL)) == 24

    def test_vertex_types(self):
        """Should contain 8 axis-aligned and 16 half-integer vertices."""
        axis_aligned = [v for v in VERTICES_24CELL if sum(c != 0 for c in v) == 1]
        half_integer = [v for v in VERTICES_24CELL if sum(c != 0 for c in v) == 4]
        assert len(axis_aligned) == 8
        assert len(half_integer) == 16

    def test_axis_vertices_magnitude(self):
        """Axis-aligned vertices have one entry ±2, rest zero."""
        for v in VERTICES_24CELL:
            if sum(c != 0 for c in v) == 1:
                assert max(abs(c) for c in v) == 2

    def test_half_int_vertices_magnitude(self):
        """Half-integer vertices have all entries ±1."""
        for v in VERTICES_24CELL:
            if sum(c != 0 for c in v) == 4:
                assert all(abs(c) == 1 for c in v)


# ── 2. D₄ Root System ────────────────────────────────────────────

class TestD4RootSystem:
    """Test D₄ root system construction and axioms."""

    def test_root_count(self):
        assert N_D4_ROOTS == 24

    def test_roots_unique(self):
        assert len(set(D4_ROOTS)) == 24

    def test_root_dimension(self):
        """All roots should be 4-dimensional."""
        for r in D4_ROOTS:
            assert len(r) == 4

    def test_root_norm(self):
        """All D₄ roots have |α|² = 2."""
        for r in D4_ROOTS:
            assert norm_sq(r) == 2

    def test_exactly_two_nonzero_entries(self):
        """Each D₄ root has exactly two nonzero entries of ±1."""
        for r in D4_ROOTS:
            nonzero = [c for c in r if c != 0]
            assert len(nonzero) == 2
            assert all(abs(c) == 1 for c in nonzero)

    def test_negation_closed(self):
        rs = verify_root_system(D4_ROOTS)
        assert rs["negation_closed"]

    def test_reflection_closed(self):
        rs = verify_root_system(D4_ROOTS)
        assert rs["reflection_closed"]

    def test_crystallographic(self):
        rs = verify_root_system(D4_ROOTS)
        assert rs["crystallographic"]

    def test_simply_laced(self):
        rs = verify_root_system(D4_ROOTS)
        assert rs["simply_laced"]

    def test_simple_roots_are_roots(self):
        """The 4 simple roots must be elements of D₄."""
        root_set = set(D4_ROOTS)
        for alpha in D4_SIMPLE_ROOTS:
            assert alpha in root_set

    def test_simple_roots_count(self):
        assert len(D4_SIMPLE_ROOTS) == 4


# ── 3. Cartan Matrix ─────────────────────────────────────────────

class TestCartanMatrix:
    """Test D₄ Cartan matrix."""

    def test_cartan_shape(self):
        assert len(D4_CARTAN_MATRIX) == 4
        for row in D4_CARTAN_MATRIX:
            assert len(row) == 4

    def test_diagonal_is_two(self):
        for i in range(4):
            assert D4_CARTAN_MATRIX[i][i] == 2

    def test_matches_d4_expected(self):
        assert D4_CARTAN_MATRIX == D4_CARTAN_EXPECTED

    def test_symmetric(self):
        """Simply-laced ⟹ Cartan matrix is symmetric."""
        for i in range(4):
            for j in range(4):
                assert D4_CARTAN_MATRIX[i][j] == D4_CARTAN_MATRIX[j][i]

    def test_branch_point_structure(self):
        """α₂ (row 1) should connect to α₁, α₃, α₄ (three −1 entries)."""
        row2 = D4_CARTAN_MATRIX[1]
        off_diag = [row2[j] for j in range(4) if j != 1]
        assert off_diag.count(-1) == 3

    def test_outer_nodes_disconnected(self):
        """α₁, α₃, α₄ are mutually disconnected (off-diagonal = 0)."""
        assert D4_CARTAN_MATRIX[0][2] == 0  # α₁-α₃
        assert D4_CARTAN_MATRIX[0][3] == 0  # α₁-α₄
        assert D4_CARTAN_MATRIX[2][3] == 0  # α₃-α₄


# ── 4. Positive Roots ────────────────────────────────────────────

class TestPositiveRoots:
    """Test positive root identification."""

    def test_positive_root_count(self):
        """D₄ has n(n−1) = 12 positive roots."""
        assert len(D4_POSITIVE_ROOTS) == 12

    def test_positive_plus_negative_equals_total(self):
        """Positive + negative roots = all roots."""
        neg_roots = tuple(sorted(vec_neg(r) for r in D4_POSITIVE_ROOTS))
        all_roots = tuple(sorted(set(D4_POSITIVE_ROOTS) | set(neg_roots)))
        assert len(all_roots) == 24

    def test_simple_roots_are_positive(self):
        pos_set = set(D4_POSITIVE_ROOTS)
        for alpha in D4_SIMPLE_ROOTS:
            assert alpha in pos_set

    def test_highest_root(self):
        """Highest root of D₄ is (1,1,0,0) = α₁+2α₂+α₃+α₄."""
        highest = (1, 1, 0, 0)
        assert highest in set(D4_POSITIVE_ROOTS)
        # Verify: α₁ + 2α₂ + α₃ + α₄
        computed = vec_add(ALPHA_1, vec_add(vec_add(ALPHA_2, ALPHA_2),
                                            vec_add(ALPHA_3, ALPHA_4)))
        assert computed == highest


# ── 5. SM Decomposition ──────────────────────────────────────────

class TestSMDecomposition:
    """Test D₄ → SU(3) × SU(2) × U(1) decomposition."""

    def test_decomposition_type(self):
        assert isinstance(SM_DECOMPOSITION, GaugeDecomposition)

    def test_a2_root_count(self):
        """SU(3) sector: 6 roots."""
        assert len(SM_DECOMPOSITION.a2_roots) == 6

    def test_a1_root_count(self):
        """SU(2) sector: 2 roots."""
        assert len(SM_DECOMPOSITION.a1_roots) == 2

    def test_coset_root_count(self):
        """Matter sector: 16 coset roots."""
        assert SM_DECOMPOSITION.n_coset == 16

    def test_partition_exhaustive(self):
        """6 + 2 + 16 = 24 (all D₄ roots accounted for)."""
        total = (len(SM_DECOMPOSITION.a2_roots)
                 + len(SM_DECOMPOSITION.a1_roots)
                 + SM_DECOMPOSITION.n_coset)
        assert total == 24

    def test_partition_disjoint(self):
        """A₂, A₁, and coset root sets must be disjoint."""
        a2 = set(SM_DECOMPOSITION.a2_roots)
        a1 = set(SM_DECOMPOSITION.a1_roots)
        co = set(SM_DECOMPOSITION.coset_roots)
        assert len(a2 & a1) == 0
        assert len(a2 & co) == 0
        assert len(a1 & co) == 0

    def test_all_roots_from_d4(self):
        """All decomposition roots must be valid D₄ roots."""
        root_set = set(D4_ROOTS)
        for r in SM_DECOMPOSITION.a2_roots:
            assert r in root_set
        for r in SM_DECOMPOSITION.a1_roots:
            assert r in root_set
        for r in SM_DECOMPOSITION.coset_roots:
            assert r in root_set

    def test_n_gluons(self):
        assert N_GLUONS == 8

    def test_n_weak_bosons(self):
        assert N_WEAK_BOSONS == 3

    def test_n_photon(self):
        assert N_PHOTON == 1

    def test_n_gauge_bosons(self):
        assert N_GAUGE_BOSONS == 12

    def test_gauge_bosons_sum(self):
        assert N_GLUONS + N_WEAK_BOSONS + N_PHOTON == 12

    def test_decompose_function_matches_precomputed(self):
        """decompose_d4_to_sm() should produce same result as SM_DECOMPOSITION."""
        fresh = decompose_d4_to_sm()
        assert fresh == SM_DECOMPOSITION


# ── 6. Sub-Root-System Verification ──────────────────────────────

class TestSubsystemVerification:
    """Test A₂ and A₁ sub-root-systems."""

    def test_a2_is_valid(self):
        result = verify_a2_subsystem(SM_DECOMPOSITION.a2_roots)
        assert result["is_a2"]

    def test_a2_cartan_verified(self):
        result = verify_a2_subsystem(SM_DECOMPOSITION.a2_roots)
        assert result["a2_cartan_verified"]

    def test_a2_reflection_closed(self):
        result = verify_a2_subsystem(SM_DECOMPOSITION.a2_roots)
        assert result["reflection_closed"]

    def test_a2_crystallographic(self):
        result = verify_a2_subsystem(SM_DECOMPOSITION.a2_roots)
        assert result["crystallographic"]

    def test_a1_is_valid(self):
        result = verify_a1_subsystem(SM_DECOMPOSITION.a1_roots)
        assert result["is_a1"]

    def test_a1_negation_pair(self):
        result = verify_a1_subsystem(SM_DECOMPOSITION.a1_roots)
        assert result["negation_pair"]


# ── 7. Triality ──────────────────────────────────────────────────

class TestTriality:
    """Test D₄ triality (Z₃ outer automorphism)."""

    def test_three_permutations(self):
        perms = triality_permutation()
        assert len(perms) == 3

    def test_identity_is_first(self):
        s0, _, _ = triality_permutation()
        assert s0 == D4_SIMPLE_ROOTS

    def test_all_preserve_cartan(self):
        result = verify_triality()
        assert result["all_preserve_cartan"]

    def test_each_preserves_cartan(self):
        result = verify_triality()
        for k in range(3):
            assert result[f"sigma_{k}_preserves_cartan"]

    def test_alpha2_fixed(self):
        """Central node α₂ is fixed by all triality permutations."""
        for perm in triality_permutation():
            assert perm[1] == ALPHA_2

    def test_triality_reps_all_dim_8(self):
        for rep_info in TRIALITY_REPS.values():
            assert rep_info["dim"] == 8


# ── 8. Symmetry Groups ───────────────────────────────────────────

class TestSymmetryGroups:
    """Test Weyl group and automorphism orders."""

    def test_weyl_order(self):
        assert WEYL_D4_ORDER == 192

    def test_weyl_formula(self):
        """W(D₄) = 2^(n-1) × n! with n=4."""
        assert WEYL_D4_ORDER == 2**3 * math.factorial(4)

    def test_outer_aut_order(self):
        assert OUTER_AUT_ORDER == 6

    def test_outer_is_s3(self):
        """Out(D₄) = S₃."""
        assert OUTER_AUT_ORDER == math.factorial(3)

    def test_full_aut_order(self):
        assert AUT_24CELL_ORDER == 1152

    def test_full_aut_product(self):
        assert AUT_24CELL_ORDER == WEYL_D4_ORDER * OUTER_AUT_ORDER


# ── 9. SDGFT Constants ───────────────────────────────────────────

class TestSDGFTConstants:
    """Test connection to SDGFT axiomatic constants."""

    def test_delta_g_from_roots(self):
        assert DELTA_G_FROM_ROOTS == Fraction(1, 24)

    def test_delta_from_geometry(self):
        assert DELTA_FROM_GEOMETRY == Fraction(5, 24)

    def test_delta_g_consistent(self):
        assert DELTA_G_CONSISTENT

    def test_delta_consistent(self):
        assert DELTA_CONSISTENT

    def test_edge_angle(self):
        assert EDGE_ANGLE_DEG == 60

    def test_cos_edge_angle(self):
        assert COS_EDGE_ANGLE == Fraction(1, 2)

    def test_sin2_complement(self):
        assert SIN2_COMPLEMENT == Fraction(1, 4)

    def test_sin2_30_consistent(self):
        assert SIN2_30_CONSISTENT

    def test_edge_angle_from_roots(self):
        """Verify edge angle: cos θ = ⟨α,β⟩/(|α||β|) = 1/2 for neighbours."""
        # Pick two connected roots (inner product = 1)
        r1 = (1, 1, 0, 0)
        r2 = (0, 1, 1, 0)
        assert inner(r1, r2) == 1
        # cos θ = 1 / (√2 × √2) = 1/2
        cos_theta = inner(r1, r2) / (norm_sq(r1) ** 0.5 * norm_sq(r2) ** 0.5)
        assert abs(cos_theta - 0.5) < 1e-15


# ── 10. 24-Cell ↔ D₄ Isomorphism ─────────────────────────────────

class TestIsomorphism:
    """Test 24-cell ↔ D₄ programmatic proof."""

    def test_count_24(self):
        result = verify_24cell_d4_isomorphism()
        assert result["count_24"]

    def test_edge_count(self):
        result = verify_24cell_d4_isomorphism()
        assert result["edges"] == 96

    def test_vertex_degree(self):
        result = verify_24cell_d4_isomorphism()
        assert result["vertex_degree"] == 8

    def test_uniform_degree(self):
        result = verify_24cell_d4_isomorphism()
        assert result["uniform_degree"]

    def test_aut_correct(self):
        result = verify_24cell_d4_isomorphism()
        assert result["aut_correct"]

    def test_is_24cell(self):
        result = verify_24cell_d4_isomorphism()
        assert result["is_24cell"]


# ── 11. SM Content ───────────────────────────────────────────────

class TestSMContent:
    """Test Standard Model field content."""

    def test_sm_content_type(self):
        assert isinstance(SM_CONTENT, SMContent)

    def test_dim_so8(self):
        """dim so(8) = 28 = 24 roots + 4 Cartan."""
        assert SM_CONTENT.dim_so8 == 28

    def test_gauge_total(self):
        assert SM_CONTENT.n_gauge_total == 12

    def test_d4_roots(self):
        assert SM_CONTENT.n_d4_roots == 24

    def test_coset_roots(self):
        assert SM_CONTENT.n_coset_roots == 16

    def test_matter_pairs(self):
        assert SM_CONTENT.n_matter_pairs == 8


# ── 12. Coset Pairs ──────────────────────────────────────────────

class TestCosetPairs:
    """Test coset analysis — matter content."""

    def test_eight_pairs(self):
        pairs = coset_pairs()
        assert len(pairs) == 8

    def test_pairs_are_negations(self):
        """Each pair (α, −α) must be a negation pair."""
        for r, neg_r in coset_pairs():
            assert vec_neg(r) == neg_r

    def test_pairs_cover_coset(self):
        """All 16 coset roots must appear in the 8 pairs."""
        pair_roots = set()
        for r, neg_r in coset_pairs():
            pair_roots.add(r)
            pair_roots.add(neg_r)
        assert pair_roots == set(SM_DECOMPOSITION.coset_roots)

    def test_pairs_disjoint_from_gauge(self):
        """No coset pair root should be in A₂ or A₁."""
        gauge = set(SM_DECOMPOSITION.a2_roots) | set(SM_DECOMPOSITION.a1_roots)
        for r, neg_r in coset_pairs():
            assert r not in gauge
            assert neg_r not in gauge

    def test_matter_times_generations(self):
        """8 matter pairs × 3 generations (triality) = 24 = |D₄|."""
        assert len(coset_pairs()) * 3 == N_D4_ROOTS


# ── 13. Registry ─────────────────────────────────────────────────

class TestRegistry:
    """Test observable registration."""

    def test_register_creates_observables(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        names = [o.name for o in reg.all()]
        assert "qft_n_gauge_bosons" in names
        assert "qft_n_d4_roots" in names
        assert "qft_aut_24cell" in names
        assert "qft_n_matter_pairs" in names

    def test_gauge_boson_observable_value(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        obs = {o.name: o for o in reg.all()}
        assert obs["qft_n_gauge_bosons"].predicted == 12.0
        assert obs["qft_n_gauge_bosons"].observed == 12.0

    def test_d4_root_observable_value(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        obs = {o.name: o for o in reg.all()}
        assert obs["qft_n_d4_roots"].predicted == 24.0

    def test_matter_pairs_observable(self):
        from sdgft.registry import Registry
        reg = Registry()
        register_all(reg)
        obs = {o.name: o for o in reg.all()}
        assert obs["qft_n_matter_pairs"].predicted == 8.0
        assert obs["qft_n_matter_pairs"].observed == 8.0


# ── 14. Summary ──────────────────────────────────────────────────

class TestSummary:
    """Test print_summary smoke test."""

    def test_summary_no_crash(self, capsys):
        print_summary()
        captured = capsys.readouterr()
        assert "QFT BRIDGE" in captured.out

    def test_summary_contains_key_results(self, capsys):
        print_summary()
        captured = capsys.readouterr()
        assert "SU(3)" in captured.out
        assert "SU(2)" in captured.out
        assert "U(1)" in captured.out
        assert "12" in captured.out
        assert "Triality" in captured.out or "triality" in captured.out

    def test_summary_contains_isomorphism(self, capsys):
        print_summary()
        captured = capsys.readouterr()
        assert "ISOMORPHIC" in captured.out

    def test_summary_contains_sdgft_constants(self, capsys):
        print_summary()
        captured = capsys.readouterr()
        assert "1/24" in captured.out
        assert "5/24" in captured.out


# ── 15. Linear Algebra Utilities ─────────────────────────────────

class TestLinearAlgebra:
    """Test utility functions."""

    def test_inner_product(self):
        assert inner((1, 0, 0, 0), (1, 0, 0, 0)) == 1
        assert inner((1, 1, 0, 0), (1, -1, 0, 0)) == 0
        assert inner((1, 1, 0, 0), (0, 1, 1, 0)) == 1

    def test_norm_sq(self):
        assert norm_sq((1, 1, 0, 0)) == 2
        assert norm_sq((1, 0, 0, 0)) == 1

    def test_vec_add(self):
        assert vec_add((1, -1, 0, 0), (0, 1, -1, 0)) == (1, 0, -1, 0)

    def test_vec_neg(self):
        assert vec_neg((1, -1, 0, 0)) == (-1, 1, 0, 0)

    def test_cartan_a1(self):
        """Cartan matrix of A₁ is [[2]]."""
        cm = cartan_matrix(((1, -1, 0, 0),))
        assert cm == ((2,),)

    def test_cartan_a2(self):
        """Cartan matrix of A₂ is [[2,-1],[-1,2]]."""
        cm = cartan_matrix((ALPHA_1, ALPHA_2))
        assert cm == ((2, -1), (-1, 2))
