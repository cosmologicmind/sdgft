"""Tests for the cross-module validation suite."""

import pytest

from sdgft.validation import (
    ValidationResult,
    check_axioms,
    check_dimension,
    check_gravity,
    check_cosmology,
    check_inflation,
    check_particle,
    check_all,
    validate,
)


class TestValidationResult:
    def test_empty_is_ok(self):
        r = ValidationResult()
        assert r.ok
        assert len(r.passed) == 0
        assert len(r.failed) == 0

    def test_pass_recorded(self):
        r = ValidationResult()
        r.check(True, "test_pass")
        assert r.ok
        assert "test_pass" in r.passed

    def test_fail_recorded(self):
        r = ValidationResult()
        r.check(False, "test_fail", "something broke")
        assert not r.ok
        assert ("test_fail", "something broke") in r.failed

    def test_mixed(self):
        r = ValidationResult()
        r.check(True, "good")
        r.check(False, "bad", "reason")
        assert not r.ok
        assert len(r.passed) == 1
        assert len(r.failed) == 1

    def test_summary_contains_counts(self):
        r = ValidationResult()
        r.check(True, "a")
        r.check(False, "b", "detail")
        s = r.summary()
        assert "1 passed" in s
        assert "1 failed" in s
        assert "PASS" in s
        assert "FAIL" in s


class TestIndividualChecks:
    """Each check function must pass on a fresh ValidationResult."""

    def test_check_axioms(self):
        r = ValidationResult()
        check_axioms(r)
        assert r.ok, r.summary()
        assert len(r.passed) >= 5

    def test_check_dimension(self):
        r = ValidationResult()
        check_dimension(r)
        assert r.ok, r.summary()
        assert len(r.passed) >= 8

    def test_check_gravity(self):
        r = ValidationResult()
        check_gravity(r)
        assert r.ok, r.summary()
        assert len(r.passed) >= 5

    def test_check_cosmology(self):
        r = ValidationResult()
        check_cosmology(r)
        assert r.ok, r.summary()
        assert len(r.passed) >= 5

    def test_check_inflation(self):
        r = ValidationResult()
        check_inflation(r)
        assert r.ok, r.summary()
        assert len(r.passed) >= 5

    def test_check_particle(self):
        r = ValidationResult()
        check_particle(r)
        assert r.ok, r.summary()
        assert len(r.passed) >= 6


class TestCheckAll:
    def test_all_pass(self):
        result = check_all()
        assert result.ok, result.summary()

    def test_total_check_count(self):
        result = check_all()
        total = len(result.passed) + len(result.failed)
        assert total >= 30, f"Only {total} checks ran, expected >= 30"

    def test_summary_shows_all_passed(self):
        result = check_all()
        assert "All checks passed" in result.summary()


class TestValidate:
    def test_validate_does_not_raise(self):
        validate()

    def test_validate_loud(self, capsys):
        validate(loud=True)
        captured = capsys.readouterr()
        assert "SDGFT Validation" in captured.out
        assert "All checks passed" in captured.out

    def test_validate_fails_on_bad_result(self, monkeypatch):
        """Verify that validate() raises on failure."""
        def fake_check_all():
            r = ValidationResult()
            r.check(False, "fake_broken", "injected failure")
            return r

        import sdgft.validation as val_mod
        monkeypatch.setattr(val_mod, "check_all", fake_check_all)
        with pytest.raises(AssertionError, match="SDGFT validation failed"):
            validate()
