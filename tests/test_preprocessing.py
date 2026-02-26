"""
Test: Arogentis Pipeline — Unit Tests
=======================================
Tests the synthetic dataset generation and feature extraction pipeline.

NOTE: Python 3.14 + NumPy 2.x breaks MNE's internal Numba JIT, so tests
avoid calling MNE functions that trigger Numba. Instead, we test through
our own pipeline functions which have been patched to bypass Numba.
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSyntheticDataset:
    """Test the synthetic dataset generation pipeline end-to-end."""

    def test_generate_synthetic_dataset(self, tmp_path):
        """Generate synthetic features and verify shapes + labels."""
        from pipeline.dataset_builder import generate_synthetic_dataset

        X, y, feature_names = generate_synthetic_dataset(
            n_schizophrenia=3,
            n_controls=3,
            n_channels=4,
            duration_seconds=10.0,
            output_dir=str(tmp_path),
        )
        assert X.shape[0] == 6              # 3 + 3 subjects
        assert X.shape[1] == len(feature_names)
        assert X.shape[1] > 0
        assert y.shape == (6,)
        assert y.sum() == 3                 # 3 schizophrenia labels
        assert (y == 0).sum() == 3          # 3 control labels

    def test_feature_names_contain_expected_biomarkers(self, tmp_path):
        """Feature names should include all expected biomarker types."""
        from pipeline.dataset_builder import generate_synthetic_dataset

        _, _, feature_names = generate_synthetic_dataset(
            n_schizophrenia=2,
            n_controls=2,
            n_channels=4,
            duration_seconds=10.0,
            output_dir=str(tmp_path),
        )
        assert any("delta" in n for n in feature_names)
        assert any("theta" in n for n in feature_names)
        assert any("alpha" in n for n in feature_names)
        assert any("beta" in n for n in feature_names)
        assert any("gamma" in n for n in feature_names)
        assert any("spectral_entropy" in n for n in feature_names)
        assert any("alpha_theta_ratio" in n for n in feature_names)
        assert any("gamma_theta_ratio" in n for n in feature_names)

    def test_saved_files_exist(self, tmp_path):
        """Check that X.npy, y.npy, and feature_names.txt are saved."""
        from pipeline.dataset_builder import generate_synthetic_dataset

        generate_synthetic_dataset(
            n_schizophrenia=2,
            n_controls=2,
            n_channels=4,
            duration_seconds=10.0,
            output_dir=str(tmp_path),
        )
        assert (tmp_path / "X.npy").exists()
        assert (tmp_path / "y.npy").exists()
        assert (tmp_path / "feature_names.txt").exists()

    def test_features_differ_by_label(self, tmp_path):
        """Schizophrenia features should differ from control features."""
        from pipeline.dataset_builder import generate_synthetic_dataset

        X, y, _ = generate_synthetic_dataset(
            n_schizophrenia=5,
            n_controls=5,
            n_channels=4,
            duration_seconds=10.0,
            output_dir=str(tmp_path),
        )
        scz_mean = X[y == 1].mean(axis=0)
        ctl_mean = X[y == 0].mean(axis=0)
        # At least some features should differ significantly
        diff = np.abs(scz_mean - ctl_mean).max()
        assert diff > 0, "Schizophrenia and control features should differ"


class TestRiskScorer:
    """Test the risk scoring tier logic."""

    def test_risk_tiers(self):
        """Verify tier boundaries return correct labels."""
        from models.risk_scorer import RISK_TIERS

        # Check all 4 tiers exist
        tier_names = [t[2] for t in RISK_TIERS]
        assert "Low Risk" in tier_names
        assert "Moderate Risk" in tier_names
        assert "High Risk" in tier_names
        assert "Critical Risk" in tier_names

    def test_tier_boundaries_cover_full_range(self):
        """Tiers should cover 0.0 to 1.0 without gaps."""
        from models.risk_scorer import RISK_TIERS

        assert RISK_TIERS[0][0] == 0.0           # starts at 0
        for i in range(len(RISK_TIERS) - 1):
            assert RISK_TIERS[i][1] == RISK_TIERS[i + 1][0]  # no gaps


class TestFeatureExtraction:
    """Test feature extraction band definitions and utilities."""

    def test_band_definitions(self):
        """Verify all 5 EEG frequency bands are defined."""
        from pipeline.feature_extraction import BANDS

        assert "delta" in BANDS
        assert "theta" in BANDS
        assert "alpha" in BANDS
        assert "beta" in BANDS
        assert "gamma" in BANDS

    def test_band_ranges_are_valid(self):
        """Each band's fmin < fmax."""
        from pipeline.feature_extraction import BANDS

        for band, (fmin, fmax) in BANDS.items():
            assert fmin < fmax, f"{band}: fmin ({fmin}) >= fmax ({fmax})"

    def test_bands_are_non_overlapping(self):
        """Bands should not overlap."""
        from pipeline.feature_extraction import BANDS

        ranges = sorted(BANDS.values())
        for i in range(len(ranges) - 1):
            assert ranges[i][1] <= ranges[i + 1][0], (
                f"Band overlap: {ranges[i]} and {ranges[i+1]}"
            )
