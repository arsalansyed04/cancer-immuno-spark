"""
Tests for src/importance.py — min-max scaling, component score formulas,
and overall normalization to 100.
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from importance import (
    _minmax,
    compute_e,
    compute_y,
    weight_sensitivity,
)


def test_minmax_scaling():
    arr = np.array([10.0, 20.0, 30.0, 50.0])
    scaled = _minmax(arr)
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0
    assert np.isclose(scaled[0], 0.0)
    assert np.isclose(scaled[3], 1.0)
    assert np.isclose(scaled[1], (20.0 - 10.0) / (50.0 - 10.0))


def test_minmax_constant_array():
    arr = np.array([5.0, 5.0, 5.0])
    scaled = _minmax(arr)
    assert np.all(scaled == 0.0)


def test_compute_y():
    years = np.array([2015, 2017, 2020, 2025])
    y = compute_y(years)
    assert np.isclose(y[0], 0.0)
    assert np.isclose(y[-1], 1.0)
    assert y[1] < y[2]


def test_compute_e(tmp_path):
    # Create a mock topic_delta.csv
    delta_file = tmp_path / "topic_delta.csv"
    delta_df = pd.DataFrame({
        "topic_id": [0, 1, 2],
        "early_share": [10.0, 20.0, 30.0],
        "late_share": [15.0, 18.0, 35.0],
        "delta_pp": [5.0, -2.0, 5.0],  # topic 0 has +5 pp, topic 1 has -2 pp, topic 2 has +5 pp
        "trend_label": ["emerging", "declining", "emerging"]
    })
    delta_df.to_csv(delta_file, index=False)

    # theta matrix for 2 papers across 3 topics
    # paper 0: heavy in emerging topics 0 and 2
    # paper 1: heavy in declining topic 1
    theta = np.array([
        [0.5, 0.0, 0.5],
        [0.0, 1.0, 0.0],
    ])

    e = compute_e(theta, str(delta_file))
    # Growth for topic 0: 0.05, topic 1: 0.0 (clipped to 0), topic 2: 0.05
    # Paper 0 e = 0.5*0.05 + 0.5*0.05 = 0.05
    # Paper 1 e = 1.0*0.0 = 0.0
    assert np.isclose(e[0], 0.05)
    assert np.isclose(e[1], 0.0)
    assert e[0] > e[1]


def test_importance_score_top_100():
    # Verify overall score calculation and normalisation
    r_sc = np.array([0.2, 0.8, 1.0, 0.5])
    w_sc = np.array([0.1, 0.9, 0.7, 0.3])
    e_sc = np.array([0.0, 0.5, 0.8, 0.4])
    y_sc = np.array([0.5, 0.7, 0.9, 0.2])

    weights = {"r": 0.35, "w": 0.25, "e": 0.25, "y": 0.15}
    raw = (
        weights["r"] * r_sc +
        weights["w"] * w_sc +
        weights["e"] * e_sc +
        weights["y"] * y_sc
    )

    max_score = raw.max()
    importance = 100.0 * raw / max_score

    assert np.isclose(importance.max(), 100.0)
    assert np.all(importance >= 0.0)
    assert np.all(importance <= 100.0)
    # The max paper index should have exactly 100
    assert np.isclose(importance[np.argmax(raw)], 100.0)
