"""
Unit tests for AUC calculator.
Validates that our implementation matches scikit-learn's implementation.
"""

import pytest
import sklearn.metrics
from agent_inspect.utils.auc_calculator import auc


def test_auc_matches_sklearn_simple_trapezoid():
    """Test basic trapezoidal area calculation matches sklearn."""
    # Rectangle: x=[0,1], y=[1,1] -> area = 1
    x, y = [0, 1], [1, 1]
    assert auc(x, y) == sklearn.metrics.auc(x, y) == 1.0

    # Triangle: x=[0,1], y=[0,1] -> area = 0.5
    x, y = [0, 1], [0, 1]
    assert auc(x, y) == sklearn.metrics.auc(x, y) == 0.5


def test_auc_matches_sklearn_multiple_points():
    """Test with multiple points matches sklearn."""
    # Three points forming two trapezoids
    x = [0, 0.5, 1]
    y = [0, 0.5, 1]
    custom_result = auc(x, y)
    sklearn_result = sklearn.metrics.auc(x, y)
    assert custom_result == sklearn_result == 0.5


def test_auc_matches_sklearn_decreasing_x():
    """Test with decreasing x values matches sklearn."""
    # Should work with monotonic decreasing x
    x = [1, 0.5, 0]
    y = [1, 0.5, 0]
    custom_result = auc(x, y)
    sklearn_result = sklearn.metrics.auc(x, y)
    assert custom_result == sklearn_result == 0.5


def test_auc_matches_sklearn_four_points():
    """Test with four points matches sklearn (similar to sklearn docs example)."""
    x = [0, 1 / 3, 2 / 3, 1]
    y = [0, 0.5, 0.5, 1]
    custom_result = auc(x, y)
    sklearn_result = sklearn.metrics.auc(x, y)
    assert abs(custom_result - sklearn_result) < 1e-10


def test_auc_matches_sklearn_uniform_progress():
    """Test case similar to the actual use in AUC metric matches sklearn."""
    # Simulating 5 turns with linear progress
    n = 5
    x = [i / (n - 1) for i in range(n)]  # [0, 0.25, 0.5, 0.75, 1]
    y = [0, 0.25, 0.5, 0.75, 1]  # Linear progress
    custom_result = auc(x, y)
    sklearn_result = sklearn.metrics.auc(x, y)
    assert abs(custom_result - sklearn_result) < 1e-10


def test_auc_matches_sklearn_realistic_progress():
    """Test with realistic progress score scenario matches sklearn."""
    # 4 turns with improving progress
    n = 4
    x = [i / (n - 1) for i in range(n)]  # [0, 0.333..., 0.666..., 1]
    y = [0, 0.3, 0.7, 1.0]  # Non-linear progress

    custom_result = auc(x, y)
    sklearn_result = sklearn.metrics.auc(x, y)
    assert abs(custom_result - sklearn_result) < 1e-10


def test_auc_matches_sklearn_many_points():
    """Test with many points to ensure numerical stability matches sklearn."""
    n = 20
    x = [i / (n - 1) for i in range(n)]
    y = [i**2 / (n - 1) ** 2 for i in range(n)]  # Quadratic progression

    custom_result = auc(x, y)
    sklearn_result = sklearn.metrics.auc(x, y)
    assert abs(custom_result - sklearn_result) < 1e-10


def test_auc_matches_sklearn_irregular_spacing():
    """Test with irregularly spaced points matches sklearn."""
    x = [0, 0.1, 0.3, 0.8, 1.0]
    y = [0, 0.2, 0.4, 0.9, 1.0]

    custom_result = auc(x, y)
    sklearn_result = sklearn.metrics.auc(x, y)
    assert abs(custom_result - sklearn_result) < 1e-10


def test_auc_error_different_lengths():
    """Test error when x and y have different lengths."""
    with pytest.raises(ValueError, match="same length"):
        auc([0, 1], [0, 1, 2])


def test_auc_error_too_few_points():
    """Test error with less than 2 points."""
    with pytest.raises(ValueError, match="At least 2 points"):
        auc([0], [0])


def test_auc_error_non_monotonic():
    """Test error when x is not monotonic."""
    with pytest.raises(ValueError, match="neither increasing nor decreasing"):
        auc([0, 1, 0.5], [0, 1, 0.5])


def test_auc_matches_sklearn_edge_case_two_points():
    """Test edge case with exactly 2 points matches sklearn."""
    x = [0.0, 1.0]
    y = [0.5, 0.8]

    custom_result = auc(x, y)
    sklearn_result = sklearn.metrics.auc(x, y)
    assert abs(custom_result - sklearn_result) < 1e-10


def test_auc_matches_sklearn_constant_y():
    """Test with constant y values (rectangle) matches sklearn."""
    x = [0, 0.25, 0.5, 0.75, 1.0]
    y = [0.7, 0.7, 0.7, 0.7, 0.7]

    custom_result = auc(x, y)
    sklearn_result = sklearn.metrics.auc(x, y)
    assert abs(custom_result - sklearn_result) < 1e-10
    assert abs(custom_result - 0.7) < 1e-10  # Area should be height * width = 0.7 * 1
