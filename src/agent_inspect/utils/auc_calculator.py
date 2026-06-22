"""
Simple AUC (Area Under Curve) calculator using the trapezoidal rule.
Replaces sklearn.metrics.auc dependency.
"""


def auc(x: list[float], y: list[float]) -> float:
    """
    Compute Area Under the Curve (AUC) using the trapezoidal rule.

    This implementation assumes x coordinates are monotonic (either all increasing
    or all decreasing) and both lists have the same length with at least 2 points.

    Args:
        x: X coordinates (must be monotonic)
        y: Y coordinates

    Returns:
        Area under the curve as a float

    Raises:
        ValueError: If lists have different lengths, less than 2 points,
                   or x is not monotonic
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")

    if len(x) < 2:
        raise ValueError(f"At least 2 points are needed to compute area under curve, got {len(x)}")

    # Check if x is monotonic and determine direction
    is_increasing = None

    for i in range(1, len(x)):
        diff = x[i] - x[i - 1]
        if diff > 0:
            if is_increasing is False:
                raise ValueError(f"x is neither increasing nor decreasing: {x}")
            is_increasing = True
        elif diff < 0:
            if is_increasing is True:
                raise ValueError(f"x is neither increasing nor decreasing: {x}")
            is_increasing = False

    direction = 1 if is_increasing else -1

    # Calculate area using trapezoidal rule: sum of (width * average_height)
    area = 0.0
    for i in range(1, len(x)):
        width = x[i] - x[i - 1]
        avg_height = (y[i] + y[i - 1]) / 2
        area += width * avg_height

    return direction * area
