from unittest.mock import MagicMock

import pytest

from agent_inspect.exception import EvaluationError, InvalidInputValueError
from agent_inspect.metrics.scorer import (
    ProgressScore,
    SuccessScore,
    AUC,
    ToolCorrectnessMetric,
)
from agent_inspect.models.metrics import NumericalScore


def test_get_progress_score_handles_mixed_completion_states():
    validation_results = [
        MagicMock(is_completed=True),
        MagicMock(is_completed=False),
        MagicMock(is_completed=True),
    ]
    result = ProgressScore.get_progress_score_from_validation_results(validation_results)
    assert result.score == 0.6667


def test_get_progress_score_handles_no_validation_results():
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050004, Error Message: No validation result present to aggregate for progress score.",
    ):
        ProgressScore.get_progress_score_from_validation_results([])


def test_get_progress_score_handles_all_incomplete_states():
    validation_results = [MagicMock(is_completed=False), MagicMock(is_completed=False)]
    result = ProgressScore.get_progress_score_from_validation_results(validation_results)
    assert result.score == 0.0


def test_get_success_score_returns_false_for_mixed_completion_states():
    validation_results = [MagicMock(is_completed=True), MagicMock(is_completed=False)]
    result = SuccessScore.get_success_score_from_validation_results(validation_results)
    assert result.score == 0


def test_get_success_score_returns_true_for_all_completed_states():
    validation_results = [MagicMock(is_completed=True), MagicMock(is_completed=True)]
    result = SuccessScore.get_success_score_from_validation_results(validation_results)
    assert result.score == 1


def test_get_success_score_returns_true_for_no_validation_results():
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050004, Error Message: No validation result present to aggregate for success score.",
    ):
        SuccessScore.get_success_score_from_validation_results([])


def test_get_get_success_score_from_progress_score_success_1():
    progress_score = NumericalScore(
        score=1.00, explanations=["This is explanation 1", "This is explanation 2"]
    )
    result = SuccessScore.get_success_score_from_progress_score(progress_score)
    assert result.score == 1


def test_get_get_success_score_from_progress_score_success_0():
    progress_score = NumericalScore(
        score=0.75, explanations=["This is explanation 1", "This is explanation 2"]
    )
    result = SuccessScore.get_success_score_from_progress_score(progress_score)
    assert result.score == 0


def test_get_auc_score_no_validation_results():
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050005, Error Message: No progress score provided for AUC calculation.",
    ):
        AUC.get_auc_score_from_progress_scores([])


def test_get_auc_score_calculation_only_one_score():
    progress_score = NumericalScore(score=0.7)
    with pytest.raises(
        EvaluationError,
        match="Internal Code: 050006, Error Message: Error calculating AUC: At least 2 points are needed to compute area under curve, got 1",
    ):
        AUC.get_auc_score_from_progress_scores([progress_score])


def test_get_auc_score_calculation_multiple_scores():
    progress_scores = [
        NumericalScore(score=0.0),
        NumericalScore(score=0.5),
        NumericalScore(score=1.0),
    ]

    result = AUC.get_auc_score_from_progress_scores(progress_scores)
    assert result.score == 0.5


def test_get_tool_correctness_score_handles_mixed_correctness():
    validation_results = [
        MagicMock(is_completed=True),
        MagicMock(is_completed=False),
        MagicMock(is_completed=True),
    ]
    result = ToolCorrectnessMetric.get_tool_correctness_score_from_validation_results(
        validation_results
    )
    assert result.score == 0.6667


def test_get_tool_correctness_score_handles_incorrectness():
    validation_results = [
        MagicMock(is_completed=False),
        MagicMock(is_completed=False),
        MagicMock(is_completed=False),
    ]
    result = ToolCorrectnessMetric.get_tool_correctness_score_from_validation_results(
        validation_results
    )
    assert result.score == 0.0


def test_get_tool_correctness_score_handles_correctness():
    validation_results = [
        MagicMock(is_completed=True),
        MagicMock(is_completed=True),
        MagicMock(is_completed=True),
    ]
    result = ToolCorrectnessMetric.get_tool_correctness_score_from_validation_results(
        validation_results
    )
    assert result.score == 1.0


def test_get_tool_correctness_score_handles_no_validation_results():
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050004, Error Message: No validation results present to aggregate for tool correctness.",
    ):
        ToolCorrectnessMetric.get_tool_correctness_score_from_validation_results([])
