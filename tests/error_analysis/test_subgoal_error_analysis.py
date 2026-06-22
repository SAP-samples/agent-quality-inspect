"""
Unit tests for SubgoalErrorAnalysis (abstract base class).

Tests cover shared methods defined in subgoal_error_analysis.py:
- _get_judge_trial_explanations_from_subgoal_validation
- _has_failed_consistently
- _summarize_error
- _analyze
- _split_analysed_subgoal_validations_by_completeness
- _build_clustered_result
- analyze_batch

Uses a concrete stub subclass to isolate the parent class logic.
"""

import json
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.exception.exception import InvalidInputValueError
from agent_inspect.metrics.constants import STATUS_200
from agent_inspect.exception import ToolError
from agent_inspect.models.metrics.agent_data_sample import SubGoal
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult
from agent_inspect.models.llm_response import LLMResponse
from agent_inspect.tools.error_analysis.subgoal_error_analysis import (
    SubgoalErrorAnalysis,
)
from agent_inspect.models.tools.analysis_models import (
    AnalyzedSubgoalValidation,
    SubgoalErrorAnalysisDataSample,
    SubgoalErrorAnalysisResult,
)


class ConcreteSubgoalErrorAnalysis(SubgoalErrorAnalysis):
    """Concrete stub for testing the abstract SubgoalErrorAnalysis."""

    async def _summarize_errors_into_base_error(self, subgoal_validation):
        if subgoal_validation.is_completed:
            return None
        return "Stub base error"

    async def _cluster_errors(self, _analyzed_subgoal_validations):
        return {"clusters": []}


def mock_asyncio_run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.make_request_with_payload = AsyncMock()
    return client


@pytest.fixture
def analyzer(mock_llm_client):
    return ConcreteSubgoalErrorAnalysis(llm_client=mock_llm_client, config={"max_workers": 20})


@pytest.fixture
def mock_subgoal_1():
    return SubGoal(details="Ensure calendar tool execution")


@pytest.fixture
def mock_subgoal_2():
    return SubGoal(details="Draft an email to the client")


@pytest.fixture
def mock_subgoal_validation_result_all_incomplete(mock_subgoal_1):
    return SubGoalValidationResult(
        is_completed=False,
        explanations=[
            "Overall explanation of failure.",
            "Judge Trial 1 Explanation. The agent failed to use the calendar tool correctly.\n\nGrade: I",
            "Judge Trial 2 Explanation. The agent did not schedule the meeting as required.\n\nGrade: I",
        ],
        sub_goal=mock_subgoal_1,
    )


@pytest.fixture
def mock_subgoal_validation_result_some_complete(mock_subgoal_2):
    return SubGoalValidationResult(
        is_completed=False,
        explanations=[
            "Overall explanation of failure.",
            "Judge Trial 1 Explanation. The agent drafted the email but missed key details.\n\nGrade: C",
            "Judge Trial 2 Explanation. The agent failed to draft the email entirely.\n\nGrade: I",
        ],
        sub_goal=mock_subgoal_2,
    )


@pytest.fixture
def mock_subgoal_validation_result_is_complete(mock_subgoal_2):
    return SubGoalValidationResult(
        is_completed=True,
        explanations=[
            "Overall explanation of success.",
            "Judge Trial 1 Explanation. The agent drafted the email correctly.\n\nGrade: C",
            "Judge Trial 2 Explanation. The agent drafted the email correctly.\n\nGrade: C",
        ],
        sub_goal=mock_subgoal_2,
    )


@pytest.fixture
def mock_analyzed_subgoal_validation_1(mock_subgoal_validation_result_all_incomplete):
    return AnalyzedSubgoalValidation(
        subgoal_validation=mock_subgoal_validation_result_all_incomplete,
        data_sample_id=1,
        base_error="Tool misuse",
    )


@pytest.fixture
def mock_analyzed_subgoal_validation_2(mock_subgoal_validation_result_some_complete):
    return AnalyzedSubgoalValidation(
        subgoal_validation=mock_subgoal_validation_result_some_complete,
        data_sample_id=2,
        base_error="Incomplete task execution",
    )


@pytest.fixture
def mock_analyzed_subgoal_validation_3(mock_subgoal_validation_result_some_complete):
    return AnalyzedSubgoalValidation(
        subgoal_validation=mock_subgoal_validation_result_some_complete,
        data_sample_id=3,
        base_error="Task not fully completed",
    )


@pytest.fixture
def mock_analyzed_subgoal_validation_4(mock_subgoal_validation_result_is_complete):
    return AnalyzedSubgoalValidation(
        subgoal_validation=mock_subgoal_validation_result_is_complete,
        data_sample_id=4,
        base_error=None,
    )


@pytest.fixture
def sample_llm_clusterings():
    return {
        "clusters": [
            {
                "cluster_label": "Calendar tool issues",
                "error_types": ["Tool misuse"],
                "error_ids": ["0"],
            },
            {
                "cluster_label": "Communication gaps",
                "error_types": ["Incomplete task execution"],
                "error_ids": ["1", "2"],
            },
        ]
    }


# ==================== Tests for _get_judge_trial_explanations_from_subgoal_validation ====================


def test_get_judge_trial_explanations_returns_trials_excluding_overall(
    analyzer, mock_subgoal_validation_result_all_incomplete
):
    judge_trials = analyzer._get_judge_trial_explanations_from_subgoal_validation(
        mock_subgoal_validation_result_all_incomplete
    )

    assert judge_trials == [
        "Judge Trial 1 Explanation. The agent failed to use the calendar tool correctly.\n\nGrade: I",
        "Judge Trial 2 Explanation. The agent did not schedule the meeting as required.\n\nGrade: I",
    ]


@pytest.mark.parametrize(
    "explanations",
    [["Only overall explanation present"], []],
)
def test_get_judge_trial_explanations_raises_for_invalid_format(
    analyzer, mock_subgoal_1, explanations
):
    subgoal_validation = SubGoalValidationResult(
        is_completed=False, explanations=explanations, sub_goal=mock_subgoal_1
    )

    with pytest.raises(
        InvalidInputValueError,
        match="Invalid SubGoalValidationResult.explanation format",
    ):
        analyzer._get_judge_trial_explanations_from_subgoal_validation(subgoal_validation)


# ==================== Tests for _has_failed_consistently ====================


def test_has_failed_consistently_returns_true_with_all_incomplete(
    analyzer, mock_subgoal_validation_result_all_incomplete
):
    assert analyzer._has_failed_consistently(mock_subgoal_validation_result_all_incomplete) is True


def test_has_failed_consistently_returns_false_when_any_complete(
    analyzer, mock_subgoal_validation_result_some_complete
):
    assert analyzer._has_failed_consistently(mock_subgoal_validation_result_some_complete) is False


def test_has_failed_consistently_raises_error_for_invalid_responses(analyzer, mock_subgoal_1):
    """Test that _has_failed_consistently raises an error when there are invalid judge responses."""
    from agent_inspect.exception import InvalidInputValueError

    # Create a SubGoalValidationResult with explanations that will produce invalid responses
    # The tally_votes function will mark responses as invalid if they can't be parsed
    subgoal_validation = SubGoalValidationResult(
        is_completed=False,
        sub_goal=mock_subgoal_1,
        explanations=[
            "Overall explanation",
            "GRADE: I",
            "Invalid response without GRADE marker",  # This will be invalid
            "GRADE: I",
        ],
    )

    # Should raise InvalidInputValueError when encountering invalid judge responses
    with pytest.raises(
        InvalidInputValueError,
        match="Subgoal error analysis encountered 1 invalid judge response",
    ):
        analyzer._has_failed_consistently(subgoal_validation)


# ==================== Tests for _summarize_error ====================


@pytest.mark.asyncio
async def test_summarize_error_returns_error_type(analyzer, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200,
        completion=json.dumps({"error_type": "Tool misuse", "explanation": "desc"}),
    )

    result = await analyzer._summarize_error("Judge explanation", "Use calendar tool")

    assert result == "Tool misuse"
    mock_llm_client.make_request_with_payload.assert_awaited_once()


@pytest.mark.asyncio
async def test_summarize_error_raises_when_llm_fails(analyzer, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=500, completion=None, error_message="Boom"
    )

    with pytest.raises(ToolError, match="LLM request failed with status 500 and error: Boom"):
        await analyzer._summarize_error("Judge explanation", "Use calendar tool")


@pytest.mark.asyncio
async def test_summarize_error_raises_when_error_type_key_missing(analyzer, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200,
        completion=json.dumps({"wrong_key": "some value", "explanation": "some explanation"}),
    )

    with pytest.raises(
        Exception,
        match="LLM error summarization request failed as no error_type found in response",
    ):
        await analyzer._summarize_error("Judge explanation", "Use calendar tool")


# ==================== Tests for _analyze ====================


@pytest.mark.asyncio
async def test_analyze_returns_analyzed_subgoal_validations(
    analyzer,
    monkeypatch,
    mock_subgoal_validation_result_all_incomplete,
    mock_subgoal_validation_result_some_complete,
):
    summarize_mock = AsyncMock(side_effect=["Tool misuse", "Incomplete task execution"])
    monkeypatch.setattr(analyzer, "_summarize_errors_into_base_error", summarize_mock)

    data_sample = SubgoalErrorAnalysisDataSample(
        data_sample_id=1,
        subgoal_validations=[
            mock_subgoal_validation_result_all_incomplete,
            mock_subgoal_validation_result_some_complete,
        ],
        agent_run_id=1,
    )

    result = await analyzer._analyze(data_sample)

    assert len(result) == 2
    assert all(isinstance(asv, AnalyzedSubgoalValidation) for asv in result)
    assert result[0].data_sample_id == 1
    assert result[0].base_error == "Tool misuse"
    assert result[0].subgoal_validation == mock_subgoal_validation_result_all_incomplete
    assert result[0].agent_run_id == 1
    assert result[1].data_sample_id == 1
    assert result[1].base_error == "Incomplete task execution"
    assert result[1].subgoal_validation == mock_subgoal_validation_result_some_complete
    assert result[1].agent_run_id == 1
    assert summarize_mock.await_count == 2


@pytest.mark.asyncio
async def test_analyze_handles_empty_subgoal_validations(analyzer):
    """Test that _analyze handles data samples with no subgoal validations."""
    data_sample = SubgoalErrorAnalysisDataSample(data_sample_id=1, subgoal_validations=[])

    result = await analyzer._analyze(data_sample)

    assert result == []


# ==================== Tests for _split_analysed_subgoal_validations_by_completeness ====================


def test_split_by_completeness_separates_correctly(
    analyzer,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_4,
):
    complete_asv = mock_analyzed_subgoal_validation_4
    incomplete_asv_1 = mock_analyzed_subgoal_validation_1
    incomplete_asv_2 = mock_analyzed_subgoal_validation_2
    all_asvs = [[complete_asv, incomplete_asv_1], [incomplete_asv_2]]

    completed, incomplete = analyzer._split_analysed_subgoal_validations_by_completeness(all_asvs)

    assert len(completed) == 1
    assert len(incomplete) == 2
    assert completed[0] == complete_asv
    assert incomplete[0] == incomplete_asv_1
    assert incomplete[1] == incomplete_asv_2


# ==================== Tests for _build_clustered_result ====================


def test_build_clustered_result_maps_indices(
    analyzer,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_3,
    sample_llm_clusterings,
):
    result = analyzer._build_clustered_result(
        sample_llm_clusterings,
        [
            mock_analyzed_subgoal_validation_1,
            mock_analyzed_subgoal_validation_2,
            mock_analyzed_subgoal_validation_3,
        ],
    )

    assert isinstance(result, dict)
    assert result["Calendar tool issues"] == [mock_analyzed_subgoal_validation_1]
    assert result["Communication gaps"] == [
        mock_analyzed_subgoal_validation_2,
        mock_analyzed_subgoal_validation_3,
    ]


def test_build_clustered_result_handles_missing_error_ids(
    analyzer,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_3,
    caplog,
):
    """Test that missing error_ids are added to a None cluster and a warning is logged."""
    # LLM clustering only includes error_ids 0 and 2, missing error_id 1
    incomplete_llm_clusterings = {
        "clusters": [
            {
                "cluster_label": "Calendar tool issues",
                "error_types": ["Tool misuse"],
                "error_ids": ["0"],
            },
            {
                "cluster_label": "Communication gaps",
                "error_types": ["Task not fully completed"],
                "error_ids": ["2"],
            },
        ]
    }

    analyzed_validations = [
        mock_analyzed_subgoal_validation_1,  # error_id 0
        mock_analyzed_subgoal_validation_2,  # error_id 1 (will be missing)
        mock_analyzed_subgoal_validation_3,  # error_id 2
    ]

    import logging

    with caplog.at_level(logging.WARNING):
        result = analyzer._build_clustered_result(incomplete_llm_clusterings, analyzed_validations)

    # Verify the warning was logged
    assert "LLM clustering missed 1 error(s) out of 3" in caplog.text
    assert "Missing error_ids: [1]" in caplog.text

    # Verify clustered results
    assert isinstance(result, dict)
    assert len(result) == 3  # Two named clusters + None cluster
    assert result["Calendar tool issues"] == [mock_analyzed_subgoal_validation_1]
    assert result["Communication gaps"] == [mock_analyzed_subgoal_validation_3]

    # Verify missing error_id was added to None cluster
    assert None in result
    assert result[None] == [mock_analyzed_subgoal_validation_2]


def test_build_clustered_result_handles_multiple_missing_error_ids(
    analyzer,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_3,
    caplog,
):
    """Test that multiple missing error_ids are added to None cluster in sorted order."""
    # LLM clustering only includes error_id 1, missing error_ids 0 and 2
    incomplete_llm_clusterings = {
        "clusters": [
            {
                "cluster_label": "Communication gaps",
                "error_types": ["Incomplete task execution"],
                "error_ids": ["1"],
            }
        ]
    }

    analyzed_validations = [
        mock_analyzed_subgoal_validation_1,  # error_id 0 (will be missing)
        mock_analyzed_subgoal_validation_2,  # error_id 1
        mock_analyzed_subgoal_validation_3,  # error_id 2 (will be missing)
    ]

    import logging

    with caplog.at_level(logging.WARNING):
        result = analyzer._build_clustered_result(incomplete_llm_clusterings, analyzed_validations)

    # Verify the warning was logged
    assert "LLM clustering missed 2 error(s) out of 3" in caplog.text
    assert "Missing error_ids: [0, 1]" in caplog.text or "Missing error_ids: [0, 2]" in caplog.text

    # Verify clustered results
    assert isinstance(result, dict)
    assert len(result) == 2  # One named cluster + None cluster
    assert result["Communication gaps"] == [mock_analyzed_subgoal_validation_2]

    # Verify missing error_ids were added to None cluster in sorted order
    assert None in result
    assert result[None] == [
        mock_analyzed_subgoal_validation_1,
        mock_analyzed_subgoal_validation_3,
    ]


def test_build_clustered_result_no_missing_error_ids(
    analyzer,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_3,
    sample_llm_clusterings,
    caplog,
):
    """Test that no None cluster is created when all error_ids are assigned."""
    import logging

    with caplog.at_level(logging.WARNING):
        result = analyzer._build_clustered_result(
            sample_llm_clusterings,
            [
                mock_analyzed_subgoal_validation_1,
                mock_analyzed_subgoal_validation_2,
                mock_analyzed_subgoal_validation_3,
            ],
        )

    # Verify no warning was logged
    assert "LLM clustering missed" not in caplog.text

    # Verify no None cluster exists
    assert None not in result
    assert len(result) == 2  # Only the two named clusters


# ==================== Async tests (test the real business logic) ====================


@pytest.mark.asyncio
async def test_analyze_batch_async_returns_error_analysis_result(
    analyzer,
    monkeypatch,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_3,
    mock_analyzed_subgoal_validation_4,
    sample_llm_clusterings,
):
    """Test that analyze_batch_async() processes multiple data samples and returns clustered results."""
    data_sample_1 = SubgoalErrorAnalysisDataSample(data_sample_id=1, subgoal_validations=[])
    data_sample_2 = SubgoalErrorAnalysisDataSample(data_sample_id=2, subgoal_validations=[])

    # Mock _analyze to return analyzed subgoal validations
    analyze_mock = AsyncMock(
        side_effect=[
            [mock_analyzed_subgoal_validation_1, mock_analyzed_subgoal_validation_4],
            [mock_analyzed_subgoal_validation_2, mock_analyzed_subgoal_validation_3],
        ]
    )
    monkeypatch.setattr(analyzer, "_analyze", analyze_mock)

    # Mock _cluster_errors
    cluster_mock = AsyncMock(return_value=sample_llm_clusterings)
    monkeypatch.setattr(analyzer, "_cluster_errors", cluster_mock)

    # Mock _build_clustered_result
    expected_clustered = {
        "Calendar tool issues": [mock_analyzed_subgoal_validation_1],
        "Communication gaps": [
            mock_analyzed_subgoal_validation_2,
            mock_analyzed_subgoal_validation_3,
        ],
    }
    monkeypatch.setattr(
        analyzer,
        "_build_clustered_result",
        lambda _llm_cluster, _asv_list: expected_clustered,
    )

    result = await analyzer.analyze_batch_async([data_sample_1, data_sample_2])

    # Verify only incomplete validations were passed to clustering
    cluster_mock.assert_called_once_with(
        [
            mock_analyzed_subgoal_validation_1,
            mock_analyzed_subgoal_validation_2,
            mock_analyzed_subgoal_validation_3,
        ]
    )

    # Verify the result structure
    assert isinstance(result, SubgoalErrorAnalysisResult)
    assert result.analyzed_validations_clustered_by_errors == expected_clustered
    assert len(result.completed_subgoal_validations) == 1
    assert result.completed_subgoal_validations[0] == mock_analyzed_subgoal_validation_4


@pytest.mark.asyncio
async def test_analyze_batch_async_handles_empty_data_samples(analyzer, monkeypatch):
    """Test that analyze_batch_async() handles empty data samples correctly."""
    cluster_mock = AsyncMock(return_value={"clusters": []})
    monkeypatch.setattr(analyzer, "_cluster_errors", cluster_mock)
    monkeypatch.setattr(analyzer, "_build_clustered_result", lambda _llm_cluster, _asv_list: {})

    result = await analyzer.analyze_batch_async([])

    assert isinstance(result, SubgoalErrorAnalysisResult)
    assert result.analyzed_validations_clustered_by_errors == {}
    assert result.completed_subgoal_validations == []


# ==================== Sync wrapper tests (verify delegation only) ====================


def test_analyze_batch_delegates_to_async_version(analyzer, monkeypatch):
    """Test that analyze_batch() is a thin wrapper that delegates to analyze_batch_async()."""
    expected_result = SubgoalErrorAnalysisResult(
        analyzed_validations_clustered_by_errors={"Error Type": []},
        completed_subgoal_validations=[],
    )

    # Mock analyze_batch_async to return expected result
    async_mock = AsyncMock(return_value=expected_result)
    monkeypatch.setattr(analyzer, "analyze_batch_async", async_mock)

    result = analyzer.analyze_batch([])

    # Verify it returns the result from analyze_batch_async
    assert result == expected_result
    async_mock.assert_called_once_with([])


@pytest.mark.asyncio
async def test_analyze_batch_raises_when_called_in_async_context(analyzer):
    """Test that analyze_batch() raises an error if called within an existing event loop."""
    with pytest.raises(
        RuntimeError,
        match="analyze_batch cannot be called from an async context. Use analyze_batch_async instead.",
    ):
        analyzer.analyze_batch([])
