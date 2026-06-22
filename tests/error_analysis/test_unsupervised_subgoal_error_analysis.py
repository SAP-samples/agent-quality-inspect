"""
Unit tests for UnsupervisedSubgoalErrorAnalysis.

Tests cover methods specific to the unsupervised implementation:
- _summarize_errors_into_base_error (unsupervised majority voting approach)
- _cluster_errors (unsupervised clustering)
- _perform_majority_voting
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.metrics.constants import STATUS_200
from agent_inspect.exception import ToolError
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.models.metrics.agent_data_sample import SubGoal
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult
from agent_inspect.models.llm_response import LLMResponse
from agent_inspect.tools.error_analysis.llm_output_schemas import (
    UNSUPERVISED_CLUSTERING_OUTPUT_SCHEMA,
)
from agent_inspect.tools.error_analysis.unsupervised_subgoal_error_analysis import (
    UnsupervisedSubgoalErrorAnalysis,
)
from agent_inspect.tools.error_analysis.llm_templates import (
    UNSUPERVISED_CLUSTERING_PROMPT_TEMPLATE,
)
from agent_inspect.models.tools.analysis_models import AnalyzedSubgoalValidation


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.make_request_with_payload = AsyncMock()
    return client


@pytest.fixture
def analyzer(mock_llm_client):
    return UnsupervisedSubgoalErrorAnalysis(llm_client=mock_llm_client, config={"max_workers": 20})


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


# ==================== Tests for _summarize_errors_into_base_error ====================


@pytest.mark.asyncio
async def test_summarize_errors_into_base_error_returns_none_when_completed(
    analyzer, mock_subgoal_1
):
    completed_validation = SubGoalValidationResult(
        is_completed=True,
        explanations=["Overall: Completed", "Trial 1: C"],
        sub_goal=mock_subgoal_1,
    )

    result = await analyzer._summarize_errors_into_base_error(completed_validation)

    assert result is None


@pytest.mark.asyncio
async def test_summarize_errors_into_base_error_returns_base_error_when_consistent(
    analyzer, monkeypatch, mock_subgoal_validation_result_all_incomplete
):
    monkeypatch.setattr(analyzer, "_has_failed_consistently", lambda _: True)
    summarize_mock = AsyncMock(return_value="Base error")
    monkeypatch.setattr(analyzer, "_summarize_error", summarize_mock)

    result = await analyzer._summarize_errors_into_base_error(
        mock_subgoal_validation_result_all_incomplete
    )

    assert result == "Base error"
    summarize_mock.assert_awaited_once_with(
        mock_subgoal_validation_result_all_incomplete.explanations[1],
        mock_subgoal_validation_result_all_incomplete.sub_goal.details,
    )


@pytest.mark.asyncio
async def test_summarize_errors_into_base_error_performs_majority_vote_when_inconsistent(
    analyzer, monkeypatch, mock_subgoal_validation_result_some_complete
):
    monkeypatch.setattr(analyzer, "_has_failed_consistently", lambda _: False)
    summarize_mock = AsyncMock(side_effect=["Error one", "Error two"])
    majority_mock = AsyncMock(return_value="Majority error")
    monkeypatch.setattr(analyzer, "_summarize_error", summarize_mock)
    monkeypatch.setattr(analyzer, "_perform_majority_voting", majority_mock)

    result = await analyzer._summarize_errors_into_base_error(
        mock_subgoal_validation_result_some_complete
    )

    assert result == "Majority error"
    assert summarize_mock.await_count == 2
    majority_mock.assert_awaited_once_with(["Error one", "Error two"])


# ==================== Tests for _perform_majority_voting ====================


@pytest.mark.asyncio
async def test_perform_majority_voting_returns_most_probable_error(analyzer, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200,
        completion=json.dumps({"most_probable_error_type": "Tool misuse"}),
    )

    result = await analyzer._perform_majority_voting(["Err A", "Err B"])

    assert result == "Tool misuse"
    mock_llm_client.make_request_with_payload.assert_awaited_once()


@pytest.mark.asyncio
async def test_perform_majority_voting_raises_when_llm_fails(analyzer, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=500, completion=None, error_message="Failure"
    )

    with pytest.raises(ToolError, match="LLM request failed with status 500 and error: Failure"):
        await analyzer._perform_majority_voting(["Err A", "Err B"])


@pytest.mark.asyncio
async def test_perform_majority_voting_raises_when_key_missing(analyzer, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200,
        completion=json.dumps({"wrong_key": "some value", "explanation": "some explanation"}),
    )

    with pytest.raises(
        Exception,
        match="LLM majority voting request failed as no most_probable_error_type found in response",
    ):
        await analyzer._perform_majority_voting(["Error A", "Error B"])


# ==================== Tests for _cluster_errors ====================


@pytest.mark.asyncio
async def test_cluster_errors_successfully_returns_clusters(
    analyzer,
    mock_llm_client,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_3,
    sample_llm_clusterings,
):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200, completion=json.dumps(sample_llm_clusterings)
    )
    analyzed = [
        mock_analyzed_subgoal_validation_1,
        mock_analyzed_subgoal_validation_2,
        mock_analyzed_subgoal_validation_3,
    ]

    result = await analyzer._cluster_errors(analyzed)

    expected_error_types = json.dumps(
        {
            "0": "Tool misuse",
            "1": "Incomplete task execution",
            "2": "Task not fully completed",
        },
        indent=2,
    )
    expected_subgoals = json.dumps(
        ["Ensure calendar tool execution", "Draft an email to the client"], indent=2
    )
    expected_payload = LLMPayload(
        user_prompt=UNSUPERVISED_CLUSTERING_PROMPT_TEMPLATE.format(
            error_types=expected_error_types, subgoals=expected_subgoals
        ),
        structured_output=UNSUPERVISED_CLUSTERING_OUTPUT_SCHEMA,
    )

    mock_llm_client.make_request_with_payload.assert_called_with(expected_payload)
    assert result == sample_llm_clusterings


@pytest.mark.asyncio
async def test_cluster_errors_raises_when_llm_fails(
    analyzer, mock_llm_client, mock_analyzed_subgoal_validation_1
):
    analyzed = [mock_analyzed_subgoal_validation_1]
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=500, completion=None, error_message="Failure"
    )

    with pytest.raises(ToolError, match="LLM request failed with status 500 and error: Failure"):
        _ = await analyzer._cluster_errors(analyzed)
