"""
Unit tests for the SemisupervisedSubgoalErrorAnalysis class.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import asyncio

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.exception.exception import InvalidInputValueError
from agent_inspect.metrics.constants import STATUS_200
from agent_inspect.models.metrics.agent_data_sample import SubGoal
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.models.llm_response import LLMResponse
from agent_inspect.tools.error_analysis.llm_output_schemas import (
    SEMI_SUPERVISED_CLUSTERING_OUTPUT_SCHEMA,
)
from agent_inspect.tools.error_analysis.semisupervised_subgoal_error_analysis import (
    SemisupervisedSubgoalErrorAnalysis,
)
from agent_inspect.models.tools.analysis_models import AnalyzedSubgoalValidation
from agent_inspect.models.tools.error_cluster import (
    IncorrectToolInputCluster,
    WrongToolSelectionCluster,
    DEFAULT_SUBGOAL_ERROR_CLUSTERS,
)


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
    return SemisupervisedSubgoalErrorAnalysis(llm_client=mock_llm_client, config={"max_workers": 1})


@pytest.fixture
def mock_subgoal_1():
    return SubGoal(details="Ensure calendar tool execution")


@pytest.fixture
def mock_subgoal_2():
    return SubGoal(details="Draft an email to the client")


# ==================== Tests for __init__ ====================


def test_init_uses_default_clusters(mock_llm_client):
    """Test that default clusters are used when none provided."""
    analyzer = SemisupervisedSubgoalErrorAnalysis(llm_client=mock_llm_client)
    assert analyzer.error_clusters == DEFAULT_SUBGOAL_ERROR_CLUSTERS


def test_init_uses_default_clusters_when_empty_list(mock_llm_client):
    """Test that default clusters are used when empty list provided."""
    analyzer = SemisupervisedSubgoalErrorAnalysis(llm_client=mock_llm_client, error_clusters=[])
    assert analyzer.error_clusters == DEFAULT_SUBGOAL_ERROR_CLUSTERS


def test_init_accepts_custom_clusters(mock_llm_client):
    """Test that custom clusters are accepted."""
    custom_clusters = [
        IncorrectToolInputCluster(),
        WrongToolSelectionCluster(),
    ]
    analyzer = SemisupervisedSubgoalErrorAnalysis(
        llm_client=mock_llm_client, error_clusters=custom_clusters
    )
    assert analyzer.error_clusters == custom_clusters
    assert len(analyzer.error_clusters) == 2


def test_init_raises_for_invalid_clusters(mock_llm_client):
    """Test that invalid clusters raise InvalidInputValueError."""
    with pytest.raises(InvalidInputValueError, match="must be a ErrorCluster instance"):
        SemisupervisedSubgoalErrorAnalysis(
            llm_client=mock_llm_client,
            error_clusters=[{"cluster_label": "bad", "description": "not a dataclass"}],
        )


# ==================== Tests for _summarize_errors_into_base_error ====================


@pytest.mark.asyncio
async def test_summarize_errors_into_base_error_returns_none_when_completed(
    analyzer, mock_subgoal_1
):
    """Test that completed validations return None."""
    completed_validation = SubGoalValidationResult(
        is_completed=True,
        explanations=["Overall: Completed", "Trial 1: C"],
        sub_goal=mock_subgoal_1,
    )

    result = await analyzer._summarize_errors_into_base_error(completed_validation)

    assert result is None


@pytest.mark.asyncio
async def test_summarize_errors_into_base_error_uses_first_trial_for_consistent_failure(
    analyzer, monkeypatch, mock_subgoal_1
):
    """Test that consistent failures summarize the first trial explanation."""
    validation = SubGoalValidationResult(
        is_completed=False,
        explanations=[
            "Overall explanation.",
            "Judge Trial 1. Agent failed.\n\nGrade: I",
            "Judge Trial 2. Agent also failed.\n\nGrade: I",
        ],
        sub_goal=mock_subgoal_1,
    )

    monkeypatch.setattr(analyzer, "_has_failed_consistently", lambda _: True)
    summarize_mock = AsyncMock(return_value="Calendar tool misuse")
    monkeypatch.setattr(analyzer, "_summarize_error", summarize_mock)

    result = await analyzer._summarize_errors_into_base_error(validation)

    assert result == "Calendar tool misuse"
    summarize_mock.assert_awaited_once_with(
        "Judge Trial 1. Agent failed.\n\nGrade: I", "Ensure calendar tool execution"
    )


@pytest.mark.asyncio
async def test_summarize_errors_into_base_error_uses_first_incomplete_trial_for_mixed_results(
    analyzer, monkeypatch, mock_subgoal_2
):
    """Test that mixed results use the first incomplete trial (score=0), not majority voting."""
    validation = SubGoalValidationResult(
        is_completed=False,
        explanations=[
            "Overall explanation.",
            "Judge Trial 1. Agent succeeded.\n\nGrade: C",
            "Judge Trial 2. Agent failed to draft email.\n\nGrade: I",
        ],
        sub_goal=mock_subgoal_2,
    )

    monkeypatch.setattr(analyzer, "_has_failed_consistently", lambda _: False)
    summarize_mock = AsyncMock(return_value="Email drafting failure")
    monkeypatch.setattr(analyzer, "_summarize_error", summarize_mock)

    result = await analyzer._summarize_errors_into_base_error(validation)

    assert result == "Email drafting failure"
    # Should be called with the second trial (first incomplete one)
    summarize_mock.assert_awaited_once_with(
        "Judge Trial 2. Agent failed to draft email.\n\nGrade: I",
        "Draft an email to the client",
    )


# ==================== Tests for _cluster_errors ====================


@pytest.mark.asyncio
async def test_cluster_errors_builds_correct_llm_payload(
    analyzer, mock_llm_client, mock_subgoal_1, mock_subgoal_2
):
    """Test that _cluster_errors builds the correct LLM payload with predefined clusters."""
    analyzed_validations = [
        AnalyzedSubgoalValidation(
            subgoal_validation=SubGoalValidationResult(
                is_completed=False,
                explanations=["Overall", "Trial 1\n\nGrade: I"],
                sub_goal=mock_subgoal_1,
            ),
            data_sample_id=1,
            base_error="Tool misuse",
        ),
        AnalyzedSubgoalValidation(
            subgoal_validation=SubGoalValidationResult(
                is_completed=False,
                explanations=["Overall", "Trial 1\n\nGrade: I"],
                sub_goal=mock_subgoal_2,
            ),
            data_sample_id=2,
            base_error="Incomplete email",
        ),
    ]

    clustering_response = {
        "clusters": [
            {
                "cluster_label": "Incorrect Tool Input",
                "is_predefined": True,
                "error_types": ["Tool misuse"],
                "error_ids": ["0"],
                "rationale": "Tool usage error",
            },
            {
                "cluster_label": "Incomplete or Vague Communication",
                "is_predefined": True,
                "error_types": ["Incomplete email"],
                "error_ids": ["1"],
                "rationale": "Communication error",
            },
        ]
    }

    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200, completion=json.dumps(clustering_response)
    )

    result = await analyzer._cluster_errors(analyzed_validations)

    # Verify the LLM was called with the right payload
    call_args = mock_llm_client.make_request_with_payload.call_args[0][0]
    assert isinstance(call_args, LLMPayload)

    # Verify predefined clusters are included in the prompt
    predefined_clusters_dicts = [c.to_dict() for c in DEFAULT_SUBGOAL_ERROR_CLUSTERS]
    expected_predefined = json.dumps(predefined_clusters_dicts, indent=2)
    assert expected_predefined in call_args.user_prompt

    # Verify error types mapping is included
    expected_error_types = json.dumps({"0": "Tool misuse", "1": "Incomplete email"}, indent=2)
    assert expected_error_types in call_args.user_prompt

    # Verify subgoals are included
    expected_subgoals = json.dumps(
        ["Ensure calendar tool execution", "Draft an email to the client"], indent=2
    )
    assert expected_subgoals in call_args.user_prompt

    # Verify structured output schema
    assert call_args.structured_output == SEMI_SUPERVISED_CLUSTERING_OUTPUT_SCHEMA

    assert result == clustering_response
