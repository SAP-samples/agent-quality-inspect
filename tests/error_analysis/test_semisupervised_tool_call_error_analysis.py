"""
Unit tests for the SemisupervisedToolCallErrorAnalysis class.
"""

from dataclasses import dataclass
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_inspect.exception import InvalidInputValueError
from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.metrics.constants import STATUS_200
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.models.llm_response import LLMResponse
from agent_inspect.models.metrics.validation_result import ToolCallValidationResult
from agent_inspect.models.metrics.agent_data_sample import (
    ExpectedToolCall,
    ToolInputParameter,
    ToolOutput,
)
from agent_inspect.tools.error_analysis.llm_output_schemas import (
    TOOL_VALIDATION_CLASSIFICATION_OUTPUT_SCHEMA,
)
from agent_inspect.tools.error_analysis.semisupervised_tool_call_error_analysis import (
    SemisupervisedToolCallErrorAnalysis,
)
from agent_inspect.models.tools.analysis_models import AnalyzedToolValidation
from agent_inspect.models.tools.error_cluster import (
    ErrorCluster,
    IncorrectToolInputCluster,
    DEFAULT_TOOL_CALL_ERROR_CLUSTERS,
)


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.make_request_with_payload = AsyncMock()
    return client


@pytest.fixture
def analyzer(mock_llm_client):
    return SemisupervisedToolCallErrorAnalysis(
        llm_client=mock_llm_client, config={"max_workers": 1}
    )


@pytest.fixture
def mock_expected_tool_call():
    return ExpectedToolCall(
        tool="reminder_tool",
        expected_parameters=[
            ToolInputParameter(name="message", check="Should contain reminder text"),
            ToolInputParameter(name="time", value="10:00"),
        ],
        expected_output=ToolOutput(check="Should confirm reminder was set"),
    )


@pytest.fixture
def mock_failed_validation(mock_expected_tool_call):
    return ToolCallValidationResult(
        is_completed=False,
        expected_tool_call=mock_expected_tool_call,
        explanations=[
            'Tool "reminder_tool" call has failed input arguments check or output check.',
            "Tool name: reminder_tool matched.",
            'Argument "message" has failed llm-as-a-judge.',
            "GRADE: I",
        ],
    )


# ==================== Tests for __init__ ====================


def test_init_uses_default_clusters(mock_llm_client):
    """Test that default clusters are used when none provided."""
    analyzer = SemisupervisedToolCallErrorAnalysis(llm_client=mock_llm_client)
    assert analyzer.error_clusters == DEFAULT_TOOL_CALL_ERROR_CLUSTERS


def test_init_uses_default_clusters_when_empty_list(mock_llm_client):
    """Test that default clusters are used when empty list provided."""
    analyzer = SemisupervisedToolCallErrorAnalysis(llm_client=mock_llm_client, error_clusters=[])
    assert analyzer.error_clusters == DEFAULT_TOOL_CALL_ERROR_CLUSTERS


def test_init_accepts_custom_clusters(mock_llm_client):
    """Test that custom clusters are accepted."""

    @dataclass(frozen=True)
    class CustomCluster(ErrorCluster):
        cluster_label: str = "Custom Cluster"
        description: str = "Custom cluster for testing"

    custom_clusters = [IncorrectToolInputCluster(), CustomCluster()]
    analyzer = SemisupervisedToolCallErrorAnalysis(
        llm_client=mock_llm_client, error_clusters=custom_clusters
    )
    assert analyzer.error_clusters == custom_clusters


def test_init_raises_for_none_llm_client():
    """Test that None llm_client raises InvalidInputValueError."""
    with pytest.raises(InvalidInputValueError, match="LLM client must be provided"):
        SemisupervisedToolCallErrorAnalysis(llm_client=None)


def test_init_raises_for_invalid_clusters(mock_llm_client):
    """Test that invalid clusters raise InvalidInputValueError."""
    with pytest.raises(InvalidInputValueError, match="must be a ErrorCluster instance"):
        SemisupervisedToolCallErrorAnalysis(
            llm_client=mock_llm_client,
            error_clusters=[{"cluster_label": "bad", "description": "not a dataclass"}],
        )


# ==================== Tests for _classify_single_validation ====================


@pytest.mark.asyncio
async def test_classify_single_validation_builds_correct_llm_payload(
    analyzer, mock_llm_client, mock_failed_validation
):
    """Test that _classify_single_validation builds the correct LLM payload."""
    classification_response = {
        "cluster_label": "Incorrect Tool Input",
        "is_predefined": True,
        "error_type": "Argument validation failure",
        "rationale": "Message argument failed LLM judge",
    }

    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200, completion=json.dumps(classification_response)
    )

    analyzed, cluster_label = await analyzer._classify_single_validation(
        validation_result=mock_failed_validation, data_sample_id=1, agent_run_id=100
    )

    # Verify the LLM was called with the right payload
    call_args = mock_llm_client.make_request_with_payload.call_args[0][0]
    assert isinstance(call_args, LLMPayload)

    # Verify predefined clusters are in the prompt
    clusters_dicts = [c.to_dict() for c in DEFAULT_TOOL_CALL_ERROR_CLUSTERS]
    expected_clusters = json.dumps(clusters_dicts, indent=2)
    assert expected_clusters in call_args.user_prompt

    # Verify tool info is in the prompt
    assert "reminder_tool" in call_args.user_prompt

    # Verify structured output schema
    assert call_args.structured_output == TOOL_VALIDATION_CLASSIFICATION_OUTPUT_SCHEMA


@pytest.mark.asyncio
async def test_classify_single_validation_returns_correct_tuple(
    analyzer, mock_llm_client, mock_failed_validation
):
    """Test that _classify_single_validation returns (AnalyzedToolValidation, cluster_label) tuple."""
    classification_response = {
        "cluster_label": "Incorrect Tool Input",
        "is_predefined": True,
        "error_type": "Argument failure",
        "rationale": "reason",
    }

    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200, completion=json.dumps(classification_response)
    )

    analyzed, cluster_label = await analyzer._classify_single_validation(
        validation_result=mock_failed_validation, data_sample_id=42, agent_run_id=99
    )

    assert isinstance(analyzed, AnalyzedToolValidation)
    assert analyzed.data_sample_id == 42
    assert analyzed.agent_run_id == 99
    assert analyzed.tool_call_validation == mock_failed_validation
    assert analyzed.base_error == "\n".join(mock_failed_validation.explanations)
    assert cluster_label == "Incorrect Tool Input"
