"""
Unit tests for ToolCallErrorAnalysis (abstract base class).

Tests cover the shared analyze_batch method defined in tool_call_error_analysis.py.
Uses a concrete stub with mocked _classify_single_validation to isolate the
parent class logic.
"""

from unittest.mock import AsyncMock

import pytest

from agent_inspect.models.metrics.validation_result import ToolCallValidationResult
from agent_inspect.models.metrics.agent_data_sample import (
    ExpectedToolCall,
    ToolInputParameter,
    ToolOutput,
)
from agent_inspect.models.tools import (
    AnalyzedToolValidation,
    ToolCallErrorAnalysisDataSample,
    ToolCallErrorAnalysisResult,
)
from agent_inspect.tools.error_analysis.tool_call_error_analysis import (
    ToolCallErrorAnalysis,
)


class ConcreteToolCallErrorAnalysis(ToolCallErrorAnalysis):
    """Concrete stub for testing the abstract ToolCallErrorAnalysis."""

    async def _classify_single_validation(
        self, validation_result, data_sample_id, agent_run_id=None
    ):
        return (
            AnalyzedToolValidation(
                data_sample_id=data_sample_id,
                base_error="\n".join(validation_result.explanations),
                tool_call_validation=validation_result,
                agent_run_id=agent_run_id,
            ),
            "Cluster A",
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


@pytest.fixture
def mock_passed_validation(mock_expected_tool_call):
    return ToolCallValidationResult(
        is_completed=True,
        expected_tool_call=mock_expected_tool_call,
        explanations=[
            'Tool "reminder_tool" call has passed all checks.',
        ],
    )


@pytest.fixture
def analyzer():
    return ConcreteToolCallErrorAnalysis()


# ==================== Async tests (test the real business logic) ====================


@pytest.mark.asyncio
async def test_analyze_batch_async_returns_result_with_failures(analyzer, mock_failed_validation):
    """Test that analyze_batch_async() processes data samples and returns clustered results."""
    data_samples = [
        ToolCallErrorAnalysisDataSample(
            data_sample_id=1,
            tool_call_validations=[mock_failed_validation],
            agent_run_id=100,
        )
    ]

    result = await analyzer.analyze_batch_async(data_samples)

    assert isinstance(result, ToolCallErrorAnalysisResult)
    assert "Cluster A" in result.analyzed_validations_clustered_by_errors
    assert len(result.analyzed_validations_clustered_by_errors["Cluster A"]) == 1
    analyzed = result.analyzed_validations_clustered_by_errors["Cluster A"][0]
    assert analyzed.data_sample_id == 1
    assert analyzed.agent_run_id == 100
    assert analyzed.base_error is not None


@pytest.mark.asyncio
async def test_analyze_batch_async_skips_completed_validations(analyzer, mock_passed_validation):
    """Test that analyze_batch_async() skips completed validations."""
    data_samples = [
        ToolCallErrorAnalysisDataSample(
            data_sample_id=1, tool_call_validations=[mock_passed_validation]
        )
    ]

    result = await analyzer.analyze_batch_async(data_samples)

    assert isinstance(result, ToolCallErrorAnalysisResult)
    assert len(result.analyzed_validations_clustered_by_errors) == 0


@pytest.mark.asyncio
async def test_analyze_batch_async_handles_mixed_passed_and_failed(
    analyzer, mock_failed_validation, mock_passed_validation
):
    """Test that analyze_batch_async() correctly processes mixed passed and failed validations."""
    data_samples = [
        ToolCallErrorAnalysisDataSample(
            data_sample_id=1,
            tool_call_validations=[mock_failed_validation, mock_passed_validation],
        )
    ]

    result = await analyzer.analyze_batch_async(data_samples)

    assert len(result.analyzed_validations_clustered_by_errors["Cluster A"]) == 1


@pytest.mark.asyncio
async def test_analyze_batch_async_handles_empty_data_samples(analyzer):
    """Test that analyze_batch_async() handles empty data samples correctly."""
    result = await analyzer.analyze_batch_async([])

    assert isinstance(result, ToolCallErrorAnalysisResult)
    assert len(result.analyzed_validations_clustered_by_errors) == 0


@pytest.mark.asyncio
async def test_analyze_batch_async_handles_multiple_data_samples(
    analyzer, mock_failed_validation, mock_expected_tool_call
):
    """Test that analyze_batch_async() processes multiple data samples correctly."""
    second_failed = ToolCallValidationResult(
        is_completed=False,
        expected_tool_call=mock_expected_tool_call,
        explanations=["Some other failure."],
    )

    data_samples = [
        ToolCallErrorAnalysisDataSample(
            data_sample_id=1,
            tool_call_validations=[mock_failed_validation],
            agent_run_id=100,
        ),
        ToolCallErrorAnalysisDataSample(
            data_sample_id=2, tool_call_validations=[second_failed], agent_run_id=200
        ),
    ]

    result = await analyzer.analyze_batch_async(data_samples)

    assert len(result.analyzed_validations_clustered_by_errors["Cluster A"]) == 2
    ids = [a.data_sample_id for a in result.analyzed_validations_clustered_by_errors["Cluster A"]]
    assert 1 in ids
    assert 2 in ids


@pytest.mark.asyncio
async def test_analyze_batch_async_groups_by_cluster_label(
    analyzer, monkeypatch, mock_failed_validation, mock_expected_tool_call
):
    """Test that analyze_batch_async() groups results correctly when _classify returns different cluster labels."""
    second_failed = ToolCallValidationResult(
        is_completed=False,
        expected_tool_call=mock_expected_tool_call,
        explanations=["Different failure."],
    )

    call_count = 0

    async def mock_classify(validation_result, data_sample_id, agent_run_id=None):
        nonlocal call_count
        call_count += 1
        label = "Cluster A" if call_count == 1 else "Cluster B"
        return (
            AnalyzedToolValidation(
                data_sample_id=data_sample_id,
                base_error="\n".join(validation_result.explanations),
                tool_call_validation=validation_result,
                agent_run_id=agent_run_id,
            ),
            label,
        )

    monkeypatch.setattr(analyzer, "_classify_single_validation", mock_classify)

    data_samples = [
        ToolCallErrorAnalysisDataSample(
            data_sample_id=1,
            tool_call_validations=[mock_failed_validation],
            agent_run_id=100,
        ),
        ToolCallErrorAnalysisDataSample(
            data_sample_id=2, tool_call_validations=[second_failed], agent_run_id=200
        ),
    ]

    result = await analyzer.analyze_batch_async(data_samples)

    assert "Cluster A" in result.analyzed_validations_clustered_by_errors
    assert "Cluster B" in result.analyzed_validations_clustered_by_errors
    assert len(result.analyzed_validations_clustered_by_errors["Cluster A"]) == 1
    assert len(result.analyzed_validations_clustered_by_errors["Cluster B"]) == 1


def test_analyze_batch_delegates_to_async_version(analyzer):
    """Test that analyze_batch() is a thin wrapper that delegates to analyze_batch_async()."""
    expected_result = ToolCallErrorAnalysisResult(
        analyzed_validations_clustered_by_errors={"Error Type": []}
    )

    # Mock analyze_batch_async to return expected result
    analyzer.analyze_batch_async = AsyncMock(return_value=expected_result)

    result = analyzer.analyze_batch([])

    # Verify it returns the result from analyze_batch_async
    assert result == expected_result
    analyzer.analyze_batch_async.assert_called_once_with([])


@pytest.mark.asyncio
async def test_analyze_batch_raises_when_called_in_async_context(analyzer):
    """Test that analyze_batch() raises an error if called within an existing event loop."""
    with pytest.raises(
        RuntimeError,
        match="analyze_batch cannot be called from an async context. Use analyze_batch_async instead.",
    ):
        analyzer.analyze_batch([])
