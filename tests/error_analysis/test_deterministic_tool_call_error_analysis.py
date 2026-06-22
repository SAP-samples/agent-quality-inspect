"""
Unit tests for the DeterministicToolCallErrorAnalysis class.
"""

import pytest

from agent_inspect.models.metrics.agent_data_sample import (
    ExpectedToolCall,
    ToolInputParameter,
    ToolOutput,
)
from agent_inspect.models.metrics.validation_result import ToolCallValidationResult
from agent_inspect.models.tools.analysis_models import AnalyzedToolValidation
from agent_inspect.models.tools.error_cluster import DEFAULT_TOOL_CALL_ERROR_CLUSTERS
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.tools.error_analysis.deterministic_tool_call_error_analysis import (
    DeterministicToolCallErrorAnalysis,
)


@pytest.fixture
def analyzer():
    return DeterministicToolCallErrorAnalysis(config={"max_workers": 1})


# ==================== Tests for _classify_from_explanations ====================


def test_classify_from_explanations_tool_not_found(analyzer):
    """Test classification when tool is not found."""
    explanations = [
        'No matching tool name "send_email" is found for expected tool in this agent turn'
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    assert cluster_label == "Wrong Tool Selection"


def test_classify_from_explanations_argument_failure_exact_match(analyzer):
    """Test classification when an argument fails exact match validation."""
    explanations = [
        'Tool "calculator" call has failed input arguments check or output check.',
        "Tool name: calculator matched.",
        "Argument \"expression\" has failed exact match. Expected value: 2+2, Expected type: <class 'str'>. Actual value: 3+3, Actual type: <class 'str'>.",
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    assert cluster_label == "Incorrect Tool Input"


def test_classify_from_explanations_argument_failure_llmj(analyzer):
    """Test classification when an argument fails LLM judge validation."""
    explanations = [
        'Tool "reminder_tool" call has failed input arguments check or output check.',
        "Tool name: reminder_tool matched.",
        'Argument "reminder_message" has failed llm-as-a-judge.',
        "The message content is incorrect... GRADE: I",
        'Argument "time" has passed exact match successfully.',
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    assert cluster_label == "Incorrect Tool Input"


def test_classify_from_explanations_argument_not_found(analyzer):
    """Test classification when an expected argument is not found in actual tool call."""
    explanations = [
        'Tool "search_tool" call has failed input arguments check or output check.',
        "Tool name: search_tool matched.",
        "Argument \"query\" not even found in actual tool call. Expected value: test search, Expected type: <class 'str'>.",
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    assert cluster_label == "Incorrect Tool Input"


def test_classify_from_explanations_output_failure_exact_match(analyzer):
    """Test classification when only tool output fails exact match validation."""
    explanations = [
        'Tool "calculator" call has failed input arguments check or output check.',
        "Tool name: calculator matched.",
        'Argument "expression" has passed exact match successfully.',
        "Tool output has failed exact match. Expected output: 4, Expected type: <class 'int'>. Actual output: 6, Actual type: <class 'int'>.",
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    assert cluster_label == "Incorrect Tool Output Handling"


def test_classify_from_explanations_output_failure_llmj(analyzer):
    """Test classification when tool output fails LLM judge validation."""
    explanations = [
        'Tool "reminder_tool" call has failed input arguments check or output check.',
        "Tool name: reminder_tool matched.",
        'Argument "time" has passed exact match successfully.',
        "Tool output has failed llm-as-a-judge.",
        "The output is missing information. GRADE: I",
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    assert cluster_label == "Incorrect Tool Output Handling"


def test_classify_from_explanations_multiple_failed_arguments(analyzer):
    """Test classification with multiple failed arguments."""
    explanations = [
        'Tool "reminder_tool" call has failed input arguments check or output check.',
        "Tool name: reminder_tool matched.",
        'Argument "message" has failed llm-as-a-judge.',
        "GRADE: I",
        "Argument \"time\" has failed exact match. Expected value: 10:00, Expected type: <class 'str'>. Actual value: 11:00, Actual type: <class 'str'>.",
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    assert cluster_label == "Incorrect Tool Input"


def test_classify_from_explanations_priority_argument_over_output(analyzer):
    """Test that argument failures take priority over output failures."""
    explanations = [
        'Tool "reminder_tool" call has failed input arguments check or output check.',
        "Tool name: reminder_tool matched.",
        'Argument "message" has failed llm-as-a-judge.',
        "GRADE: I",
        "Tool output has failed llm-as-a-judge.",
        "GRADE: I",
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    # Should be "Incorrect Tool Input" even though output also failed
    assert cluster_label == "Incorrect Tool Input"


def test_classify_from_explanations_priority_tool_not_found_over_others(analyzer):
    """Test that tool not found takes highest priority."""
    explanations = [
        'No matching tool name "send_email" is found for expected tool in this agent turn',
        'Argument "message" has failed llm-as-a-judge.',
        "Tool output has failed llm-as-a-judge.",
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    assert cluster_label == "Wrong Tool Selection"


def test_classify_from_explanations_user_example(analyzer):
    """Test classification with the complex user example."""
    explanations = [
        'Tool "reminder_tool" call has failed input arguments check or output check.',
        "Tool name: reminder_tool matched.",
        'Argument "reminder_message" has failed llm-as-a-judge.',
        'The [Argument Value] "Purchase a loaf of butter" does not semantically match the [Ground Truth Value] "Buy a loaf of bread at 10:00 AM tmr." The main discrepancy is that "butter" and "bread" are entirely different items.\n\nGRADE: I',
        'Argument "time" has passed exact match successfully.',
        "Tool output has failed llm-as-a-judge.",
        "The [Argument Value] states that a reminder has been set for 10:00 AM tomorrow with the message, but it does not specify the actual message content.\n\nGRADE: I",
    ]

    cluster_label = analyzer._classify_from_explanations(explanations)

    # Argument failure takes priority over output failure
    assert cluster_label == "Incorrect Tool Input"


def test_classify_from_explanations_fallback_for_unknown_format(analyzer):
    """Test that unrecognized explanation format raises InvalidInputValueError."""
    explanations = [
        "Some completely unexpected explanation format.",
    ]

    with pytest.raises(InvalidInputValueError, match="Unable to classify validation failure"):
        analyzer._classify_from_explanations(explanations)


def test_classify_from_explanations_empty_explanations(analyzer):
    """Test that empty explanation list raises InvalidInputValueError."""
    with pytest.raises(InvalidInputValueError, match="Validation explanations cannot be empty"):
        analyzer._classify_from_explanations([])


def test_classify_from_explanations_none_explanations(analyzer):
    """Test that None explanations raises InvalidInputValueError."""
    with pytest.raises(InvalidInputValueError, match="Validation explanations cannot be empty"):
        analyzer._classify_from_explanations(None)


# ==================== Tests for _classify_single_validation ====================


@pytest.mark.asyncio
async def test_classify_single_validation_returns_correct_tuple(analyzer):
    """Test that _classify_single_validation returns (AnalyzedToolValidation, cluster_label) with correct fields."""
    validation_result = ToolCallValidationResult(
        is_completed=False,
        expected_tool_call=ExpectedToolCall(
            tool="reminder_tool",
            expected_parameters=[
                ToolInputParameter(name="message", check="Should contain reminder text"),
            ],
            expected_output=ToolOutput(check="Should confirm reminder was set"),
        ),
        explanations=[
            'Tool "reminder_tool" call has failed input arguments check or output check.',
            "Tool name: reminder_tool matched.",
            'Argument "message" has failed llm-as-a-judge.',
            "GRADE: I",
        ],
    )

    analyzed, cluster_label = await analyzer._classify_single_validation(
        validation_result=validation_result, data_sample_id=42, agent_run_id=99
    )

    assert isinstance(analyzed, AnalyzedToolValidation)
    assert analyzed.data_sample_id == 42
    assert analyzed.agent_run_id == 99
    assert analyzed.tool_call_validation == validation_result
    assert analyzed.base_error == "\n".join(validation_result.explanations)
    assert cluster_label == "Incorrect Tool Input"


# ==================== Tests for __init__ ====================


def test_init_with_default_config():
    """Test default initialization with no config."""
    analyzer = DeterministicToolCallErrorAnalysis()
    assert analyzer.error_clusters == DEFAULT_TOOL_CALL_ERROR_CLUSTERS
