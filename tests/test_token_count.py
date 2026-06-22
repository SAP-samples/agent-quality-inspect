import pytest

from agent_inspect.metrics.observed import (
    InputTotalTokenCount,
    OutputTotalTokenCount,
    ReasoningTotalTokenCount,
    TotalTokenConsumption,
)
from agent_inspect.models.metrics import TurnTrace, AgentResponse, Step


@pytest.fixture
def mock_turn_trace_1():
    return TurnTrace(
        id="1",
        agent_input="Hello",
        steps=[
            Step(
                id="step1",
                parent_ids=[],
                agent_thought="Taking user input",
                input_token_consumption=14,
                output_token_consumption=20,
                reasoning_token_consumption=30,
            ),
            Step(
                id="step2",
                parent_ids=["step1"],
                agent_thought="Formulating response",
                input_token_consumption=14,
                output_token_consumption=20,
                reasoning_token_consumption=30,
            ),
            Step(
                id="step3",
                parent_ids=["step2"],
                agent_thought="Finalizing output",
                input_token_consumption=16,
                output_token_consumption=28,
                reasoning_token_consumption=45,
            ),
        ],
        agent_response=AgentResponse(response="Hi, how can I help you?"),
    )


@pytest.fixture
def mock_turn_trace_2():
    return TurnTrace(
        id="2",
        agent_input="Hello",
        steps=[
            Step(
                id="step1",
                parent_ids=[],
                agent_thought="Taking user input",
                input_token_consumption=0,
                output_token_consumption=0,
                reasoning_token_consumption=0,
            )
        ],
        agent_response=AgentResponse(response="Hi, how can I help you?"),
    )


@pytest.fixture
def mock_turn_trace_3():
    return TurnTrace(
        id="3",
        agent_input="Agent Input",
        steps=[
            Step(
                id="step1",
                parent_ids=[],
                agent_thought="Calculating the sum",
                output_token_consumption=20,
                reasoning_token_consumption=30,
            ),
            Step(
                id="step2",
                parent_ids=[],
                agent_thought="Calculating the sum",
                input_token_consumption=14,
                reasoning_token_consumption=30,
            ),
            Step(
                id="step3",
                parent_ids=[],
                agent_thought="Calculating the sum",
                input_token_consumption=14,
                output_token_consumption=20,
            ),
        ],
        agent_response=AgentResponse(response="Agent Output"),
    )


@pytest.fixture
def mock_turn_trace_4():
    return TurnTrace(
        id="4",
        agent_input="Agent Input",
        steps=[],
        agent_response=AgentResponse(response="Agent Output"),
    )


def test_input_total_token_count_1_step_no_error(mock_turn_trace_1):
    input_token_count_metric = InputTotalTokenCount()
    count = input_token_count_metric.evaluate([mock_turn_trace_1])
    assert count.score == 44


def test_output_total_token_count_1_step_no_error(mock_turn_trace_1):
    output_token_count_metric = OutputTotalTokenCount()
    count = output_token_count_metric.evaluate([mock_turn_trace_1])
    assert count.score == 68


def test_reasoning_total_token_count_1_step_no_error(mock_turn_trace_1):
    reasoning_token_count_metric = ReasoningTotalTokenCount()
    count = reasoning_token_count_metric.evaluate([mock_turn_trace_1])
    assert count.score == 105


def test_zero_input_token_count_evaluate_no_error(mock_turn_trace_2):
    input_token_count_metric = InputTotalTokenCount()
    count = input_token_count_metric.evaluate([mock_turn_trace_2])
    assert count.score == 0


def test_zero_output_token_count_evaluate_no_error(mock_turn_trace_2):
    output_token_count_metric = OutputTotalTokenCount()
    count = output_token_count_metric.evaluate([mock_turn_trace_2])
    assert count.score == 0


def test_zero_reasoning_token_count_evaluate_no_error(mock_turn_trace_2):
    reasoning_token_count_metric = ReasoningTotalTokenCount()
    count = reasoning_token_count_metric.evaluate([mock_turn_trace_2])
    assert count.score == 0


def test_missing_input_tokens_in_step_no_error(mock_turn_trace_3):
    input_token_count_metric = InputTotalTokenCount()
    count = input_token_count_metric.evaluate([mock_turn_trace_3])
    assert count.score == 28


def test_missing_output_tokens_in_step_no_error(mock_turn_trace_3):
    output_token_count_metric = OutputTotalTokenCount()
    count = output_token_count_metric.evaluate([mock_turn_trace_3])
    assert count.score == 40


def test_missing_reasoning_tokens_in_step_no_error(mock_turn_trace_3):
    reasoning_token_count_metric = ReasoningTotalTokenCount()
    count = reasoning_token_count_metric.evaluate([mock_turn_trace_3])
    assert count.score == 60


def test_input_total_multiple_steps_no_error(mock_turn_trace_1, mock_turn_trace_3):
    input_token_count_metric = InputTotalTokenCount()
    count = input_token_count_metric.evaluate([mock_turn_trace_1, mock_turn_trace_3])
    assert count.score == 72


def test_output_total_multiple_steps_no_error(mock_turn_trace_1, mock_turn_trace_3):
    output_token_count_metric = OutputTotalTokenCount()
    count = output_token_count_metric.evaluate([mock_turn_trace_1, mock_turn_trace_3])
    assert count.score == 108


def test_reasoning_total_multiple_steps_no_error(mock_turn_trace_1, mock_turn_trace_3):
    reasoning_token_count_metric = ReasoningTotalTokenCount()
    count = reasoning_token_count_metric.evaluate([mock_turn_trace_1, mock_turn_trace_3])
    assert count.score == 165


def test_import_total_token_count_no_steps(mock_turn_trace_4):
    total_output_latency_metric = OutputTotalTokenCount()
    count = total_output_latency_metric.evaluate([mock_turn_trace_4])
    assert count.score == 0


def test_total_token_count_no_error(mock_turn_trace_1, mock_turn_trace_3):
    total_token_count_metric = TotalTokenConsumption()
    total_token_count = total_token_count_metric.evaluate([mock_turn_trace_1, mock_turn_trace_3])
    assert total_token_count.score == 345
