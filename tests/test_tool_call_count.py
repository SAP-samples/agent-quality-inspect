import pytest

from agent_inspect.models.metrics import (
    TurnTrace,
    ToolInputParameter,
    AgentResponse,
    Step,
)
from agent_inspect.metrics.observed import ToolCallCount


@pytest.fixture
def mock_turn_trace_1():
    return TurnTrace(
        id="1",
        agent_input="Hello",
        steps=[],
        agent_response=AgentResponse(response="Hi, how can I help you?"),
    )


@pytest.fixture
def mock_turn_trace_2():
    return TurnTrace(
        id="1",
        agent_input="Agent Input",
        steps=[
            Step(
                id="step1",
                parent_ids=[],
                tool="ToolA",
                tool_input_args=[ToolInputParameter(name="param1", value="value1")],
                tool_output="Tool Output",
            )
        ],
        agent_response=AgentResponse(response="Agent Output"),
    )


@pytest.fixture
def mock_turn_trace_3():
    return TurnTrace(
        id="2",
        agent_input="what is 1 + 3?",
        steps=[
            Step(
                id="step1",
                parent_ids=[],
                agent_thought="Which tool can I use to calculate this?",
            ),
            Step(
                id="step2",
                parent_ids=["step1"],
                tool="Calculator",
                tool_input_args=[
                    ToolInputParameter(name="first_value", value="1"),
                    ToolInputParameter(name="second_value", value="3"),
                ],
                tool_output="4",
            ),
            Step(
                id="step3",
                parent_ids=["step2"],
                tool="    ",
                tool_input_args=[ToolInputParameter(name="value", value="4")],
            ),
        ],
        agent_response=AgentResponse(response="The answer is 4."),
    )


def test_tool_call_count_1_tool_count(mock_turn_trace_1, mock_turn_trace_2):
    metric = ToolCallCount()
    score1 = metric.evaluate([mock_turn_trace_1, mock_turn_trace_2])
    assert score1.score == 1


def test_tool_call_count_multiple_tools(mock_turn_trace_2, mock_turn_trace_3):
    metric = ToolCallCount()
    score2 = metric.evaluate([mock_turn_trace_2, mock_turn_trace_3])
    assert score2.score == 2


def test_tool_call_count_no_tools(mock_turn_trace_1):
    metric = ToolCallCount()
    score3 = metric.evaluate([mock_turn_trace_1])
    assert score3.score == 0
