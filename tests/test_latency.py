import pytest

from agent_inspect.exception import InvalidInputValueError
from agent_inspect.metrics.observed import TotalLatency, AverageLatency
from agent_inspect.models.metrics import TurnTrace, AgentResponse, Step


@pytest.fixture
def mock_turn_trace_1():
    return TurnTrace(
        id="1",
        agent_input="Agent Input",
        steps=[],
        agent_response=AgentResponse(response="Agent Output"),
        latency_in_ms=100.0,
    )


@pytest.fixture
def mock_turn_trace_2():
    return TurnTrace(
        id="2",
        agent_input="Agent Input",
        steps=[],
        agent_response=AgentResponse(response="Agent Output"),
        latency_in_ms=150.0,
    )


@pytest.fixture
def mock_turn_trace_3():
    return TurnTrace(
        id="3",
        agent_input="Agent Input",
        steps=[],
        agent_response=AgentResponse(response="Agent Output"),
        latency_in_ms=0.0,
    )


@pytest.fixture
def mock_turn_trace_4():
    return TurnTrace(
        id="4",
        agent_input="Hello",
        steps=[Step(id="step1", parent_ids=[], agent_thought="Taking user input")],
        agent_response=AgentResponse(response="Hi, how can I help you?"),
    )


@pytest.fixture
def mock_turn_trace_5():
    return TurnTrace(
        id="5",
        agent_input="Hello",
        steps=[Step(id="step1", parent_ids=[], agent_thought="Taking user input")],
        agent_response=AgentResponse(response="Hi, how can I help you?"),
    )


def test_total_latency_no_error(mock_turn_trace_1, mock_turn_trace_2):
    total_latency_metric = TotalLatency()

    score1 = total_latency_metric.evaluate([mock_turn_trace_1, mock_turn_trace_2])
    assert score1.score == 250.0


def test_total_latency_zero_no_error(mock_turn_trace_3):
    total_latency_metric = TotalLatency()
    score1 = total_latency_metric.evaluate([mock_turn_trace_3])
    assert score1.score == 0.0


def test_total_latency_missing_latency_error(mock_turn_trace_4):
    with pytest.raises(InvalidInputValueError) as exc_info:
        total_latency_metric = TotalLatency()
        total_latency_metric.evaluate([mock_turn_trace_4])
    assert (
        exc_info.value.message
        == "Internal Code: 050008, Error Message: Turn(s): 4 are missing latency values."
    )


def test_total_latency_multiple_missing_latency_list_all_error(
    mock_turn_trace_4, mock_turn_trace_1, mock_turn_trace_5
):
    with pytest.raises(InvalidInputValueError) as exc_info:
        total_latency_metric = TotalLatency()
        total_latency_metric.evaluate([mock_turn_trace_4, mock_turn_trace_1, mock_turn_trace_5])
    assert (
        exc_info.value.message
        == "Internal Code: 050008, Error Message: Turn(s): 4, 5 are missing latency values."
    )


def test_average_latency_no_error(mock_turn_trace_1, mock_turn_trace_2):
    average_latency_metric = AverageLatency()
    score1 = average_latency_metric.evaluate([mock_turn_trace_1, mock_turn_trace_2])
    assert score1.score == 125.0


def test_average_latency_zero_no_error(mock_turn_trace_3):
    average_latency_metric = AverageLatency()
    score1 = average_latency_metric.evaluate([mock_turn_trace_3])
    assert score1.score == 0.0


def test_average_latency_multiple_missing_latency_list_all_error(
    mock_turn_trace_4, mock_turn_trace_1, mock_turn_trace_5
):
    with pytest.raises(InvalidInputValueError) as exc_info:
        average_latency_metric = AverageLatency()
        average_latency_metric.evaluate([mock_turn_trace_4, mock_turn_trace_1, mock_turn_trace_5])
    assert (
        exc_info.value.message
        == "Internal Code: 050008, Error Message: Turn(s): 4, 5 are missing latency values."
    )
