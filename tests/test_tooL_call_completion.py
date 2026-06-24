import asyncio
from unittest.mock import AsyncMock

import pytest

from agent_inspect.metrics.constants import STATUS_200, INCLUDE_JUDGE_EXPLANATION
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.metrics.validator import ToolCallCompletionValidator
from agent_inspect.models.metrics import (
    ToolInputParameter,
    ExpectedToolCall,
    ToolOutput,
    TurnTrace,
    AgentResponse,
    Step,
    AgentDialogueTrace,
)
from agent_inspect.models import LLMResponse

###############################
# Tests for validate_static_last_turn
###############################


@pytest.fixture
def mock_trace_tool_1():
    tool_input_args = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_2", value="args_value_2"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    step_obj = Step(
        id="custom_tool_trace_1",
        parent_ids=["0"],
        tool="tool_name_1",
        tool_input_args=tool_input_args,
    )
    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_1",
        agent_input="Mock agent input",
        agent_response=AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj],
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace


@pytest.fixture
def mock_trace_tool_1_str_output():
    tool_input_args = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_2", value="args_value_2"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    step_obj = Step(
        id="custom_tool_trace_1",
        parent_ids=["0"],
        tool="tool_name_1",
        tool_input_args=tool_input_args,
        tool_output="350.9",
    )
    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_1",
        agent_input="Mock agent input",
        agent_response=AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj],
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace


@pytest.fixture
def mock_trace_tool_1_float_output():
    tool_input_args = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_2", value="args_value_2"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    step_obj = Step(
        id="custom_tool_trace_1",
        parent_ids=["0"],
        tool="tool_name_1",
        tool_input_args=tool_input_args,
        tool_output=350.9,
    )
    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_1",
        agent_input="Mock agent input",
        agent_response=AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj],
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace


@pytest.fixture
def mock_trace_tool_2_float_output():
    tool_input_args_1 = [
        ToolInputParameter(name="args_name_1", value="args_value_1_incorrect"),
        ToolInputParameter(name="args_name_2", value=400),
    ]

    tool_input_args_2 = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_2", value=20.5),
    ]
    step_obj_1 = Step(
        id="custom_tool_trace_2_step_1",
        parent_ids=["0"],
        tool="tool_name_1",
        tool_input_args=tool_input_args_1,
        tool_output=350.9,
    )
    step_obj_2 = Step(
        id="custom_tool_trace_2_step_2",
        parent_ids=["0"],
        tool="tool_name_1",
        tool_input_args=tool_input_args_2,
        tool_output=350.9,
    )

    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_2",
        agent_input="Mock agent input",
        agent_response=AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj_1, step_obj_2],
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace


@pytest.fixture
def mock_trace_tool_3_float_output():
    tool_input_args_1 = [
        ToolInputParameter(name="args_name_1", value="args_value_1_incorrect"),
        ToolInputParameter(name="args_name_2", value=400),
    ]

    tool_input_args_2 = [
        ToolInputParameter(name="args_name_1", value="args_value_1_incorrect"),
        ToolInputParameter(name="args_name_2", value=20.5),
    ]
    step_obj_1 = Step(
        id="custom_tool_trace_2_step_1",
        parent_ids=["0"],
        tool="tool_name_1",
        tool_input_args=tool_input_args_1,
        tool_output=350.9,
    )
    step_obj_2 = Step(
        id="custom_tool_trace_2_step_2",
        parent_ids=["0"],
        tool="tool_name_1",
        tool_input_args=tool_input_args_2,
        tool_output=400.0,
    )

    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_2",
        agent_input="Mock agent input",
        agent_response=AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj_1, step_obj_2],
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace


@pytest.fixture
def expected_tool_sample_1_judge():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", check="args_value_1"),
        ToolInputParameter(name="args_name_2", check="args_value_2_incorrect"),
    ]
    expected_tool = ExpectedToolCall(
        tool="tool_name_1", expected_parameters=expected_parameters, turn=0
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_1_em():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    expected_tool = ExpectedToolCall(
        tool="tool_name_1", expected_parameters=expected_parameters, turn=0
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_1_mixed():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", check="args_value_1"),
        ToolInputParameter(name="args_name_2", check="args_value_2_incorrect"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    expected_tool = ExpectedToolCall(
        tool="tool_name_1", expected_parameters=expected_parameters, turn=0
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_1_mixed_with_output():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", check="args_value_1"),
        ToolInputParameter(name="args_name_2", check="args_value_2"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    expected_output = ToolOutput(value=350.90)
    expected_tool = ExpectedToolCall(
        tool="tool_name_1",
        expected_parameters=expected_parameters,
        expected_output=expected_output,
        turn=0,
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_1_mixed_with_output_2():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", check="args_value_1"),
        ToolInputParameter(name="args_name_2", check="args_value_2"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    expected_output = ToolOutput(check="350.9")
    expected_tool = ExpectedToolCall(
        tool="tool_name_1",
        expected_parameters=expected_parameters,
        expected_output=expected_output,
        turn=0,
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_1_mixed_with_output_3():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", check="args_value_1"),
        ToolInputParameter(name="args_name_2", check="args_value_2_incorrect"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    expected_output = ToolOutput(check="200.00")
    expected_tool = ExpectedToolCall(
        tool="tool_name_1",
        expected_parameters=expected_parameters,
        expected_output=expected_output,
        turn=0,
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_2_em():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", value="args_value_1_incorrect"),
        ToolInputParameter(name="args_name_3", value=30.0),
    ]
    expected_tool = ExpectedToolCall(
        tool="tool_name_1", expected_parameters=expected_parameters, turn=0
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_3_invalid():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", value="args_value_1", check="args_value_1"),
        ToolInputParameter(name="args_name_3", value=30.0, check="30.0"),
    ]
    expected_tool = ExpectedToolCall(
        tool="tool_name_1", expected_parameters=expected_parameters, turn=0
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_4_invalid():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    expected_output = ToolOutput(value=350.9, check="350.9")
    expected_tool = ExpectedToolCall(
        tool="tool_name_1",
        expected_parameters=expected_parameters,
        expected_output=expected_output,
        turn=0,
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_5():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_5", value=20.5),
    ]
    expected_tool = ExpectedToolCall(
        tool="tool_name_1", expected_parameters=expected_parameters, turn=0
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_6():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    expected_output = ToolOutput(check="100.0")
    expected_tool = ExpectedToolCall(
        tool="tool_name_1",
        expected_parameters=expected_parameters,
        expected_output=expected_output,
        turn=0,
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_7_no_tool_name():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    expected_tool = ExpectedToolCall(
        tool="tool_name_Z", expected_parameters=expected_parameters, turn=0
    )
    return expected_tool


@pytest.fixture
def expected_tool_sample_8_missing_tool_name():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_3", value=20.5),
    ]
    expected_tool = ExpectedToolCall(tool=None, expected_parameters=expected_parameters, turn=0)
    return expected_tool


@pytest.fixture
def expected_tool_sample_9():
    expected_parameters = [
        ToolInputParameter(name="args_name_1", value="args_value_1"),
        ToolInputParameter(name="args_name_2", value=20.5),
    ]
    expected_output = ToolOutput(value=350.9)
    expected_tool = ExpectedToolCall(
        tool="tool_name_1",
        expected_parameters=expected_parameters,
        expected_output=expected_output,
        turn=0,
    )
    return expected_tool


def test_validate_tool_partial_completion_no_judge_explanation_tool_sample_1_judge_trace_1_optimized(
    expected_tool_sample_1_judge, mock_trace_tool_1
):
    mock_llm_client = AsyncMock()
    toggle = True

    async def mock_make_llm_requests_fn(prompts):
        nonlocal toggle
        if len(prompts) == 3 and toggle:
            toggle = False
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        elif len(prompts) == 3 and not toggle:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(f"Expected 3 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    agent_trace = mock_trace_tool_1.turns
    expected_tool = expected_tool_sample_1_judge

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 4
    assert mock_llm_client.make_llm_requests.await_count == 2
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed llm-as-a-judge successfully.'
    )
    assert validation_result.explanations[3] == 'Argument "args_name_2" has failed llm-as-a-judge.'


def test_validate_tool_full_completion_no_judge_explanation_tool_sample_1_em_trace_1(
    expected_tool_sample_1_em, mock_trace_tool_1
):
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_tool_1.turns
    expected_tool = expected_tool_sample_1_em

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 4
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has passed all input arguments check and output check successfully.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed exact match successfully.'
    )
    assert (
        validation_result.explanations[3]
        == 'Argument "args_name_3" has passed exact match successfully.'
    )
    mock_llm_client.make_llm_requests.assert_not_called()


def test_validate_tool_partial_completion_no_judge_explanation_tool_sample_1_mixed_trace_1(
    expected_tool_sample_1_mixed, mock_trace_tool_1
):
    mock_llm_client = AsyncMock()
    toggle = True

    async def mock_make_llm_requests_fn(prompts):
        nonlocal toggle
        if len(prompts) == 3 and toggle:
            toggle = False
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        elif len(prompts) == 3 and not toggle:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(f"Expected 3 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    agent_trace = mock_trace_tool_1.turns
    expected_tool = expected_tool_sample_1_mixed

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 5
    assert mock_llm_client.make_llm_requests.await_count == 2
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed llm-as-a-judge successfully.'
    )
    assert validation_result.explanations[3] == 'Argument "args_name_2" has failed llm-as-a-judge.'
    assert (
        validation_result.explanations[4]
        == 'Argument "args_name_3" has passed exact match successfully.'
    )


def test_validate_tool_full_completion_no_judge_explanation_tool_sample_1_mixed_output_trace_1_float(
    expected_tool_sample_1_mixed_with_output, mock_trace_tool_1_float_output
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 3:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(f"Expected 3 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    agent_trace = mock_trace_tool_1_float_output.turns
    expected_tool = expected_tool_sample_1_mixed_with_output

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 6
    assert mock_llm_client.make_llm_requests.await_count == 2
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has passed all input arguments check and output check successfully.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed llm-as-a-judge successfully.'
    )
    assert (
        validation_result.explanations[3]
        == 'Argument "args_name_2" has passed llm-as-a-judge successfully.'
    )
    assert (
        validation_result.explanations[4]
        == 'Argument "args_name_3" has passed exact match successfully.'
    )
    assert validation_result.explanations[5] == "Tool output has passed exact match successfully."


def test_validate_tool_partial_completion_no_judge_explanation_tool_sample_1_mixed_output_trace_1_str(
    expected_tool_sample_1_mixed_with_output, mock_trace_tool_1_str_output
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 3:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(f"Expected 3 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    agent_trace = mock_trace_tool_1_str_output.turns
    expected_tool = expected_tool_sample_1_mixed_with_output

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 6
    assert mock_llm_client.make_llm_requests.await_count == 2
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed llm-as-a-judge successfully.'
    )
    assert (
        validation_result.explanations[3]
        == 'Argument "args_name_2" has passed llm-as-a-judge successfully.'
    )
    assert (
        validation_result.explanations[4]
        == 'Argument "args_name_3" has passed exact match successfully.'
    )
    assert (
        validation_result.explanations[5]
        == "Tool output has failed exact match. Expected output: 350.9, Expected type: <class 'float'>. Actual output: 350.9, Actual type: <class 'str'>."
    )


def test_validate_tool_partial_completion_no_judge_explanation_tool_sample_1_mixed_output_trace_1(
    expected_tool_sample_1_mixed_with_output, mock_trace_tool_1
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 3:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(f"Expected 3 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    agent_trace = mock_trace_tool_1.turns
    expected_tool = expected_tool_sample_1_mixed_with_output

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 6
    assert mock_llm_client.make_llm_requests.await_count == 2
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed llm-as-a-judge successfully.'
    )
    assert (
        validation_result.explanations[3]
        == 'Argument "args_name_2" has passed llm-as-a-judge successfully.'
    )
    assert (
        validation_result.explanations[4]
        == 'Argument "args_name_3" has passed exact match successfully.'
    )
    assert (
        validation_result.explanations[5]
        == "Tool output has failed exact match. Expected output: 350.9, Expected type: <class 'float'>. Actual output: None, Actual type: <class 'NoneType'>."
    )


def test_validate_tool_full_completion_no_judge_explanation_tool_sample_1_mixed_output_2_trace_1_str(
    expected_tool_sample_1_mixed_with_output_2, mock_trace_tool_1_str_output
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 3:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(f"Expected 3 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    agent_trace = mock_trace_tool_1_str_output.turns
    expected_tool = expected_tool_sample_1_mixed_with_output_2

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 6
    assert mock_llm_client.make_llm_requests.await_count == 3
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has passed all input arguments check and output check successfully.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed llm-as-a-judge successfully.'
    )
    assert (
        validation_result.explanations[3]
        == 'Argument "args_name_2" has passed llm-as-a-judge successfully.'
    )
    assert (
        validation_result.explanations[4]
        == 'Argument "args_name_3" has passed exact match successfully.'
    )
    assert (
        validation_result.explanations[5] == "Tool output has passed llm-as-a-judge successfully."
    )


def test_validate_tool_no_completion_no_judge_explanation_tool_sample_2_trace_1(
    expected_tool_sample_2_em, mock_trace_tool_1
):
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_tool_1.turns
    expected_tool = expected_tool_sample_2_em

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 4
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[2]
        == "Argument \"args_name_1\" has failed exact match. Expected value: args_value_1_incorrect, Expected type: <class 'str'>. Actual value: args_value_1, Actual type: <class 'str'>."
    )
    assert (
        validation_result.explanations[3]
        == "Argument \"args_name_3\" has failed exact match. Expected value: 30.0, Expected type: <class 'float'>. Actual value: 20.5, Actual type: <class 'float'>."
    )
    mock_llm_client.make_llm_requests.assert_not_called()


def test_validate_tool_handles_input_args_value_check_present_tool_sample_3_trace_1(
    expected_tool_sample_3_invalid, mock_trace_tool_1
):
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_tool_1.turns
    expected_tool = expected_tool_sample_3_invalid

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050009, Error Message: ExpectedToolCall parameter args_name_1 cannot have both value and check specified at the same time.",
    ):
        asyncio.run(
            validator.validate_static_last_turn(
                agent_trace_turns=agent_trace, expected_tool_call=expected_tool
            )
        )


def test_validate_tool_handles_output_value_check_present_tool_sample_4_trace_1_str(
    expected_tool_sample_4_invalid, mock_trace_tool_1_str_output
):
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_tool_1_str_output.turns
    expected_tool = expected_tool_sample_4_invalid

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050009, Error Message: ExpectedToolCall output cannot have both value and check specified at the same time.",
    ):
        asyncio.run(
            validator.validate_static_last_turn(
                agent_trace_turns=agent_trace, expected_tool_call=expected_tool
            )
        )


def test_validate_tool_partial_completion_no_judge_explanation_tool_sample_5_trace_1(
    expected_tool_sample_5, mock_trace_tool_1
):
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_tool_1.turns
    expected_tool = expected_tool_sample_5

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 4
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed exact match successfully.'
    )
    assert (
        validation_result.explanations[3]
        == "Argument \"args_name_5\" not even found in actual tool call. Expected value: 20.5, Expected type: <class 'float'>."
    )
    mock_llm_client.make_llm_requests.assert_not_called()


def test_validate_tool_partial_completion_no_judge_explanation_tool_sample_6_trace_1_str(
    expected_tool_sample_6, mock_trace_tool_1_str_output
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 3:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(f"Expected 3 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    agent_trace = mock_trace_tool_1_str_output.turns
    expected_tool = expected_tool_sample_6

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 5
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed exact match successfully.'
    )
    assert (
        validation_result.explanations[3]
        == 'Argument "args_name_3" has passed exact match successfully.'
    )
    assert validation_result.explanations[4] == "Tool output has failed llm-as-a-judge."
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_tool_partial_completion_with_judge_explanation_tool_sample_1_mixed_output_3_trace_1_str(
    expected_tool_sample_1_mixed_with_output_3, mock_trace_tool_1_str_output
):
    mock_llm_client = AsyncMock()
    toggle = True

    async def mock_make_llm_requests_fn(prompts):
        nonlocal toggle
        if len(prompts) == 3 and toggle:
            toggle = False
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge positive explanation 0.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge positive explanation 1.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge positive explanation 2.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        elif len(prompts) == 3 and not toggle:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge negative explanation 0.\n\nGRADE: I",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge negative explanation 1.\n\nGRADE: I",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge negative explanation 2.\n\nGRADE: I",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(f"Expected 3 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    agent_trace = mock_trace_tool_1_str_output.turns
    expected_tool = expected_tool_sample_1_mixed_with_output_3

    validator = ToolCallCompletionValidator(
        llm_client=mock_llm_client, config={INCLUDE_JUDGE_EXPLANATION: True}
    )
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    print(validation_result.explanations)
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 15
    assert mock_llm_client.make_llm_requests.await_count == 3
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed llm-as-a-judge successfully.'
    )
    assert (
        validation_result.explanations[3]
        == "This is a just dummy judge positive explanation 0.\n\nGRADE: C"
    )
    assert (
        validation_result.explanations[4]
        == "This is a just dummy judge positive explanation 1.\n\nGRADE: C"
    )
    assert (
        validation_result.explanations[5]
        == "This is a just dummy judge positive explanation 2.\n\nGRADE: C"
    )
    assert validation_result.explanations[6] == 'Argument "args_name_2" has failed llm-as-a-judge.'
    assert (
        validation_result.explanations[7]
        == validation_result.explanations[12]
        == "This is a just dummy judge negative explanation 0.\n\nGRADE: I"
    )
    assert (
        validation_result.explanations[8]
        == validation_result.explanations[13]
        == "This is a just dummy judge negative explanation 1.\n\nGRADE: I"
    )
    assert (
        validation_result.explanations[9]
        == validation_result.explanations[14]
        == "This is a just dummy judge negative explanation 2.\n\nGRADE: I"
    )
    assert (
        validation_result.explanations[10]
        == 'Argument "args_name_3" has passed exact match successfully.'
    )
    assert validation_result.explanations[11] == "Tool output has failed llm-as-a-judge."


def test_validate_tool_no_tool_name_no_completion_no_judge_explanation_tool_sample_7_trace_1(
    expected_tool_sample_7_no_tool_name, mock_trace_tool_1
):
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_tool_1.turns
    expected_tool = expected_tool_sample_7_no_tool_name

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'No matching tool name "tool_name_Z" is found for expected tool in this agent turn'
    )
    mock_llm_client.make_llm_requests.assert_not_called()


def test_validate_tool_handles_missing_expected_tool_name_tool_sample_8_trace_1(
    expected_tool_sample_8_missing_tool_name, mock_trace_tool_1
):
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_tool_1.turns
    expected_tool = expected_tool_sample_8_missing_tool_name

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050008, Error Message: ExpectedToolCall is missing Tool Name.",
    ):
        asyncio.run(
            validator.validate_static_last_turn(
                agent_trace_turns=agent_trace, expected_tool_call=expected_tool
            )
        )


def test_validate_tool_full_completion_multiple_same_tool_no_judge_explanation_tool_sample_9_trace_2_float(
    expected_tool_sample_9, mock_trace_tool_2_float_output
):
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_tool_2_float_output.turns
    expected_tool = expected_tool_sample_9

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    assert validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 5
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has passed all input arguments check and output check successfully.'
    )
    assert (
        validation_result.explanations[2]
        == 'Argument "args_name_1" has passed exact match successfully.'
    )
    assert (
        validation_result.explanations[3]
        == 'Argument "args_name_2" has passed exact match successfully.'
    )
    assert validation_result.explanations[4] == "Tool output has passed exact match successfully."
    mock_llm_client.make_llm_requests.assert_not_called()


def test_validate_tool_no_completion_multiple_same_tool_no_judge_explanation_tool_sample_9_trace_3_float(
    expected_tool_sample_9, mock_trace_tool_3_float_output
):
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_tool_3_float_output.turns
    expected_tool = expected_tool_sample_9

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_static_last_turn(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )
    print(validation_result.explanations)
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) == 10
    assert (
        validation_result.explanations[0]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[2]
        == "Argument \"args_name_1\" has failed exact match. Expected value: args_value_1, Expected type: <class 'str'>. Actual value: args_value_1_incorrect, Actual type: <class 'str'>."
    )
    assert (
        validation_result.explanations[3]
        == "Argument \"args_name_2\" has failed exact match. Expected value: 20.5, Expected type: <class 'float'>. Actual value: 400, Actual type: <class 'int'>."
    )
    assert validation_result.explanations[4] == "Tool output has passed exact match successfully."
    assert (
        validation_result.explanations[5]
        == 'Tool "tool_name_1" call has failed input arguments check or output check.'
    )
    assert (
        validation_result.explanations[7]
        == "Argument \"args_name_1\" has failed exact match. Expected value: args_value_1, Expected type: <class 'str'>. Actual value: args_value_1_incorrect, Actual type: <class 'str'>."
    )
    assert (
        validation_result.explanations[8]
        == 'Argument "args_name_2" has passed exact match successfully.'
    )
    assert (
        validation_result.explanations[9]
        == "Tool output has failed exact match. Expected output: 350.9, Expected type: <class 'float'>. Actual output: 400.0, Actual type: <class 'float'>."
    )
    mock_llm_client.make_llm_requests.assert_not_called()


###############################
# Tests for validate_multi_turns
###############################


@pytest.fixture
def mock_trace_multiple_turns_with_tool_in_turn_2():
    """Create a trace with 5 turns where toolA is correctly called in turn 2."""
    # Turn 0: No tool calls
    turn0 = TurnTrace(
        id="turn-0",
        agent_input="Initial request",
        steps=[],
        agent_response=AgentResponse(response="Processing request...", status_code=STATUS_200),
    )

    # Turn 1: Wrong tool call
    step_wrong_tool = Step(
        id="s-wrong-tool",
        parent_ids=[None],
        tool="wrong_tool",
        tool_input_args=[ToolInputParameter(name="param1", value="value1")],
        tool_output={"result": "wrong"},
    )
    turn1 = TurnTrace(
        id="turn-1",
        agent_input="Request for toolA",
        steps=[step_wrong_tool],
        agent_response=AgentResponse(response="Called wrong tool", status_code=STATUS_200),
    )

    # Turn 2: Correct toolA call
    step_tool_a = Step(
        id="s-tool-a",
        parent_ids=[None],
        tool="toolA",
        tool_input_args=[
            ToolInputParameter(name="param1", value="value1"),
            ToolInputParameter(name="param2", value="value2"),
        ],
        tool_output={"status": "success", "result": 42},
    )
    turn2 = TurnTrace(
        id="turn-2",
        agent_input="Another request",
        steps=[step_tool_a],
        agent_response=AgentResponse(response="toolA executed", status_code=STATUS_200),
    )

    # Turn 3-4: More turns after the match
    turn3 = TurnTrace(
        id="turn-3",
        agent_input="Follow up",
        steps=[],
        agent_response=AgentResponse(response="Acknowledged", status_code=STATUS_200),
    )
    turn4 = TurnTrace(
        id="turn-4",
        agent_input="Final request",
        steps=[],
        agent_response=AgentResponse(response="Complete", status_code=STATUS_200),
    )

    return AgentDialogueTrace(turns=[turn0, turn1, turn2, turn3, turn4])


@pytest.fixture
def mock_trace_multiple_turns_no_match():
    """Create a trace with 4 turns where toolA is never correctly called."""
    # Turn 0: No tool calls
    turn0 = TurnTrace(
        id="turn-0",
        agent_input="Initial request",
        steps=[],
        agent_response=AgentResponse(response="Processing...", status_code=STATUS_200),
    )

    # Turn 1: Wrong tool
    step_wrong_tool = Step(
        id="s-wrong-tool",
        parent_ids=[None],
        tool="wrong_tool",
        tool_input_args=[ToolInputParameter(name="param1", value="wrong_value")],
        tool_output={"result": "wrong"},
    )
    turn1 = TurnTrace(
        id="turn-1",
        agent_input="Request",
        steps=[step_wrong_tool],
        agent_response=AgentResponse(response="Wrong tool", status_code=STATUS_200),
    )

    # Turn 2: Correct tool name but wrong parameters
    step_tool_a_wrong_params = Step(
        id="s-tool-a-wrong",
        parent_ids=[None],
        tool="toolA",
        tool_input_args=[
            ToolInputParameter(name="param1", value="wrong_value1"),
            ToolInputParameter(name="param2", value="wrong_value2"),
        ],
        tool_output={"status": "success"},
    )
    turn2 = TurnTrace(
        id="turn-2",
        agent_input="Another request",
        steps=[step_tool_a_wrong_params],
        agent_response=AgentResponse(response="Called with wrong params", status_code=STATUS_200),
    )

    # Turn 3: No tool calls
    turn3 = TurnTrace(
        id="turn-3",
        agent_input="Final request",
        steps=[],
        agent_response=AgentResponse(response="No tools", status_code=STATUS_200),
    )

    return AgentDialogueTrace(turns=[turn0, turn1, turn2, turn3])


@pytest.fixture
def expected_tool_a():
    """Expected tool call for toolA with specific parameters."""
    expected_parameters = [
        ToolInputParameter(name="param1", value="value1"),
        ToolInputParameter(name="param2", value="value2"),
    ]
    expected_output = ToolOutput(value={"status": "success", "result": 42})
    return ExpectedToolCall(
        tool="toolA",
        expected_parameters=expected_parameters,
        expected_output=expected_output,
        turn=0,
    )


def test_validate_multi_turns_early_stop_when_tool_found_in_turn_2(
    mock_trace_multiple_turns_with_tool_in_turn_2, expected_tool_a
):
    """Test that validate_multi_turns stops early when a valid tool call is found in turn 2."""
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_multiple_turns_with_tool_in_turn_2.turns
    expected_tool = expected_tool_a

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)

    validation_result = asyncio.run(
        validator.validate_multi_turns(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )

    # Should find tool in turn 2 and stop early (not check turns 3 and 4)
    assert validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) >= 1
    # Should have found toolA successfully
    assert "toolA" in str(validation_result.explanations)
    # Verify the tool parameters and output match turn 2 data
    assert (
        validation_result.explanations[0]
        == 'Tool "toolA" call has passed all input arguments check and output check successfully.'
    )
    assert "Tool name: toolA matched." in validation_result.explanations
    assert (
        'Argument "param1" has passed exact match successfully.' in validation_result.explanations
    )
    assert (
        'Argument "param2" has passed exact match successfully.' in validation_result.explanations
    )
    assert "Tool output has passed exact match successfully." in validation_result.explanations
    mock_llm_client.make_llm_requests.assert_not_called()


def test_validate_multi_turns_no_valid_tool_found_returns_last_result(
    mock_trace_multiple_turns_no_match, expected_tool_a
):
    """Test that validate_multi_turns returns the last result when no valid tool is found in any turn."""
    mock_llm_client = AsyncMock()
    agent_trace = mock_trace_multiple_turns_no_match.turns
    expected_tool = expected_tool_a

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)

    validation_result = asyncio.run(
        validator.validate_multi_turns(
            agent_trace_turns=agent_trace, expected_tool_call=expected_tool
        )
    )

    # Should return failure since no valid tool found in any turn
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) >= 1
    # Should have explanations about failed validation from the last turn checked
    # The last turn (turn 3) has no tool calls, so it should say no matching tool name found
    assert (
        'No matching tool name "toolA" is found for expected tool in this agent turn'
        in validation_result.explanations[0]
    )
    mock_llm_client.make_llm_requests.assert_not_called()


def test_validate_multi_turns_finds_tool_in_first_turn(expected_tool_a):
    """Test that validate_multi_turns returns immediately if tool is found in the first turn."""
    # Create a trace with tool found in turn 0
    step_tool_a = Step(
        id="s-tool-a",
        parent_ids=[None],
        tool="toolA",
        tool_input_args=[
            ToolInputParameter(name="param1", value="value1"),
            ToolInputParameter(name="param2", value="value2"),
        ],
        tool_output={"status": "success", "result": 42},
    )
    turn0 = TurnTrace(
        id="turn-0",
        agent_input="Request toolA",
        steps=[step_tool_a],
        agent_response=AgentResponse(response="toolA executed", status_code=STATUS_200),
    )
    turn1 = TurnTrace(
        id="turn-1",
        agent_input="Follow up",
        steps=[],
        agent_response=AgentResponse(response="Done", status_code=STATUS_200),
    )

    agent_trace = AgentDialogueTrace(turns=[turn0, turn1])

    mock_llm_client = AsyncMock()
    expected_tool = expected_tool_a

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)

    validation_result = asyncio.run(
        validator.validate_multi_turns(
            agent_trace_turns=agent_trace.turns, expected_tool_call=expected_tool
        )
    )

    # Should find tool in first turn and return immediately
    assert validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) >= 1
    # Verify it found the tool in turn 0 with correct parameters
    assert (
        validation_result.explanations[0]
        == 'Tool "toolA" call has passed all input arguments check and output check successfully.'
    )
    assert "Tool name: toolA matched." in validation_result.explanations
    assert (
        'Argument "param1" has passed exact match successfully.' in validation_result.explanations
    )
    assert (
        'Argument "param2" has passed exact match successfully.' in validation_result.explanations
    )
    mock_llm_client.make_llm_requests.assert_not_called()


def test_validate_multi_turns_single_turn_no_match(expected_tool_a):
    """Test validate_multi_turns with a single turn that doesn't match."""
    turn0 = TurnTrace(
        id="turn-0",
        agent_input="Request",
        steps=[],
        agent_response=AgentResponse(response="No tools", status_code=STATUS_200),
    )
    agent_trace = AgentDialogueTrace(turns=[turn0])

    mock_llm_client = AsyncMock()
    expected_tool = expected_tool_a

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate_multi_turns(
            agent_trace_turns=agent_trace.turns, expected_tool_call=expected_tool
        )
    )

    # Should fail since no tool found in the single turn
    assert not validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    assert len(validation_result.explanations) >= 1
    # Should indicate no matching tool was found
    assert (
        'No matching tool name "toolA" is found for expected tool in this agent turn'
        in validation_result.explanations[0]
    )


def test_validate_multi_turns_empty_trace_list(expected_tool_a):
    """Test validate_multi_turns with an empty trace list."""
    agent_trace = []

    mock_llm_client = AsyncMock()
    expected_tool = expected_tool_a

    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)

    with pytest.raises(InvalidInputValueError):
        asyncio.run(
            validator.validate_multi_turns(
                agent_trace_turns=agent_trace, expected_tool_call=expected_tool
            )
        )


def test_validate_multi_turns_finds_tool_in_last_turn():
    """Test that validate_multi_turns correctly finds tool in the last turn."""
    # Create trace with tool only in the last turn
    turn0 = TurnTrace(
        id="turn-0",
        agent_input="Request",
        steps=[],
        agent_response=AgentResponse(response="No tools", status_code=STATUS_200),
    )
    turn1 = TurnTrace(
        id="turn-1",
        agent_input="Another request",
        steps=[],
        agent_response=AgentResponse(response="Still no tools", status_code=STATUS_200),
    )

    # Last turn has the correct tool
    step_tool_a = Step(
        id="s-tool-a",
        parent_ids=[None],
        tool="toolA",
        tool_input_args=[
            ToolInputParameter(name="param1", value="value1"),
            ToolInputParameter(name="param2", value="value2"),
        ],
        tool_output={"status": "success", "result": 42},
    )
    turn2 = TurnTrace(
        id="turn-2",
        agent_input="Final request",
        steps=[step_tool_a],
        agent_response=AgentResponse(response="toolA executed", status_code=STATUS_200),
    )

    agent_trace = AgentDialogueTrace(turns=[turn0, turn1, turn2])

    expected_parameters = [
        ToolInputParameter(name="param1", value="value1"),
        ToolInputParameter(name="param2", value="value2"),
    ]
    expected_output = ToolOutput(value={"status": "success", "result": 42})
    expected_tool = ExpectedToolCall(
        tool="toolA",
        expected_parameters=expected_parameters,
        expected_output=expected_output,
        turn=0,
    )

    mock_llm_client = AsyncMock()
    validator = ToolCallCompletionValidator(llm_client=mock_llm_client)

    validation_result = asyncio.run(
        validator.validate_multi_turns(
            agent_trace_turns=agent_trace.turns, expected_tool_call=expected_tool
        )
    )

    # Should find tool in the last turn (turn 2)
    assert validation_result.is_completed
    assert expected_tool == validation_result.expected_tool_call
    # Verify it found the tool in turn 2 (the last turn) with correct match
    assert (
        validation_result.explanations[0]
        == 'Tool "toolA" call has passed all input arguments check and output check successfully.'
    )
    assert "Tool name: toolA matched." in validation_result.explanations
    assert (
        'Argument "param1" has passed exact match successfully.' in validation_result.explanations
    )
    assert (
        'Argument "param2" has passed exact match successfully.' in validation_result.explanations
    )
    assert "Tool output has passed exact match successfully." in validation_result.explanations
    mock_llm_client.make_llm_requests.assert_not_called()
