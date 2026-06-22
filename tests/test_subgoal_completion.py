import asyncio
from unittest.mock import AsyncMock

import pytest

from agent_inspect.metrics.constants import (
    STATUS_200,
    STATUS_429,
    NUM_JUDGE_TRIALS,
    INCLUDE_JUDGE_EXPLANATION,
    INCLUDE_PROMPT_SENT_TO_LLMJ,
    OPTIMIZE_JUDGE_TRIALS,
    COMPLETE_INCOMPLETE_GRADE_PATTERN,
    COMPLETE_INCOMPLETE_PAIR,
)
from agent_inspect.exception import EvaluationError
from agent_inspect.metrics.validator.templates import (
    DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL,
    DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL,
)
from agent_inspect.metrics.validator import SubGoalCompletionValidator
from agent_inspect.models.metrics import (
    ToolInputParameter,
    SubGoal,
    TurnTrace,
    AgentResponse,
    Step,
)
from agent_inspect.models import LLMResponse


@pytest.fixture
def mock_turn_trace_1():
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
def mock_turn_trace_2():
    return TurnTrace(
        id="1",
        agent_input="Hello",
        steps=[],
        agent_response=AgentResponse(response="Hi, how can I help you?"),
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
                    ToolInputParameter(name="operation", value="+"),
                ],
                tool_output="4",
            ),
        ],
        agent_response=AgentResponse(response="The answer is 4."),
    )


@pytest.fixture
def mock_sub_goal():
    return SubGoal(details="This is a dummy check or subgoal", turn="all")


def test_get_initial_traj_str_with_turn():
    str_results = SubGoalCompletionValidator.get_initial_traj_str_with_turn(0, 2)
    assert str_results == "Turn 3:\n\n"


def test_get_initial_input_response_str_with_turn():
    str_results = SubGoalCompletionValidator.get_initial_input_response_str_with_turn(0, 1)
    assert str_results == "Turn 2: "


def test_get_initial_str_without_turn():
    str_results = SubGoalCompletionValidator.get_initial_str_without_turn(0, 1)
    assert str_results == ""


def test_build_step_trajectories_handles_empty_steps():
    result = SubGoalCompletionValidator._build_step_trajectories([])
    assert result == []


def test_build_step_trajectories_handles_none_steps():
    result = SubGoalCompletionValidator._build_step_trajectories(None)
    assert result == []


def test_build_step_trajectories_handles_tool_call_only(mock_turn_trace_1):
    result = SubGoalCompletionValidator._build_step_trajectories(mock_turn_trace_1.steps)
    assert len(result) == 1
    assert result[0]["id"] == "step1"
    assert result[0]["content"]["tool_name"] == "ToolA"
    assert result[0]["content"]["tool_arguments"] == {"param1": "value1"}
    assert result[0]["content"]["tool_output"] == "Tool Output"


def test_build_step_trajectories_handles_agent_thought_only():
    steps = [Step(id="step1", parent_ids=[], agent_thought="Thinking about the problem")]
    result = SubGoalCompletionValidator._build_step_trajectories(steps)
    assert len(result) == 1
    assert result[0]["type"] == "Agent Thought"
    assert result[0]["id"] == "step1"
    assert result[0]["content"]["agent_thought"] == "Thinking about the problem"


def test_build_step_trajectories_handles_mixed_steps(mock_turn_trace_3):
    result = SubGoalCompletionValidator._build_step_trajectories(mock_turn_trace_3.steps)
    assert len(result) == 2
    assert result[0]["type"] == "Agent Thought"
    assert result[0]["id"] == "step1"
    assert result[1]["type"] == "Tool Call"
    assert result[1]["id"] == "step2"
    assert result[1]["parent_ids"] == ["step1"]


def test_get_trajectories_str_handles_empty_turn_trace():
    result = SubGoalCompletionValidator.get_trajectories_str_from_agent_trace([])
    assert result == ""


def test_get_trajectories_str_formats_single_turn_correctly(mock_turn_trace_1):
    result = SubGoalCompletionValidator.get_trajectories_str_from_agent_trace([mock_turn_trace_1])
    assert (
        result
        == "Turn 1:\n\n{'type': 'Agent Input', 'content': {'agent_input': 'Agent Input'}}\n{'id': 'step1', 'parent_ids': [], 'type': 'Tool Call', 'content': {'tool_name': 'ToolA', 'tool_arguments': {'param1': 'value1'}, 'tool_output': 'Tool Output'}}\n{'type': 'Agent Output', 'content': {'agent_output': 'Agent Output'}}\n"
    )


def test_get_trajectories_str_formats_single_turn_correctly_str_without_turn(
    mock_turn_trace_1,
):
    result = SubGoalCompletionValidator.get_trajectories_str_from_agent_trace(
        [mock_turn_trace_1],
        get_initial_traj_str_fn=SubGoalCompletionValidator.get_initial_str_without_turn,
    )
    assert (
        result
        == "{'type': 'Agent Input', 'content': {'agent_input': 'Agent Input'}}\n{'id': 'step1', 'parent_ids': [], 'type': 'Tool Call', 'content': {'tool_name': 'ToolA', 'tool_arguments': {'param1': 'value1'}, 'tool_output': 'Tool Output'}}\n{'type': 'Agent Output', 'content': {'agent_output': 'Agent Output'}}\n"
    )


def test_get_trajectories_str_formats_multiple_turns_correctly(
    mock_turn_trace_2, mock_turn_trace_3
):
    result = SubGoalCompletionValidator.get_trajectories_str_from_agent_trace(
        [mock_turn_trace_2, mock_turn_trace_3]
    )
    assert (
        result
        == "Turn 1:\n\n{'type': 'Agent Input', 'content': {'agent_input': 'Hello'}}\n{'type': 'Agent Output', 'content': {'agent_output': 'Hi, how can I help you?'}}\nTurn 2:\n\n{'type': 'Agent Input', 'content': {'agent_input': 'what is 1 + 3?'}}\n{'id': 'step1', 'parent_ids': [], 'type': 'Agent Thought', 'content': {'agent_thought': 'Which tool can I use to calculate this?'}}\n{'id': 'step2', 'parent_ids': ['step1'], 'type': 'Tool Call', 'content': {'tool_name': 'Calculator', 'tool_arguments': {'first_value': '1', 'second_value': '3', 'operation': '+'}, 'tool_output': '4'}}\n{'type': 'Agent Output', 'content': {'agent_output': 'The answer is 4.'}}\n"
    )


def test_get_agent_input_str_handles_no_turns():
    result = SubGoalCompletionValidator.get_agent_input_str([])
    assert result == ""


def test_get_agent_input_str_handles_single_turn(mock_turn_trace_2):
    result = SubGoalCompletionValidator.get_agent_input_str([mock_turn_trace_2])
    assert result == "Turn 1: Hello\n"


def test_get_agent_input_str_handles_single_turn_str_without_turn(mock_turn_trace_2):
    result = SubGoalCompletionValidator.get_agent_input_str(
        [mock_turn_trace_2],
        get_initial_input_str_fn=SubGoalCompletionValidator.get_initial_str_without_turn,
    )
    assert result == "Hello\n"


def test_get_agent_input_str_handles_multiple_turns(mock_turn_trace_2, mock_turn_trace_3):
    result = SubGoalCompletionValidator.get_agent_input_str([mock_turn_trace_2, mock_turn_trace_3])
    assert result == "Turn 1: Hello\nTurn 2: what is 1 + 3?\n"


def test_get_agent_response_str_handles_no_turns():
    result = SubGoalCompletionValidator.get_agent_responses_str([])
    assert result == ""


def test_get_agent_response_str_handles_single_turn(mock_turn_trace_2):
    result = SubGoalCompletionValidator.get_agent_responses_str([mock_turn_trace_2])
    assert result == "Turn 1: Hi, how can I help you?\n"


def test_get_agent_response_str_handles_single_turn_str_without_turn(mock_turn_trace_2):
    result = SubGoalCompletionValidator.get_agent_responses_str(
        [mock_turn_trace_2],
        get_initial_response_str_fn=SubGoalCompletionValidator.get_initial_str_without_turn,
    )
    assert result == "Hi, how can I help you?\n"


def test_get_agent_response_str_handles_multiple_turns(mock_turn_trace_2, mock_turn_trace_3):
    result = SubGoalCompletionValidator.get_agent_responses_str(
        [mock_turn_trace_2, mock_turn_trace_3]
    )
    assert result == "Turn 1: Hi, how can I help you?\nTurn 2: The answer is 4.\n"


def test_get_dialogue_str_handles_no_turns():
    result = SubGoalCompletionValidator.get_dialogue_str([])
    assert result == ""


def test_get_dialogue_str_handles_single_turn(mock_turn_trace_2):
    result = SubGoalCompletionValidator.get_dialogue_str([mock_turn_trace_2])
    assert result == "UserProxy: Hello\nAgent: Hi, how can I help you?\n"


def test_get_dialogue_str_handles_multiple_turns(mock_turn_trace_2, mock_turn_trace_3):
    result = SubGoalCompletionValidator.get_dialogue_str([mock_turn_trace_2, mock_turn_trace_3])
    assert (
        result
        == "UserProxy: Hello\nAgent: Hi, how can I help you?\nUserProxy: what is 1 + 3?\nAgent: The answer is 4.\n"
    )


def test_generate_prompt_from_sub_goal_and_turn_traces_single_turn(
    mock_turn_trace_1, mock_sub_goal
):

    prompt = SubGoalCompletionValidator.generate_prompt_from_sub_goal_and_turn_traces(
        sub_goal=mock_sub_goal, turn_traces=[mock_turn_trace_1]
    )
    assert (
        prompt
        == """
You are provided with a sample containing a gold-standard user input [Gold User Input]. Gold Expert Answer is not provided. The actual agent response is provided in section [Agent Response Submission] and the [Agent Intermediate Trajectories] section details the steps taken by the agent.

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Response Submission] to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[Gold User Input]:
Agent Input


************
[Ground Truth Subgoal]:
This is a dummy check or subgoal

************
[Agent Intermediate Trajectories]:
{'type': 'Agent Input', 'content': {'agent_input': 'Agent Input'}}
{'id': 'step1', 'parent_ids': [], 'type': 'Tool Call', 'content': {'tool_name': 'ToolA', 'tool_arguments': {'param1': 'value1'}, 'tool_output': 'Tool Output'}}
{'type': 'Agent Output', 'content': {'agent_output': 'Agent Output'}}


************
[Agent Response Submission]:
Agent Output


************
[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""
    )


def test_generate_prompt_from_sub_goal_and_turn_traces_multiple_turns(
    mock_turn_trace_2, mock_turn_trace_3, mock_sub_goal
):

    prompt = SubGoalCompletionValidator.generate_prompt_from_sub_goal_and_turn_traces(
        sub_goal=mock_sub_goal, turn_traces=[mock_turn_trace_2, mock_turn_trace_3]
    )
    assert (
        prompt
        == """
You are provided with a sample containing a gold-standard user input at the current conversational turn in the [Gold User Input] section which may include question, instruction, or response to the agent. Gold Expert Answer are not provided. The actual agent response at the current conversational turn is provided in section [Agent Response Submission] and the [Agent Intermediate Trajectories] section details the steps taken by the agent for the current conversational turn.

For additional context, user inputs, agent trajectories, and agent responses for all the past conversational turns are also provided in the sections [Past User Inputs], [Past Agent Trajectories], and [Past Agent Responses], respectively.

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved for the CURRENT turn. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Response Submission] that contain information of the CURRENT turn to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[Past User Inputs]:
Turn 1: Hello


************
[Past Agent Trajectories]:
Turn 1:

{'type': 'Agent Input', 'content': {'agent_input': 'Hello'}}
{'type': 'Agent Output', 'content': {'agent_output': 'Hi, how can I help you?'}}


************
[Past Agent Responses]:
Turn 1: Hi, how can I help you?


************
[Gold User Input]:
Turn 2: what is 1 + 3?


************
[Ground Truth Subgoal]:
This is a dummy check or subgoal

************
[Agent Intermediate Trajectories]:
Turn 2:

{'type': 'Agent Input', 'content': {'agent_input': 'what is 1 + 3?'}}
{'id': 'step1', 'parent_ids': [], 'type': 'Agent Thought', 'content': {'agent_thought': 'Which tool can I use to calculate this?'}}
{'id': 'step2', 'parent_ids': ['step1'], 'type': 'Tool Call', 'content': {'tool_name': 'Calculator', 'tool_arguments': {'first_value': '1', 'second_value': '3', 'operation': '+'}, 'tool_output': '4'}}
{'type': 'Agent Output', 'content': {'agent_output': 'The answer is 4.'}}


************
[Agent Response Submission]:
Turn 2: The answer is 4.


************
[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""
    )


def test_validate_subgoal_completion_completed_no_judge_explanation_no_optimization(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 5:
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
            raise AssertionError(f"Expected 5 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_completed_no_judge_explanation_no_optimization_with_retry(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 5:
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
                LLMResponse(status=STATUS_200, completion="", error_message=""),
                LLMResponse(status=STATUS_200, completion="", error_message=""),
                LLMResponse(status=STATUS_200, completion="", error_message=""),
            ]
        elif len(prompts) == 3:
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
                LLMResponse(status=STATUS_200, completion="", error_message=""),
            ]
        elif len(prompts) == 1:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(
                f"Expected 5, 3 or 1 prompt(s) for judge trials. Got {len(prompts)}"
            )

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(llm_client=mock_llm_client)
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    mock_llm_client.make_llm_requests.call_count = 3


def test_validate_subgoal_completion_completed_no_judge_explanation_no_optimization_with_retry_until_failure(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 5:
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
                LLMResponse(status=STATUS_200, completion="", error_message=""),
                LLMResponse(status=STATUS_200, completion="", error_message=""),
                LLMResponse(status=STATUS_200, completion="", error_message=""),
            ]
        elif len(prompts) == 3:
            return [
                LLMResponse(status=STATUS_200, completion="", error_message=""),
                LLMResponse(status=STATUS_200, completion="", error_message=""),
                LLMResponse(status=STATUS_200, completion="", error_message=""),
            ]
        else:
            raise AssertionError(f"Expected 5 or 3 prompt(s) for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(llm_client=mock_llm_client)

    with pytest.raises(
        EvaluationError,
        match="Internal Code: 050000, Error Message: One or more judge trials returned invalid responses after retries.",
    ):
        asyncio.run(validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal))
    assert mock_llm_client.make_llm_requests.await_count == 6


def test_validate_subgoal_completion_completed_no_judge_explanation_optimized(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: True}
    )
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_incomplete_no_judge_explanation_no_optimization(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 5:
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
            raise AssertionError(f"Expected 5 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: False}
    )
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert not validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0] == 'Check: "This is a dummy check or subgoal" has failed.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_incomplete_no_judge_explanation_optimised(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: True}
    )
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert not validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0] == 'Check: "This is a dummy check or subgoal" has failed.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_completed_no_judge_explanation_3_1_1_calls(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()
    toggle = True

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
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                ),
            ]
        elif len(prompts) == 1:
            nonlocal toggle
            if toggle:
                toggle = False
                return [
                    LLMResponse(
                        status=STATUS_200,
                        completion="This is a just dummy judge explanation.\n\nGRADE: I",
                        error_message="",
                    )
                ]
            else:
                return [
                    LLMResponse(
                        status=STATUS_200,
                        completion="This is a just dummy judge explanation.\n\nGRADE: C",
                        error_message="",
                    )
                ]
        else:
            raise AssertionError(f"Expected 3 or 1 prompt(s) for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: True}
    )
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    assert mock_llm_client.make_llm_requests.await_count == 3


def test_validate_subgoal_completion_completed_no_judge_explanation_3_1_1_calls_no_optimised(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 5:
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
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        else:
            raise AssertionError(f"Expected 5 prompt(s) for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: False}
    )
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    assert mock_llm_client.make_llm_requests.await_count == 1


def test_validate_subgoal_completion_incomplete_no_judge_explanation_4_2(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 4:
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
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        elif len(prompts) == 2:
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
            ]
        else:
            raise AssertionError(f"Expected 4 or 2 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={NUM_JUDGE_TRIALS: 7, OPTIMIZE_JUDGE_TRIALS: True},
    )
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert not validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0] == 'Check: "This is a dummy check or subgoal" has failed.'
    )
    assert mock_llm_client.make_llm_requests.await_count == 2


def test_validate_subgoal_completion_incomplete_no_judge_explanation_1_trial(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 1:
            return [
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: I",
                    error_message="",
                )
            ]
        else:
            raise AssertionError(f"Expected 1 prompt for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn
    validator = SubGoalCompletionValidator(llm_client=mock_llm_client, config={NUM_JUDGE_TRIALS: 1})

    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert not validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0] == 'Check: "This is a dummy check or subgoal" has failed.'
    )
    assert mock_llm_client.make_llm_requests.await_count == 1


def test_test_validate_subgoal_completion_incomplete_no_judge_explanation_1_trial_error(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 1:
            return [LLMResponse(status="", completion="", error_message="Some LLM error occurred.")]
        else:
            raise AssertionError(f"Expected 1 prompt for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn
    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={NUM_JUDGE_TRIALS: 1, OPTIMIZE_JUDGE_TRIALS: True},
    )

    with pytest.raises(
        EvaluationError,
        match="Internal Code: 050007, Error Message: Could not reach majority decision due to insufficient valid judge responses.",
    ):
        asyncio.run(validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal))
    assert mock_llm_client.make_llm_requests.await_count == 1


def test_validate_subgoal_completion_completed_no_judge_explanation_7_trial_error(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 4:
            return [
                LLMResponse(
                    status=STATUS_429,
                    completion="",
                    error_message="Some LLM error occurred.",
                ),
                LLMResponse(
                    status=STATUS_429,
                    completion="",
                    error_message="Some LLM error occurred.",
                ),
                LLMResponse(
                    status=STATUS_429,
                    completion="",
                    error_message="Some LLM error occurred.",
                ),
                LLMResponse(
                    status=STATUS_200,
                    completion="This is a just dummy judge explanation.\n\nGRADE: C",
                    error_message="",
                ),
            ]
        elif len(prompts) == 3:
            return [
                LLMResponse(
                    status=STATUS_429,
                    completion="",
                    error_message="Some LLM error occurred.",
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
            raise AssertionError(f"Expected 4 or 3 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={NUM_JUDGE_TRIALS: 7, OPTIMIZE_JUDGE_TRIALS: True},
    )

    with pytest.raises(
        EvaluationError,
        match="Internal Code: 050007, Error Message: Could not reach majority decision due to insufficient valid judge responses.",
    ):
        asyncio.run(validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal))
    assert mock_llm_client.make_llm_requests.await_count == 2


def test_validate_subgoal_completion_error_early_terminate(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 4:
            return [
                LLMResponse(
                    status=STATUS_429,
                    completion="",
                    error_message="Some LLM error occurred.",
                ),
                LLMResponse(
                    status=STATUS_429,
                    completion="",
                    error_message="Some LLM error occurred.",
                ),
                LLMResponse(
                    status=STATUS_429,
                    completion="",
                    error_message="Some LLM error occurred.",
                ),
                LLMResponse(
                    status=STATUS_429,
                    completion="",
                    error_message="Some LLM error occurred.",
                ),
            ]
        else:
            raise AssertionError(f"Expected 4 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={NUM_JUDGE_TRIALS: 7, OPTIMIZE_JUDGE_TRIALS: True},
    )

    with pytest.raises(
        EvaluationError,
        match="Internal Code: 050007, Error Message: Could not reach majority decision due to insufficient valid judge responses.",
    ):
        asyncio.run(validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal))
    assert mock_llm_client.make_llm_requests.await_count == 1


def test_validate_subgoal_completion_completed_with_judge_explanation(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={INCLUDE_JUDGE_EXPLANATION: True, OPTIMIZE_JUDGE_TRIALS: True},
    )
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 4
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    assert (
        validation_result.explanations[-1] == "This is a just dummy judge explanation.\n\nGRADE: C"
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_completed_with_prompt_sent_to_llmj_optimised(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={INCLUDE_PROMPT_SENT_TO_LLMJ: True, OPTIMIZE_JUDGE_TRIALS: True},
    )
    validation_result = asyncio.run(
        validator.validate([mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal)
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert validation_result.prompt_sent_to_llmj is not None
    mock_llm_client.make_llm_requests.assert_called_once()


def test_tally_judge_voting_invalid_judge_response():
    judge_responses = [
        LLMResponse(
            status=STATUS_200,
            completion="Some explanation.\n\nGRADE: X",
            error_message="",
        ),
        LLMResponse(
            status=STATUS_200,
            completion="Some explanation.\n\nGRADE: Y",
            error_message="",
        ),
        LLMResponse(
            status=STATUS_200,
            completion="Some explanation.\n\nGRADE: Z",
            error_message="",
        ),
    ]
    c_cnt, i_cnt, invalid_cnt = SubGoalCompletionValidator._tally_judge_voting(
        0,
        0,
        0,
        judge_responses,
        COMPLETE_INCOMPLETE_GRADE_PATTERN,
        COMPLETE_INCOMPLETE_PAIR,
    )
    assert c_cnt == 0
    assert i_cnt == 0
    assert invalid_cnt == 3


def test_get_majority_voted_score_from_judge_responses():
    llm_client = AsyncMock()
    subgoal_completion_validator = SubGoalCompletionValidator(
        llm_client=llm_client, config={NUM_JUDGE_TRIALS: 4}
    )
    with pytest.raises(
        EvaluationError,
        match="Internal Code: 050000, Error Message: Number of judge trials must be a positive odd integer.",
    ):
        asyncio.run(
            subgoal_completion_validator.get_majority_voted_score_from_judge_responses(prompt="")
        )


def test_validate_subgoal_completion_dynamic_complete_no_judge_explanation_optimised(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()
    mock_user_instruction = "This is a dummy user instruction for testing."

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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: True}
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, mock_user_instruction
        )
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_complete_no_judge_explanation_no_optimisation(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()
    mock_user_instruction = "This is a dummy user instruction for testing."

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 5:
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
            raise AssertionError(f"Expected 5 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: False}
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, mock_user_instruction
        )
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_incomplete_no_judge_explanation_optimised(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()
    mock_user_instruction = "This is a dummy user instruction for testing."

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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: True}
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, mock_user_instruction
        )
    )
    assert not validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0] == 'Check: "This is a dummy check or subgoal" has failed.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_incomplete_no_judge_explanation_no_optimised(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()
    mock_user_instruction = "This is a dummy user instruction for testing."

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 5:
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
            raise AssertionError(f"Expected 5 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: False}
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, mock_user_instruction
        )
    )
    assert not validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0] == 'Check: "This is a dummy check or subgoal" has failed.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_completed_with_judge_explanation(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()
    mock_user_instruction = "This is a dummy user instruction for testing."

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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={INCLUDE_JUDGE_EXPLANATION: True, OPTIMIZE_JUDGE_TRIALS: True},
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, mock_user_instruction
        )
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 4
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    assert (
        validation_result.explanations[-1] == "This is a just dummy judge explanation.\n\nGRADE: C"
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_completed_with_prompt_sent_to_llmj(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()
    mock_user_instruction = "This is a dummy user instruction for testing."

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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={INCLUDE_PROMPT_SENT_TO_LLMJ: True, OPTIMIZE_JUDGE_TRIALS: True},
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, mock_user_instruction
        )
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert validation_result.prompt_sent_to_llmj is not None
    assert (
        validation_result.prompt_sent_to_llmj
        == """
You are provided with a sample that contains several key components centered around an interaction between an agent and a simulated user, referred to as the user proxy. The user proxy represents a human-in-the-loop, engaging with the agent by posing questions and guiding the conversation throughout the dialogue.

The [User Summary Instructions] section outlines the user’s goals, expectations, and the overall task the agent is expected to complete. The [Agent Responses Submission] section captures the agent’s actual responses to the user proxy at each turn of the interaction. The [Agent Intermediate Trajectories] section provides a detailed step-by-step reasoning and actions taken by the agent. Finally, the [Dynamic Dialogue] section presents the full conversation between the agent and the user proxy. 

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Responses Submission] to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[User Summary Instructions]:
This is a dummy user instruction for testing.

************
[Ground Truth Subgoal]:
This is a dummy check or subgoal

************
[Agent Intermediate Trajectories]:
Turn 1:

{'type': 'Agent Input', 'content': {'agent_input': 'Hello'}}
{'type': 'Agent Output', 'content': {'agent_output': 'Hi, how can I help you?'}}
Turn 2:

{'type': 'Agent Input', 'content': {'agent_input': 'what is 1 + 3?'}}
{'id': 'step1', 'parent_ids': [], 'type': 'Agent Thought', 'content': {'agent_thought': 'Which tool can I use to calculate this?'}}
{'id': 'step2', 'parent_ids': ['step1'], 'type': 'Tool Call', 'content': {'tool_name': 'Calculator', 'tool_arguments': {'first_value': '1', 'second_value': '3', 'operation': '+'}, 'tool_output': '4'}}
{'type': 'Agent Output', 'content': {'agent_output': 'The answer is 4.'}}


************
[Agent Responses Submission]:
Turn 1: Hi, how can I help you?
Turn 2: The answer is 4.


************
[Dynamic Dialogue]:
UserProxy: Hello
Agent: Hi, how can I help you?
UserProxy: what is 1 + 3?
Agent: The answer is 4.

[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_generate_prompt_from_sub_goal_user_task_and_turn_traces(
    mock_turn_trace_2, mock_turn_trace_3, mock_sub_goal
):
    mock_user_instruction = "This is a dummy user instruction for testing."
    prompt = SubGoalCompletionValidator.generate_prompt_from_sub_goal_user_task_and_turn_traces(
        sub_goal=mock_sub_goal,
        user_instruction=mock_user_instruction,
        turn_traces=[mock_turn_trace_2, mock_turn_trace_3],
        template_subgoal=DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL,
    )
    assert (
        prompt
        == """
You are provided with a sample that contains several key components centered around an interaction between an agent and a simulated user, referred to as the user proxy. The user proxy represents a human-in-the-loop, engaging with the agent by posing questions and guiding the conversation throughout the dialogue.

The [User Summary Instructions] section outlines the user’s goals, expectations, and the overall task the agent is expected to complete. The [Agent Responses Submission] section captures the agent’s actual responses to the user proxy at each turn of the interaction. The [Agent Intermediate Trajectories] section provides a detailed step-by-step reasoning and actions taken by the agent. Finally, the [Dynamic Dialogue] section presents the full conversation between the agent and the user proxy. 

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Responses Submission] to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[User Summary Instructions]:
This is a dummy user instruction for testing.

************
[Ground Truth Subgoal]:
This is a dummy check or subgoal

************
[Agent Intermediate Trajectories]:
Turn 1:

{'type': 'Agent Input', 'content': {'agent_input': 'Hello'}}
{'type': 'Agent Output', 'content': {'agent_output': 'Hi, how can I help you?'}}
Turn 2:

{'type': 'Agent Input', 'content': {'agent_input': 'what is 1 + 3?'}}
{'id': 'step1', 'parent_ids': [], 'type': 'Agent Thought', 'content': {'agent_thought': 'Which tool can I use to calculate this?'}}
{'id': 'step2', 'parent_ids': ['step1'], 'type': 'Tool Call', 'content': {'tool_name': 'Calculator', 'tool_arguments': {'first_value': '1', 'second_value': '3', 'operation': '+'}, 'tool_output': '4'}}
{'type': 'Agent Output', 'content': {'agent_output': 'The answer is 4.'}}


************
[Agent Responses Submission]:
Turn 1: Hi, how can I help you?
Turn 2: The answer is 4.


************
[Dynamic Dialogue]:
UserProxy: Hello
Agent: Hi, how can I help you?
UserProxy: what is 1 + 3?
Agent: The answer is 4.

[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""
    )


def test_validate_dynamic_complete_no_judge_explanation_optimised(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: True}
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, user_instruction=""
        )
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_dynamic_complete_no_judge_explanation_optimised_with_user_instruction_none(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: True}
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, user_instruction=None
        )
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_without_instruction_complete_no_judge_explanation_no_optimisation(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 5:
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
            raise AssertionError(f"Expected 5 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: False}
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, user_instruction=""
        )
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_without_instruction_incomplete_no_judge_explanation_optimised(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: True}
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, user_instruction=""
        )
    )
    assert not validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0] == 'Check: "This is a dummy check or subgoal" has failed.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_without_instruction_incomplete_no_judge_explanation_no_optimised(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
):
    mock_llm_client = AsyncMock()

    async def mock_make_llm_requests_fn(prompts):
        if len(prompts) == 5:
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
            raise AssertionError(f"Expected 5 prompts for judge trials. Got {len(prompts)}")

    mock_llm_client.make_llm_requests.side_effect = mock_make_llm_requests_fn

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client, config={OPTIMIZE_JUDGE_TRIALS: False}
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, user_instruction=""
        )
    )
    assert not validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 1
    assert (
        validation_result.explanations[0] == 'Check: "This is a dummy check or subgoal" has failed.'
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_without_instruction_completed_with_judge_explanation(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={INCLUDE_JUDGE_EXPLANATION: True, OPTIMIZE_JUDGE_TRIALS: True},
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, user_instruction=""
        )
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert len(validation_result.explanations) == 4
    assert (
        validation_result.explanations[0]
        == 'Check: "This is a dummy check or subgoal" has passed successfully.'
    )
    assert (
        validation_result.explanations[-1] == "This is a just dummy judge explanation.\n\nGRADE: C"
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_validate_subgoal_completion_dynamic_without_instruction_completed_with_prompt_sent_to_llmj(
    mock_sub_goal, mock_turn_trace_2, mock_turn_trace_3
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

    validator = SubGoalCompletionValidator(
        llm_client=mock_llm_client,
        config={INCLUDE_PROMPT_SENT_TO_LLMJ: True, OPTIMIZE_JUDGE_TRIALS: True},
    )
    validation_result = asyncio.run(
        validator.validate_dynamic(
            [mock_turn_trace_2, mock_turn_trace_3], mock_sub_goal, user_instruction=""
        )
    )
    assert validation_result.is_completed
    assert validation_result.sub_goal == mock_sub_goal
    assert validation_result.prompt_sent_to_llmj is not None
    assert (
        validation_result.prompt_sent_to_llmj
        == """
You are provided with a sample that contains several key components centered around an interaction between an agent and a simulated user, referred to as the user proxy. The user proxy represents a human-in-the-loop, engaging with the agent by posing questions and guiding the conversation throughout the dialogue.

The [Agent Responses Submission] section captures the agent’s actual responses to the user proxy at each turn of the interaction. The [Agent Intermediate Trajectories] section provides a detailed step-by-step reasoning and actions taken by the agent. Finally, the [Dynamic Dialogue] section presents the full conversation between the agent and the user proxy. 

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Responses Submission] to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[Ground Truth Subgoal]:
This is a dummy check or subgoal

************
[Agent Intermediate Trajectories]:
Turn 1:

{'type': 'Agent Input', 'content': {'agent_input': 'Hello'}}
{'type': 'Agent Output', 'content': {'agent_output': 'Hi, how can I help you?'}}
Turn 2:

{'type': 'Agent Input', 'content': {'agent_input': 'what is 1 + 3?'}}
{'id': 'step1', 'parent_ids': [], 'type': 'Agent Thought', 'content': {'agent_thought': 'Which tool can I use to calculate this?'}}
{'id': 'step2', 'parent_ids': ['step1'], 'type': 'Tool Call', 'content': {'tool_name': 'Calculator', 'tool_arguments': {'first_value': '1', 'second_value': '3', 'operation': '+'}, 'tool_output': '4'}}
{'type': 'Agent Output', 'content': {'agent_output': 'The answer is 4.'}}


************
[Agent Responses Submission]:
Turn 1: Hi, how can I help you?
Turn 2: The answer is 4.


************
[Dynamic Dialogue]:
UserProxy: Hello
Agent: Hi, how can I help you?
UserProxy: what is 1 + 3?
Agent: The answer is 4.

[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""
    )
    mock_llm_client.make_llm_requests.assert_called_once()


def test_generate_prompt_from_sub_goal_without_user_task_and_turn_traces(
    mock_turn_trace_2, mock_turn_trace_3, mock_sub_goal
):
    prompt = SubGoalCompletionValidator.generate_prompt_from_sub_goal_without_user_task_and_turn_traces(
        sub_goal=mock_sub_goal,
        turn_traces=[mock_turn_trace_2, mock_turn_trace_3],
        template_subgoal=DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL,
    )
    assert (
        prompt
        == """
You are provided with a sample that contains several key components centered around an interaction between an agent and a simulated user, referred to as the user proxy. The user proxy represents a human-in-the-loop, engaging with the agent by posing questions and guiding the conversation throughout the dialogue.

The [Agent Responses Submission] section captures the agent’s actual responses to the user proxy at each turn of the interaction. The [Agent Intermediate Trajectories] section provides a detailed step-by-step reasoning and actions taken by the agent. Finally, the [Dynamic Dialogue] section presents the full conversation between the agent and the user proxy. 

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Responses Submission] to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[Ground Truth Subgoal]:
This is a dummy check or subgoal

************
[Agent Intermediate Trajectories]:
Turn 1:

{'type': 'Agent Input', 'content': {'agent_input': 'Hello'}}
{'type': 'Agent Output', 'content': {'agent_output': 'Hi, how can I help you?'}}
Turn 2:

{'type': 'Agent Input', 'content': {'agent_input': 'what is 1 + 3?'}}
{'id': 'step1', 'parent_ids': [], 'type': 'Agent Thought', 'content': {'agent_thought': 'Which tool can I use to calculate this?'}}
{'id': 'step2', 'parent_ids': ['step1'], 'type': 'Tool Call', 'content': {'tool_name': 'Calculator', 'tool_arguments': {'first_value': '1', 'second_value': '3', 'operation': '+'}, 'tool_output': '4'}}
{'type': 'Agent Output', 'content': {'agent_output': 'The answer is 4.'}}


************
[Agent Responses Submission]:
Turn 1: Hi, how can I help you?
Turn 2: The answer is 4.


************
[Dynamic Dialogue]:
UserProxy: Hello
Agent: Hi, how can I help you?
UserProxy: what is 1 + 3?
Agent: The answer is 4.

[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""
    )
