import pytest

from unittest.mock import patch, AsyncMock, MagicMock

from agent_inspect.metrics.constants import MAX_TURNS
from agent_inspect.exception import InvalidInputValueError, EvaluationError
from agent_inspect.models.metrics import (
    ToolInputParameter,
    SubGoal,
    EvaluationSample,
    TurnTrace,
    Step,
    AgentResponse,
    AgentDialogueTrace,
    SubGoalValidationResult,
)
from agent_inspect.metrics.scorer import AUC


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
def mock_sub_goal_1():
    return SubGoal(details="This is a dummy check for subgoal 1", turn=2)


@pytest.fixture
def mock_sub_goal_2():
    return SubGoal(details="This is a dummy achieve for subgoal 2", turn=2)


def test_auc_score_evaluate_fully_completed(
    mock_turn_trace_1,
    mock_turn_trace_2,
    mock_turn_trace_3,
    mock_sub_goal_1,
    mock_sub_goal_2,
):
    mock_agent_dialogue_trace = AgentDialogueTrace(
        turns=[mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3]
    )
    mock_evaluation_data_sample = EvaluationSample(sub_goals=[mock_sub_goal_1, mock_sub_goal_2])

    mock_llm_client = MagicMock()

    auc_metric = AUC(mock_llm_client)

    with patch("agent_inspect.metrics.scorer.progress.SubGoalCompletionValidator") as MockValidator:
        mock_validator_instance = MockValidator.return_value

        mock_validation_result = SubGoalValidationResult(
            sub_goal=mock_sub_goal_1,
            is_completed=True,
            explanations=["Check: This is a dummy check for subgoal has passed successfully."],
        )

        mock_validator_instance.validate_dynamic = AsyncMock(return_value=mock_validation_result)
        auc_score = auc_metric.evaluate(mock_agent_dialogue_trace, mock_evaluation_data_sample)
        assert auc_score.score == 1.0
        assert len(auc_score.explanations) == 20


def test_auc_score_evaluate_partially_completed(
    mock_turn_trace_1,
    mock_turn_trace_2,
    mock_turn_trace_3,
    mock_sub_goal_1,
    mock_sub_goal_2,
):
    mock_agent_dialogue_trace = AgentDialogueTrace(
        turns=[mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3]
    )
    mock_evaluation_data_sample = EvaluationSample(sub_goals=[mock_sub_goal_1, mock_sub_goal_2])

    mock_llm_client = MagicMock()

    auc_metric = AUC(mock_llm_client, config={MAX_TURNS: 10})

    with patch("agent_inspect.metrics.scorer.progress.SubGoalCompletionValidator") as MockValidator:
        mock_validator_instance = MockValidator.return_value

        mock_validation_result_1 = SubGoalValidationResult(
            sub_goal=mock_sub_goal_1,
            is_completed=True,
            explanations=["Check: This is a dummy check for subgoal 1 has passed successfully."],
        )

        mock_validation_result_2 = SubGoalValidationResult(
            sub_goal=mock_sub_goal_2,
            is_completed=False,
            explanations=["Check: This is a dummy achieve for subgoal 2 has failed."],
        )

        mock_validation_result_3 = SubGoalValidationResult(
            sub_goal=mock_sub_goal_2,
            is_completed=True,
            explanations=[
                "Check: This is a dummy check for subgoal 2 has finally passed successfully."
            ],
        )

        mock_validator_instance.validate_dynamic = AsyncMock(
            side_effect=[
                mock_validation_result_1,
                mock_validation_result_2,
                mock_validation_result_3,
            ]
        )
        auc_score = auc_metric.evaluate(mock_agent_dialogue_trace, mock_evaluation_data_sample)
        assert auc_score.score == 0.9722
        assert len(auc_score.explanations) == 10


def test_auc_score_no_progress_scores(
    mock_turn_trace_1,
    mock_turn_trace_2,
    mock_turn_trace_3,
    mock_sub_goal_1,
    mock_sub_goal_2,
):
    mock_agent_dialogue_trace = AgentDialogueTrace(
        turns=[mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3]
    )
    mock_evaluation_data_sample = EvaluationSample(sub_goals=[])

    mock_llm_client = MagicMock()

    progress_score_metric = AUC(llm_client=mock_llm_client)
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050004, Error Message: No validation result present to aggregate for progress score.",
    ):
        progress_score_metric.evaluate(mock_agent_dialogue_trace, mock_evaluation_data_sample)


def test_no_auc_score(mock_sub_goal_1, mock_sub_goal_2):
    with patch(
        "agent_inspect.metrics.scorer.auc.ProgressScoresThroughTurns.evaluate"
    ) as MockProgressScores:
        MockProgressScores.return_value = []

        mock_agent_dialogue_trace = AgentDialogueTrace(turns=[])
        mock_evaluation_data_sample = EvaluationSample(sub_goals=[mock_sub_goal_1, mock_sub_goal_2])

        mock_llm_client = MagicMock()

        auc_metric = AUC(mock_llm_client)

        with pytest.raises(
            EvaluationError,
            match="Internal Code: 050005, Error Message: No progress score produced for AUC calculation.",
        ):
            auc_metric.evaluate(mock_agent_dialogue_trace, mock_evaluation_data_sample)
