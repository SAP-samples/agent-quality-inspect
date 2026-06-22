import pytest

from unittest.mock import patch, AsyncMock, MagicMock

from agent_inspect.metrics.constants import INCLUDE_VALIDATION_RESULTS
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.metrics.scorer import ProgressScore, ProgressScoresThroughTurns
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


def test_progress_score_evaluate_fully_completed(
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

    mock_llm_client = AsyncMock()

    progress_score_metric = ProgressScore(llm_client=mock_llm_client)
    with patch("agent_inspect.metrics.scorer.progress.SubGoalCompletionValidator") as MockValidator:
        mock_validator_instance = MockValidator.return_value

        mock_validation_result = SubGoalValidationResult(
            sub_goal=mock_sub_goal_1,
            is_completed=True,
            explanations=["Check: This is a dummy check for subgoal has passed successfully."],
        )

        mock_validator_instance.validate = AsyncMock(return_value=mock_validation_result)
        progress_score = progress_score_metric.evaluate(
            mock_agent_dialogue_trace, mock_evaluation_data_sample
        )
        assert progress_score.score == 1.0
        assert mock_validator_instance.validate.await_count == 2
        assert len(progress_score.explanations) == 2


def test_progress_score_evaluate_partially_completed(
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

    mock_llm_client = AsyncMock()

    progress_score_metric = ProgressScore(llm_client=mock_llm_client)
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
            explanations=["Achieve: This is a dummy achieve for subgoal 2 has failed."],
        )

        mock_validator_instance.validate = AsyncMock(
            side_effect=[mock_validation_result_1, mock_validation_result_2]
        )

        progress_score = progress_score_metric.evaluate(
            mock_agent_dialogue_trace, mock_evaluation_data_sample
        )
        assert progress_score.score == 0.5
        assert mock_validator_instance.validate.await_count == 2
        assert len(progress_score.explanations) == 2
        assert "This is a dummy check for subgoal 1" in progress_score.explanations[0]
        assert "This is a dummy achieve for subgoal 2" in progress_score.explanations[1]


def test_progress_through_turns_fully_completed(
    mock_turn_trace_1,
    mock_turn_trace_2,
    mock_turn_trace_3,
    mock_sub_goal_1,
    mock_sub_goal_2,
):
    mock_agent_dialogue_trace = AgentDialogueTrace(
        turns=[mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3]
    )
    mock_evaluation_data_sample = EvaluationSample(
        sub_goals=[mock_sub_goal_1, mock_sub_goal_2],
        user_instruction="This is a dummy user instruction for testing purposes.",
    )

    mock_llm_client = AsyncMock()
    progress_score_metric = ProgressScoresThroughTurns(llm_client=mock_llm_client)

    with patch("agent_inspect.metrics.scorer.progress.SubGoalCompletionValidator") as MockValidator:
        mock_validator_instance = MockValidator.return_value

        mock_validation_result = SubGoalValidationResult(
            sub_goal=mock_sub_goal_1,
            is_completed=True,
            explanations=["Check: This is a dummy check for subgoal has passed successfully."],
        )

        mock_validator_instance.validate_dynamic = AsyncMock(return_value=mock_validation_result)
        progress_scores = progress_score_metric.evaluate(
            mock_agent_dialogue_trace, mock_evaluation_data_sample
        )
        assert len(progress_scores) == 20
        assert progress_scores[0].score == 1


def test_progress_through_turns_partially_completed(
    mock_turn_trace_1,
    mock_turn_trace_2,
    mock_turn_trace_3,
    mock_sub_goal_1,
    mock_sub_goal_2,
):
    mock_agent_dialogue_trace = AgentDialogueTrace(
        turns=[mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3]
    )
    mock_evaluation_data_sample = EvaluationSample(
        sub_goals=[mock_sub_goal_1, mock_sub_goal_2],
        user_instruction="This is a dummy user instruction for testing purposes.",
    )

    mock_llm_client = AsyncMock()
    progress_score_metric = ProgressScoresThroughTurns(llm_client=mock_llm_client)

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
        progress_scores = progress_score_metric.evaluate(
            mock_agent_dialogue_trace, mock_evaluation_data_sample
        )
        assert len(progress_scores) == 20
        assert progress_scores[0].score == 0.5
        assert progress_scores[-1].score == 1.0


def test_progress_through_turns_no_subgoals_include_validation_results(
    mock_turn_trace_1,
    mock_turn_trace_2,
    mock_turn_trace_3,
    mock_sub_goal_1,
    mock_sub_goal_2,
):
    mock_agent_dialogue_trace = AgentDialogueTrace(
        turns=[mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3]
    )
    mock_evaluation_data_sample = EvaluationSample(
        sub_goals=[mock_sub_goal_1, mock_sub_goal_2],
        user_instruction="This is a dummy user instruction for testing purposes.",
    )

    mock_llm_client = AsyncMock()
    progress_score_metric = ProgressScoresThroughTurns(
        llm_client=mock_llm_client, config={INCLUDE_VALIDATION_RESULTS: True}
    )

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
        progress_scores = progress_score_metric.evaluate(
            mock_agent_dialogue_trace, mock_evaluation_data_sample
        )
        assert len(progress_scores) == 20
        assert progress_scores[0].score == 0.5
        assert progress_scores[-1].score == 1.0
        assert progress_scores[-1].explanations is not None
        assert len(progress_scores[-1].explanations) == 2


def test_progress_through_turns_no_subgoals(
    mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3
):
    mock_agent_dialogue_trace = AgentDialogueTrace(
        turns=[mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3]
    )
    mock_evaluation_data_sample = EvaluationSample(
        sub_goals=[],
        user_instruction="This is a dummy user instruction for testing purposes.",
    )

    mock_llm_client = MagicMock()
    progress_score_metric = ProgressScoresThroughTurns(llm_client=mock_llm_client)

    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050004, Error Message: No validation result present to aggregate for progress score.",
    ):
        progress_score_metric.evaluate(mock_agent_dialogue_trace, mock_evaluation_data_sample)


def test_progress_score_no_subgoals(mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3):
    mock_agent_dialogue_trace = AgentDialogueTrace(
        turns=[mock_turn_trace_1, mock_turn_trace_2, mock_turn_trace_3]
    )
    mock_evaluation_data_sample = EvaluationSample(sub_goals=[])

    mock_llm_client = MagicMock()

    progress_score_metric = ProgressScore(llm_client=mock_llm_client)
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050004, Error Message: No validation result present to aggregate for progress score.",
    ):
        progress_score_metric.evaluate(mock_agent_dialogue_trace, mock_evaluation_data_sample)
