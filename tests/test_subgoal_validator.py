import pytest

from agent_inspect.models.metrics import SubGoal
from agent_inspect.metrics.utils.subgoal_validators import SubGoalValidator


@pytest.fixture
def mock_subgoal_valid():
    return SubGoal(details="This is the details for a valid subgoal.")


@pytest.fixture
def mock_subgoal_invalid():
    return SubGoal(details="")


def test_subgoal_valid(mock_subgoal_valid):
    SubGoalValidator.validate_sub_goal(mock_subgoal_valid)


def test_subgoal_invalid(mock_subgoal_invalid):
    with pytest.raises(
        ValueError,
        match="One of the SubGoals is missing details for judge to evaluate.",
    ):
        SubGoalValidator.validate_sub_goal(mock_subgoal_invalid)
