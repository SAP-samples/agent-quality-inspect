from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.models.metrics.agent_data_sample import SubGoal


class SubGoalValidator:
    """
    A utility class for validating subgoals in agent turn traces.
    """

    @staticmethod
    def validate_sub_goal(subgoal: SubGoal) -> None:
        if not subgoal.details:
            raise InvalidInputValueError(internal_code=ErrorCode.MISSING_VALUE.value, message="One of the SubGoals is missing details for judge to evaluate.")
