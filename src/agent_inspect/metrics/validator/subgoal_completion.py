from typing import Callable, List, Optional, Dict, Any

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, AGENT_INPUT, TOOL_CALL, \
    AGENT_THOUGHT, AGENT_OUTPUT, TEMPLATE_SUBGOAL, INCLUDE_PROMPT_SENT_TO_LLMJ
from agent_inspect.metrics.scorer.templates import \
    DEFAULT_MODEL_GRADED_FACT_SINGLE_TURN_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL, \
    DEFAULT_MODEL_GRADED_FACT_MULTI_TURN_AT_CURRENT_TURN_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL, \
    DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL, \
    DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL
from agent_inspect.models.metrics.agent_data_sample import SubGoal
from agent_inspect.models.metrics.agent_trace import TurnTrace
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult
from agent_inspect.metrics.utils.metrics_utils import get_config_or_default
from agent_inspect.metrics.utils.subgoal_validators import SubGoalValidator
from agent_inspect.metrics.utils.trace_validators import TraceValidator
from agent_inspect.metrics.validator.validator import Validator


class SubGoalCompletionValidator(Validator):
    """
    Validator based on LLM-as-a-judge to assess whether the agent has completed the specified subgoal. 

    .. math::

        LLM_{judge}(i, g_{i, j}, \\tau_i ) = 1 \ \\mathrm{if \ agent \ accomplish} \ g_{i,j}, \\mathrm{and} \ 0 \ \\mathrm{otherwise}, 

    where :math:`g_{i,j}` is the :math:`j`-th grading note for the task sample :math:`i` and
    :math:`\\tau_i` is the agent trajectory consisting of tool calls, agent responses, and user inputs.

    :param llm_client: the client which allows connection to the LLM-as-a-judge model for evaluation.
    :param config: Default to ``None``. Configuration options:

        - **templates_subgoal**: a user provided LLM-as-a-judge template which will be sent to the judge with user inputs (static setting), subgoal, trajectory, agent responses, user task (dynamic setting), and user utterances (dynamic setting). If this is not provided, the default template for either static or dynamic conversation setting will be used.
        - **num_judge_trials**: the number of LLM-as-a-judge runs. Default to ``5``. A majority vote is used when the number of LLM-as-a-judge runs is set to a value larger than 1.
        - **include_judge_explanation**: a :obj:`~typing.bool` flag to indicate whether the output should also return judge explanations. Default to ``False``.
        - **optimize_judge_trials**: a :obj:`~typing.bool` flag to indicate whether to use optimized judge runs when doing a majority vote. Default to ``False``.
        - **max_retry_judge_trials**: an :obj:`~typing.int` value indicating the maximum number of retry attempts for each judge trial in case of errors related to LLM as a judge. Default to ``5``. This will be ignored if ``optimize_judge_trials`` is set to ``True``.
        - **include_entire_prompt_in_validation_result**: a :obj:`~typing.bool` flag to indicate whether to include the entire prompt sent to LLM-as-a-judge in the SubGoalValidationResult. Use this in the debugging tool UI for display. Default to ``False``.
    """
    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_client, config)


    async def validate(self, turn_traces: List[TurnTrace], sub_goal: SubGoal) -> SubGoalValidationResult:
        """
        Returns the LLM-as-a-judge binary score given the subgoal and the agent's trace up to the current turn in a ``static`` conversation setting.

        :param turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :param sub_goal: a :obj:`~agent_inspect.models.metrics.agent_data_sample.SubGoal` object representing a grading note in the form of a natural language text.
        :return: a :obj:`~agent_inspect.models.metrics.validation_result.SubGoalValidationResult` object containing the judge binary score, the subgoal details, and the judge explanations.

        Example:

        >>> from agent_inspect.metrics.validator import SubGoalCompletionValidator
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>> import asyncio
        >>>
        >>> data_subgoal = load_subgoal(sample_path) # Load subgoal
        >>> agent_turn_traces = load_agent_turn_traces(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> validator = SubGoalCompletionValidator(
        ...     llm_client=client,
        ...     config={INCLUDE_JUDGE_EXPLANATION: True, OPTIMIZE_JUDGE_TRIALS: False}
        ... )
        >>> validator_result = asyncio.run(
        ...     validator.validate(
        ...         turn_traces=agent_turn_traces,
        ...         sub_goal=data_subgoal
        ...     )
        ... )
        >>> print(validator_result.is_completed)
        """

        TraceValidator.validate_turn_traces(turn_traces)
        SubGoalValidator.validate_sub_goal(sub_goal)

        include_judge_explanation = get_config_or_default(config=self.config, config_key=INCLUDE_JUDGE_EXPLANATION,
                                                          default=False)
        include_entire_prompt_in_validation_result = get_config_or_default(config=self.config, config_key=INCLUDE_PROMPT_SENT_TO_LLMJ, default=False)
        # TODO: allow judge custom template
        prompt = self.generate_prompt_from_sub_goal_and_turn_traces(sub_goal, turn_traces)
        majority_voted_score, judge_explanations = await self.get_majority_voted_score_from_judge_responses(prompt=prompt)

        is_completed = True if majority_voted_score == 1 else False
        explanations = []
        if is_completed:
            explanations.append(f"Check: \"{sub_goal.details}\" has passed successfully.")
        else:
            explanations.append(f"Check: \"{sub_goal.details}\" has failed.")
        if include_judge_explanation:
            explanations.extend(judge_explanations)

        if include_entire_prompt_in_validation_result:
            return SubGoalValidationResult(is_completed=is_completed, sub_goal=sub_goal, explanations=explanations, prompt_sent_to_llmj=prompt)
        return SubGoalValidationResult(is_completed=is_completed, sub_goal=sub_goal, explanations=explanations)
    

    async def validate_dynamic(self, turn_traces: List[TurnTrace], sub_goal: SubGoal, user_instruction: str) -> SubGoalValidationResult:
        """
                Returns the LLM-as-a-judge binary score given the subgoal, user task instructions (optional), and the agent's trace up to the current turn in a ``dynamic`` conversation setting.

        :param turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :param sub_goal: a :obj:`~agent_inspect.models.metrics.agent_data_sample.SubGoal` object representing a grading note in the form of a natural language text.
        :param user_instruction: a :obj:`~typing.str` object representing the user task instructions. Provide user task instruction through this variable to use judge template with user summary instruction. Otherwise, set it as empty string to use judge template without any user summary instruction.
        :return: a :obj:`~agent_inspect.models.metrics.validation_result.SubGoalValidationResult` object containing the judge binary score, the subgoal details, and the judge explanations.

        Example:

        >>> from agent_inspect.metrics.validator import SubGoalCompletionValidator
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>> import asyncio
        >>>
        >>> data_subgoal, user_instruct = load_subgoal_user_instruct(sample_path) # Load subgoal and user instructions
        >>> agent_turn_traces = load_agent_turn_traces(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> validator = SubGoalCompletionValidator(
        ...     llm_client=client,
        ...     config={INCLUDE_JUDGE_EXPLANATION: True, OPTIMIZE_JUDGE_TRIALS: False}
        ... )
        >>> validator_result = asyncio.run(
        ...     validator.validate_dynamic(
        ...         turn_traces=agent_turn_traces,
        ...         sub_goal=data_subgoal,
        ...         user_instruction=user_instruct
        ...     )
        ... )
        >>> print(validator_result.is_completed)
        """

        TraceValidator.validate_turn_traces(turn_traces)
        SubGoalValidator.validate_sub_goal(sub_goal)

        include_user_instruction = user_instruction and user_instruction.strip() != ""

        include_judge_explanation = get_config_or_default(config=self.config, config_key=INCLUDE_JUDGE_EXPLANATION,
                                                          default=False)
        template_subgoal = get_config_or_default(
            config=self.config, config_key=TEMPLATE_SUBGOAL,
            default=DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL if include_user_instruction else DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL)
        include_entire_prompt_in_validation_result = get_config_or_default(config=self.config,
                                                                           config_key=INCLUDE_PROMPT_SENT_TO_LLMJ, default=False)
        if include_user_instruction:
            prompt = self.generate_prompt_from_sub_goal_user_task_and_turn_traces(sub_goal, user_instruction, turn_traces, template_subgoal)
        else:
            prompt = self.generate_prompt_from_sub_goal_without_user_task_and_turn_traces(sub_goal, turn_traces, template_subgoal)

        majority_voted_score, judge_explanations = await self.get_majority_voted_score_from_judge_responses(
            prompt=prompt
        )

        is_completed = True if majority_voted_score == 1 else False
        explanations = []
        if is_completed:
            explanations.append(f"Check: \"{sub_goal.details}\" has passed successfully.")
        else:
            explanations.append(f"Check: \"{sub_goal.details}\" has failed.")
        if include_judge_explanation:
            explanations.extend(judge_explanations)
        if include_entire_prompt_in_validation_result:
            return SubGoalValidationResult(is_completed=is_completed, sub_goal=sub_goal, explanations=explanations, prompt_sent_to_llmj=prompt)
        return SubGoalValidationResult(is_completed=is_completed, sub_goal=sub_goal, explanations=explanations)
    

    @staticmethod
    def get_initial_traj_str_with_turn(i: int, start_turn: int):
        return f"Turn {i + 1 + start_turn}:\n\n"

    @staticmethod
    def _build_step_trajectories(steps):
        """Helper method to build trajectory items from steps."""
        step_trajectories = []

        if not steps:
            return step_trajectories

        for step in steps:
            if step.tool:
                tool_input_args = {}
                for tool_input_parameter in step.tool_input_args:
                    tool_input_args[tool_input_parameter.name] = tool_input_parameter.value
                step_trajectories.append(
                    {
                        "id": step.id,
                        "parent_ids": step.parent_ids,
                        "type": TOOL_CALL,
                        "content": {
                            "tool_name": step.tool,
                            "tool_arguments": tool_input_args,
                            "tool_output": step.tool_output
                        }
                    }
                )
            if step.agent_thought:
                step_trajectories.append(
                    {
                        "id": step.id,
                        "parent_ids": step.parent_ids,
                        "type": AGENT_THOUGHT,
                        "content": {
                            "agent_thought": step.agent_thought
                        }
                    }
                )
        return step_trajectories

    @staticmethod
    def get_initial_input_response_str_with_turn(i: int, start_turn: int):
        return f"Turn {i + 1 + start_turn}: "

    @staticmethod
    def get_initial_str_without_turn(i: int, start_turn: int):
        return ""
    
    @staticmethod
    def get_trajectories_str_from_agent_trace(agent_trace_turns: list[TurnTrace], start_turn: int = 0, get_initial_traj_str_fn: Callable[[int, int], str] = get_initial_traj_str_with_turn):
        """
        Turn 1:
        {"type": "Agent Input", "content": {"agent_input": "This is an agent input"}}
        {"id": 1, "parent_id": None, "type": "Agent Thought", "content : {"agent_thought": "this is an agent thought"}"}
        {"id": 222, "parent_id": [1], "type": "Tool Call", "content": {"tool_name": "Tool Name", "tool_arguments": {"arg1": "value1", "arg2", "value2"}, "tool_output": "this is a tool output"}}
        {"type": "Agent Output": "content": {"agent_output": "This is an agent output"}}
        """
        trajectories_str = ""

        for i, turn in enumerate(agent_trace_turns):
            trajectories_str += get_initial_traj_str_fn(i, start_turn)
            trajectories_json = [{
                "type": AGENT_INPUT,
                "content": {"agent_input": turn.agent_input}
            }]

            if turn.steps:
                trajectories_json.extend(SubGoalCompletionValidator._build_step_trajectories(turn.steps))

            if turn.agent_response:
                trajectories_json.append({
                    "type": AGENT_OUTPUT,
                    "content": {"agent_output": turn.agent_response.response}
                })
            
            for item in trajectories_json:
                trajectories_str += str(item) + "\n"
        return trajectories_str


    @staticmethod
    def get_agent_input_str(agent_trace_turns: list[TurnTrace], start_turn: int = 0, get_initial_input_str_fn: Callable[[int, int], str] = get_initial_input_response_str_with_turn):
        """
        Turn 1: This is an input for turn 1.
        Turn 2: This is an input for turn 2.
        Turn 3: This is an input for turn 3.
        """
        agent_input_str = ""
        for i, turn in enumerate(agent_trace_turns):
            agent_input_str += get_initial_input_str_fn(i, start_turn) + turn.agent_input + "\n"
        return agent_input_str


    @staticmethod
    def get_agent_responses_str(agent_trace_turns: list[TurnTrace], start_turn: int = 0, get_initial_response_str_fn: Callable[[int, int], str] = get_initial_input_response_str_with_turn):
        """
        Turn 1: This is a response for turn 1.
        Turn 2:
        Turn 3: There is no response from turn 3.
        """
        agent_responses_str = ""
        for i, turn in enumerate(agent_trace_turns):
            agent_responses_str += get_initial_response_str_fn(i, start_turn) + (
                turn.agent_response.response if turn.agent_response else "") + "\n"
        return agent_responses_str

    @staticmethod
    def get_dialogue_str(agent_trace_turns: list[TurnTrace]):
        """
        UserProxy: This is an agent input for turn 1.
        Agent: This is a response from the agent for turn 1 agent input.
        UserProxy: This is an agent input for turn 2.
        Agent:
        """
        dialog_str = ""
        for turn in agent_trace_turns:
            dialog_str += "UserProxy: " + turn.agent_input + "\n"
            dialog_str += "Agent: " + (turn.agent_response.response if turn.agent_response else "") + "\n"
        return dialog_str

    @staticmethod
    def generate_prompt_from_sub_goal_and_turn_traces(sub_goal: SubGoal, turn_traces: List[TurnTrace]):
        current_idx = len(turn_traces) - 1
        get_initial_traj_str_fn = SubGoalCompletionValidator.get_initial_str_without_turn if current_idx == 0 else SubGoalCompletionValidator.get_initial_traj_str_with_turn
        get_initial_response_str_fn = SubGoalCompletionValidator.get_initial_str_without_turn if current_idx == 0 else SubGoalCompletionValidator.get_initial_input_response_str_with_turn
        get_initial_input_str_fn = SubGoalCompletionValidator.get_initial_str_without_turn if current_idx == 0 else SubGoalCompletionValidator.get_initial_input_response_str_with_turn


        current_trajectories_str = SubGoalCompletionValidator.get_trajectories_str_from_agent_trace([turn_traces[-1]], start_turn=current_idx, get_initial_traj_str_fn=get_initial_traj_str_fn)
        current_agent_responses_str = SubGoalCompletionValidator.get_agent_responses_str([turn_traces[-1]], start_turn=current_idx, get_initial_response_str_fn=get_initial_response_str_fn)
        current_agent_input_str = SubGoalCompletionValidator.get_agent_input_str([turn_traces[-1]], start_turn=current_idx, get_initial_input_str_fn=get_initial_input_str_fn)     

        if current_idx == 0:
            template_subgoal = DEFAULT_MODEL_GRADED_FACT_SINGLE_TURN_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL
            prompt = template_subgoal.format(questions=current_agent_input_str,
                                            trajectories=current_trajectories_str,
                                            answers=current_agent_responses_str,
                                            subgoal=sub_goal.details)
        else:
            past_trajectories_str = SubGoalCompletionValidator.get_trajectories_str_from_agent_trace(turn_traces[:-1])
            past_agent_responses_str = SubGoalCompletionValidator.get_agent_responses_str(turn_traces[:-1])
            past_agent_input_str = SubGoalCompletionValidator.get_agent_input_str(turn_traces[:-1])
            template_subgoal = DEFAULT_MODEL_GRADED_FACT_MULTI_TURN_AT_CURRENT_TURN_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL
            prompt = template_subgoal.format(past_user_inputs=past_agent_input_str,
                                            past_agent_trajectories=past_trajectories_str,
                                            past_agent_responses=past_agent_responses_str,
                                            questions=current_agent_input_str,
                                            trajectories=current_trajectories_str,
                                            answers=current_agent_responses_str,
                                            subgoal=sub_goal.details)
            
        return prompt

    @staticmethod
    def generate_prompt_from_sub_goal_user_task_and_turn_traces(sub_goal: SubGoal, user_instruction: str, turn_traces: List[TurnTrace], template_subgoal: str):
        trajectories_str = SubGoalCompletionValidator.get_trajectories_str_from_agent_trace(turn_traces)
        agent_responses_str = SubGoalCompletionValidator.get_agent_responses_str(turn_traces)
        dialog_str = SubGoalCompletionValidator.get_dialogue_str(turn_traces)
        prompt = template_subgoal.format(userTask=user_instruction, subgoal=sub_goal.details,
                                            trajectories=trajectories_str, answers=agent_responses_str,
                                            dynamicDialogue=dialog_str)
        return prompt

    @staticmethod
    def generate_prompt_from_sub_goal_without_user_task_and_turn_traces(sub_goal: SubGoal, turn_traces: List[TurnTrace], template_subgoal: str):
        trajectories_str = SubGoalCompletionValidator.get_trajectories_str_from_agent_trace(turn_traces)
        agent_responses_str = SubGoalCompletionValidator.get_agent_responses_str(turn_traces)
        dialog_str = SubGoalCompletionValidator.get_dialogue_str(turn_traces)
        prompt = template_subgoal.format(subgoal=sub_goal.details,
                                            trajectories=trajectories_str, answers=agent_responses_str,
                                            dynamicDialogue=dialog_str)
        return prompt
