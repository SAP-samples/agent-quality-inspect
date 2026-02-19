from typing import List, Optional, Dict, Any

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.metrics.constants import NUM_JUDGE_TRIALS, NUM_JUDGE_TRIALS_DEFAULT, INCLUDE_JUDGE_EXPLANATION
from agent_inspect.metrics.scorer.templates import TOOL_CORRECTNESS_TEMPLATE
from agent_inspect.models.metrics.agent_data_sample import ToolInputParameter, ExpectedToolCall
from agent_inspect.models.metrics.agent_trace import TurnTrace, Step
from agent_inspect.models.metrics.validation_result import ToolCallValidationResult
from agent_inspect.metrics.utils.metrics_utils import get_config_or_default
from agent_inspect.metrics.utils.trace_validators import TraceValidator
from agent_inspect.metrics.utils.expected_tool_validators import ExpectedToolCallValidator
from agent_inspect.metrics.validator.validator import Validator


class ToolCallCompletionValidator(Validator):
    """
    Tool call completion validator using both exact match and LLM-as-a-judge approaches to assess whether the tool call is correctly completed. 
    
    This validator performs a three-dimensional evaluation of tool calls:
    
    1. **Tool name validation**: Checks if the tool name in the agent's trajectory matches the expected tool name exactly.
    2. **Tool input arguments validation**: Validates each input parameter using either exact match (when the expected parameter specifies a :obj:`~agent_inspect.models.metrics.agent_data_sample.ToolInputParameter.value`) or LLM-as-a-judge (when the expected parameter specifies a :obj:`~agent_inspect.models.metrics.agent_data_sample.ToolInputParameter.check`).
    3. **Tool output validation**: Validates the tool's output using either exact match (when the expected output specifies a :obj:`~agent_inspect.models.metrics.agent_data_sample.ToolOutput.value`) or LLM-as-a-judge (when the expected output specifies a :obj:`~agent_inspect.models.metrics.agent_data_sample.ToolOutput.check`).
    
    A tool call is considered correctly completed only if all the above three dimensions pass validation. The validator iterates through all tool calls in the latest turn of the agent trace to find a matching tool call that satisfies all validation criteria set in :obj:`~agent_inspect.models.metrics.agent_data_sample.ExpectedToolCall`.
    
    :param llm_client: the client which allows connection to the LLM-as-a-judge model for evaluation.
    :param config: Default to ``None``. Configuration options:
    
        - **num_judge_trials**: the number of LLM-as-a-judge runs. Default to ``5``. A majority vote is used when the number of LLM-as-a-judge runs is set to a value larger than 1.
        - **include_judge_explanation**: a :obj:`~typing.bool` flag to indicate whether the output should also return judge explanations. Default to ``False``.
    """
    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_client, config)
        
                
    async def validate(self, agent_trace_turns: List[TurnTrace], expected_tool_call: ExpectedToolCall) -> ToolCallValidationResult:
        """
        Returns a :obj:`~agent_inspect.models.metrics.validation_result.ToolCallValidationResult` indicating whether the tool call is correctly completed by the agent in the latest turn of the agent trace.
        
        :param agent_trace_turns: A list of :obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace` representing the entire agent trace up to the current turn.
        :param expected_tool_call: An :obj:`~agent_inspect.models.metrics.agent_data_sample.ExpectedToolCall` representing the expected tool call checklist to validate against.
        :return: a :obj:`~agent_inspect.models.metrics.validation_result.ToolCallValidationResult` indicating whether the tool call is correctly completed by the agent.
        
        Example:
        
        >>> from agent_inspect.metrics import ToolCallCompletionValidator
        >>> from agent_inspect.metrics.constants import NUM_JUDGE_TRIALS, INCLUDE_JUDGE_EXPLANATION
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>> import asyncio
        >>>
        >>> expected_tool_call = load_expected_tool_call(expected_tool_call_file_path) # Load expected tool call checklist
        >>> agent_turn_traces = load_agent_turn_traces(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> validator = ToolCallCompletionValidator(
        ...    llm_client=client, 
        ...    config={NUM_JUDGE_TRIALS: 5, INCLUDE_JUDGE_EXPLANATION: True}
        ... )
        >>> validation_result = asyncio.run(
        ...   validator.validate(
        ...        agent_trace_turns = agent_turn_traces, 
        ...        expected_tool_call = expected_tool_call
        ...    )
        ... )
        >>> print(validation_result.is_completed)  # True or False
        """
        TraceValidator.validate_turn_traces(agent_trace_turns)
        ExpectedToolCallValidator.validate_expected_tool_call(expected_tool_call)
        
        num_judge_trials = get_config_or_default(config=self.config, config_key=NUM_JUDGE_TRIALS,
                                                                default=NUM_JUDGE_TRIALS_DEFAULT)
        include_judge_explanation = get_config_or_default(config=self.config, config_key=INCLUDE_JUDGE_EXPLANATION,
                                                                default=False)

        
        agent_turn_trace = agent_trace_turns[-1]        # validate the latest turn
        agent_tool_steps = self.get_tool_steps_from_agent_trace(agent_turn_trace=agent_turn_trace)
        # set a dict to store each tool call validation result
        tool_call_validation_results: Dict[str, bool] = {}
        explanations: List[str] = []
        for agent_single_tool_step in agent_tool_steps:
            # Check steps:
            # Step 1. check tool name match
            if expected_tool_call.tool.strip() == agent_single_tool_step.tool.strip():
                # Step 2. validate input tool arguments
                parameter_validation_results = await self._validate_parameters(
                    expected_tool_call=expected_tool_call,
                    agent_single_tool_step=agent_single_tool_step,
                    num_judge_trials=num_judge_trials,
                    include_judge_explanation=include_judge_explanation
                )
                # Step 3. validate tool output
                is_tool_output_correct, output_explanation = await self._validate_output(
                    expected_tool_call=expected_tool_call,
                    agent_single_tool_step=agent_single_tool_step,
                    num_judge_trials=num_judge_trials,
                    include_judge_explanation=include_judge_explanation
                )
                # Aggregate results & explanations
                all_parameters_correct = all(res["is_completed"] for res in parameter_validation_results.values()) and is_tool_output_correct
                tool_call_name = expected_tool_call.tool
                if all_parameters_correct:
                    res_tool_call_validation = self._aggregate_success(
                        expected_tool_call=expected_tool_call,
                        parameter_validation_results=parameter_validation_results,
                        output_explanation=output_explanation
                    )
                    tool_call_validation_results[tool_call_name] = True
                    # print(f"Tool call validation succeed for tool {tool_call_name}.")
                    break
                else:
                    # print(f"Tool call validation failed for tool {tool_call_name}.")
                    tool_call_validation_results[tool_call_name] = False
                    self._aggregate_failure(
                        expected_tool_call=expected_tool_call,
                        parameter_validation_results=parameter_validation_results,
                        output_explanation=output_explanation,
                        explanations=explanations
                    )
                    res_tool_call_validation = ToolCallValidationResult(
                        is_completed=False,
                        expected_tool_call=expected_tool_call,
                        explanations=explanations
                    )
                    
        return res_tool_call_validation if tool_call_validation_results else ToolCallValidationResult(
            is_completed=False,
            expected_tool_call=expected_tool_call,
            explanations=[f"No matching tool name \"{expected_tool_call.tool}\" is found for expected tool in this agent turn"]
        )
    
    @staticmethod
    def get_tool_steps_from_agent_trace(agent_turn_trace: TurnTrace) -> List[Step]:
        """
        Return back a list of steps where tool is called in the agent trace.
        E.g., [
        Step(id='7cacb625-111b-4495-8d71-9268c8eda081', parent_ids=[None], tool='calculator', tool_input_args=[ToolInputParameter(name='expression', value='((20750089.32 + 1000000.00) / 34673013.30) * 100')], tool_output=62.729158068300926, agent_thought=None, latency_in_sec=0.0, input_token_consumption=0, output_token_consumption=0, reasoning_token_consumption=None),
        Step(id='6e0e20d1-4b5f-431d-b8be-e9253f3dcef4', parent_ids=[None], tool='calculator', tool_input_args=[ToolInputParameter(name='expression', value='34673013.30 - (20750089.32 + 1000000.00)')], tool_output=12922923.979999997, agent_thought=None, latency_in_sec=0.0, input_token_consumption=0, output_token_consumption=0, reasoning_token_consumption=None), Step(id='8547f6e4-f7f1-49ec-b8be-e9253f3dcef4', parent_ids=['8547f6e4-f7f1-49ec-9ed1-9c346cd43215'], tool=None, tool_input_args=None, tool_output=None, agent_thought=None, latency_in_sec=0.0, input_token_consumption=0, output_token_consumption=0, reasoning_token_consumption=None)
        ]
        """
        tool_steps = []
        for step in agent_turn_trace.steps:
            if step.tool is not None:
                tool_steps.append(step)
        return tool_steps
    

    async def _validate_parameters(self, expected_tool_call: ExpectedToolCall, agent_single_tool_step: Step, num_judge_trials: int, include_judge_explanation: bool) -> Dict[str, Dict[str, Any]]:
        """Validate all expected parameters for a single tool step.
        Returns a dict keyed by parameter name with validation result & explanation.
        """
        parameter_validation_results: Dict[str, Dict[str, Any]] = {}
        if expected_tool_call.expected_parameters is not None:
            for param in expected_tool_call.expected_parameters:
                # print(f"Validating parameter: {param.name} ...")
                result = await self._validate_single_parameter(
                    param=param,
                    agent_single_tool_step=agent_single_tool_step,
                    num_judge_trials=num_judge_trials,
                    include_judge_explanation=include_judge_explanation
                )
                parameter_validation_results[param.name] = result
        return parameter_validation_results
    

    async def _validate_single_parameter(self, param: ToolInputParameter, agent_single_tool_step: Step, num_judge_trials: int, include_judge_explanation: bool) -> Dict[str, Any]:
        """Validate a single parameter (exact match or llm-as-a-judge) returning the standardized result dict.
        Preserves the same structure used previously inside _validate_parameters.
        """
        if param.value is not None and param.check is not None:
            raise InvalidInputValueError(
                internal_code=ErrorCode.INVALID_VALUE.value,
                message=f"ExpectedToolCall parameter {param.name} cannot have both value and check specified at the same time."
            )
        if param.check is not None:
            is_correct, explanation_list = await self.validate_tool_call_parameter_by_llmj(
                llm_client=self.llm_client,
                expected_tool_param=param,
                actual_tool_call=agent_single_tool_step,
                num_judge_trials=num_judge_trials,
                include_judge_explanation=include_judge_explanation
            )
            return {
                "is_completed": is_correct,
                "method": "llmj",
                "expected_check": param.check,
                "explanation": explanation_list
            }
        else:
            # when param.check is None, will ONLY trigger the exact match validation
            is_correct, explanation_str = self.validate_tool_call_parameter_by_exact_match(
                expected_tool_param=param,
                actual_tool_call_step=agent_single_tool_step
            )
            return {
                "is_completed": is_correct,
                "method": "exact_match",
                "expected_value": param.value,
                "explanation": explanation_str
            }

    async def _validate_output(self, expected_tool_call: ExpectedToolCall, agent_single_tool_step: Step, num_judge_trials: int, include_judge_explanation: bool) -> tuple[bool, Any]:
        """Validate tool output (exact match or llm-as-a-judge). Returns (is_correct, explanation or None)."""
        is_tool_output_correct = True
        output_explanation: Any = None
        if expected_tool_call.expected_output is not None:
            if expected_tool_call.expected_output.value is not None and expected_tool_call.expected_output.check is not None:
                raise InvalidInputValueError(internal_code=ErrorCode.INVALID_VALUE.value, message="ExpectedToolCall output cannot have both value and check specified at the same time.")
            if expected_tool_call.expected_output.check is not None:
                is_tool_output_correct, output_explanation = await self.validate_tool_call_output_by_llmj(
                    llm_client=self.llm_client,
                    expected_tool_call=expected_tool_call,
                    actual_tool_call=agent_single_tool_step,
                    num_judge_trials=num_judge_trials,
                    include_judge_explanation=include_judge_explanation
                )
            else:
                # when expected_tool_call.expected_output.check is None, will ONLY trigger the exact match validation
                is_tool_output_correct, output_explanation = self.validate_tool_call_output_by_exact_match(
                    expected_tool_call=expected_tool_call,
                    actual_tool_call_step=agent_single_tool_step
                )
        return is_tool_output_correct, output_explanation

    @staticmethod
    def _aggregate_success(expected_tool_call: ExpectedToolCall, parameter_validation_results: Dict[str, Dict[str, Any]], output_explanation: Any) -> ToolCallValidationResult:
        """ Aggregate successful validation:
            Append general explanation at the beginning, then tool name match, parameter explanations, output explanation
            [
                "Tool {tool_call_name} call has passed all input arguments check and output check successfully."
                "Tool name: {expected_tool_call.tool} matched.",
                "Argument \"{param_name}\" has passed exact match successfully.",
                "Argument \"{param_name}\" has passed llm-as-a-judge successfully.",
                "Tool output has passed llm-as-a-judge successfully.",
            ]
        """
        tool_call_name = expected_tool_call.tool
        correct_explanation: List[str] = []
        general_explnation_str = f"Tool \"{tool_call_name}\" call has passed all input arguments check and output check successfully."
        correct_explanation.append(general_explnation_str)
        explanations_str = f"Tool name: {expected_tool_call.tool} matched."
        correct_explanation.append(explanations_str)
        for p in parameter_validation_results.values():
            correct_explanation.extend(p["explanation"] if isinstance(p["explanation"], list) else [p["explanation"]])
        if output_explanation is not None:
            correct_explanation.extend(output_explanation if isinstance(output_explanation, list) else [output_explanation])
        return ToolCallValidationResult(
            is_completed=True,
            expected_tool_call=expected_tool_call,
            explanations=correct_explanation
        )

    @staticmethod
    def _aggregate_failure(expected_tool_call: ExpectedToolCall, parameter_validation_results: Dict[str, Dict[str, Any]], output_explanation: Any, explanations: List[str]) -> None:
        """ Append failure block explanations for a single failed attempt:
            This will include all the explanations for wrong tool call with the same tool name, formatted as:
            [
                "Tool {tool_call_name} call has failed input arguments check or output check."
                "Tool name: {expected_tool_call.tool} matched.",
                "Argument \"{param_name}\" has passed exact match successfully.",
                "Argument \"{param_name}\" has failed llm-as-a-judge.",
                "judge explanation: ...",
                "Tool output has failed llm-as-a-judge.",
                "judge explanation: ..."
            ]
        """
        tool_call_name = expected_tool_call.tool
        general_explnation_str = f"Tool \"{tool_call_name}\" call has failed input arguments check or output check."
        explanations.append(general_explnation_str)
        explanations_str = f"Tool name: {expected_tool_call.tool} matched."
        explanations.append(explanations_str)
        for p in parameter_validation_results.values():
            explanations.extend(p["explanation"] if isinstance(p["explanation"], list) else [p["explanation"]])
        if output_explanation is not None:
            explanations.extend(output_explanation if isinstance(output_explanation, list) else [output_explanation])

    
    @staticmethod
    async def validate_tool_call_parameter_by_llmj(llm_client, expected_tool_param: ToolInputParameter, actual_tool_call: Step, num_judge_trials: int, include_judge_explanation: bool) -> tuple[bool, List[str]]:
        param_name = expected_tool_param.name
        gt_param_check = expected_tool_param.check
        input_arg_value: Any = ""
        for input_arg in actual_tool_call.tool_input_args:
            if input_arg.name.strip() == param_name.strip():
                input_arg_value = input_arg.value
                break       # shouldn't have multiple params with same name in actual_tool_call
            
        # construct llmj prompt
        prompt = ToolCallCompletionValidator.construct_tool_correctness_llmj_prompt(
            arg_value=input_arg_value,
            gt_param_check=gt_param_check,
            actual_tool_call=actual_tool_call,
            is_input_arg=True
        )
        majority_voted_score, judge_explanations = await Validator.get_majority_voted_score_from_judge_responses_optimised(llm_client=llm_client,
                                                        prompt=prompt,
                                                        no_of_trials=num_judge_trials)
        is_completed = True if majority_voted_score == 1 else False
        explanations = []
        if is_completed:
            explanations.append(f"Argument \"{param_name}\" has passed llm-as-a-judge successfully.")
        else:
            explanations.append(f"Argument \"{param_name}\" has failed llm-as-a-judge.")

        if include_judge_explanation:
            explanations.extend(judge_explanations)
            
        return is_completed, explanations
                
                
    
    @staticmethod
    def construct_tool_correctness_llmj_prompt(arg_value: str, gt_param_check: str, actual_tool_call: Step, is_input_arg: bool = True) -> str:
        if is_input_arg:
            agent_tool_call_step_str = f"Step(tool='{actual_tool_call.tool}', tool_input_args=["
            for arg in actual_tool_call.tool_input_args:
                agent_tool_call_step_str += f"ToolInputParameter(name='{arg.name}', value='{arg.value}'),\n"
            agent_tool_call_step_str += "])"
        else:
            agent_tool_call_step_str = f"Step(tool='{actual_tool_call.tool}', tool_output=[ToolOutput(value='{actual_tool_call.tool_output}')])"    
        
        prompt = TOOL_CORRECTNESS_TEMPLATE.format(
            arg_value=arg_value,
            gt_value=gt_param_check,
            agent_tool_step=agent_tool_call_step_str,
        )
        return prompt
        
                
    @staticmethod
    def validate_tool_call_parameter_by_exact_match(expected_tool_param: ToolInputParameter, actual_tool_call_step: Step) -> tuple[bool, str]:
        param_name = expected_tool_param.name
        for input_arg in actual_tool_call_step.tool_input_args:
            if input_arg.name.strip() == param_name.strip():
                if input_arg.value == expected_tool_param.value:
                    return True, f"Argument \"{param_name}\" has passed exact match successfully."
                else:
                    return False, f"Argument \"{param_name}\" has failed exact match. Expected value: {expected_tool_param.value}, Expected type: {type(expected_tool_param.value)}. Actual value: {input_arg.value}, Actual type: {type(input_arg.value)}."
        return False, f"Argument \"{param_name}\" not even found in actual tool call. Expected value: {expected_tool_param.value}, Expected type: {type(expected_tool_param.value)}."

    @staticmethod
    def validate_tool_call_output_by_exact_match(expected_tool_call: ExpectedToolCall, actual_tool_call_step: Step) -> tuple[bool, str]:
        actual_output = actual_tool_call_step.tool_output
        expect_output = expected_tool_call.expected_output.value
        if actual_output == expect_output:
            return True, "Tool output has passed exact match successfully."
        else:
            return False, f"Tool output has failed exact match. Expected output: {expect_output}, Expected type: {type(expect_output)}. Actual output: {actual_output}, Actual type: {type(actual_output)}."
        

    @staticmethod
    async def validate_tool_call_output_by_llmj(llm_client, expected_tool_call: ExpectedToolCall, actual_tool_call: Step, num_judge_trials: int, include_judge_explanation: bool) -> tuple[bool, List[str]]:
        prompt = ToolCallCompletionValidator.construct_tool_correctness_llmj_prompt(
            arg_value=actual_tool_call.tool_output,
            gt_param_check=expected_tool_call.expected_output.check,
            actual_tool_call=actual_tool_call,
            is_input_arg=False
        )
        majority_voted_score, judge_explanations = await Validator.get_majority_voted_score_from_judge_responses_optimised(llm_client=llm_client,
                                                        prompt=prompt,
                                                        no_of_trials=num_judge_trials)
        is_completed = True if majority_voted_score == 1 else False
        explanations = []
        if is_completed:
            explanations.append("Tool output has passed llm-as-a-judge successfully.")
        else:
            explanations.append("Tool output has failed llm-as-a-judge.")

        if include_judge_explanation:
            explanations.extend(judge_explanations)
            
        return is_completed, explanations
