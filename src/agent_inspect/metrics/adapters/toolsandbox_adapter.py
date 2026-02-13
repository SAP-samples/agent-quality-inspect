from typing import List, Dict, Any, Optional, Tuple
import json

from agent_inspect.metrics.adapters.base_adapter import BaseAdapter
from agent_inspect.models.metrics.agent_trace import AgentDialogueTrace, TurnTrace, Step, AgentResponse
from agent_inspect.models.metrics.agent_data_sample import ToolInputParameter


class ToolsandboxAdapter(BaseAdapter):
    """
    Adapter for converting toolsandbox conversation format to AgentDialogueTrace format.
    """

    def convert_to_agent_trace(self, conversation_data: List[Dict[str, Any]]) -> AgentDialogueTrace:
        """
        Convert tool_sandbox conversation format to AgentDialogueTrace format.

        :param conversation_data: List of conversation turns, each containing role-based messages.
        :return: Converted agent trace.
        """
        turns = []

        for turn_idx, turn_messages in enumerate(conversation_data):
            turn_trace = self._process_turn(turn_messages, turn_idx)
            if turn_trace:  # Only add turns with user input
                turns.append(turn_trace)

        return AgentDialogueTrace(turns=turns)

    def _process_turn(self, turn_messages: List[Dict[str, Any]], turn_idx: int) -> Optional[TurnTrace]:
        """
        Process a single conversation turn into a TurnTrace.

        :param turn_messages: List of messages in this turn
        :param turn_idx: Index of this turn
        :return: TurnTrace object or None if no user input found
        """
        user_input = self._extract_user_input(turn_messages)
        if user_input is None:
            return None

        agent_response, steps = self._process_assistant_messages(turn_messages, turn_idx)

        #TODO: To revisit after paper submission
        if agent_response is None:
            agent_response = AgentResponse(response="Agent did not respond.", status_code="200")

        return TurnTrace(
            id=f"turn_{turn_idx}",
            agent_input=user_input,
            agent_response=agent_response,
            from_id=f"turn_{turn_idx - 1}" if turn_idx > 0 else None,
            steps=steps,
            latency_in_ms=None
        )

    def _extract_user_input(self, turn_messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract user input from turn messages.

        :param turn_messages: List of messages in the turn
        :return: User input content or None
        """
        for message in turn_messages:
            if message.get("role") == "user":
                return message.get("content")
        return None

    def _process_assistant_messages(self, turn_messages: List[Dict[str, Any]], turn_idx: int) -> Tuple[
        Optional[AgentResponse], List[Step]]:
        """
        Process assistant messages to extract response and tool steps.

        :param turn_messages: List of messages in the turn
        :param turn_idx: Index of this turn
        :return: Tuple of (agent_response, steps)
        """
        agent_response = None
        steps = []
        step_counter = 0

        for message in turn_messages:
            if message.get("role") == "assistant":
                # Process tool calls first
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    new_steps = self._process_tool_calls(tool_calls, turn_messages, turn_idx, step_counter)
                    steps.extend(new_steps)
                    step_counter += len(new_steps)

                # Process assistant response content
                content = message.get("content")
                if content:
                    agent_response = AgentResponse(response=content, status_code="200")

        return agent_response, steps

    def _process_tool_calls(self, tool_calls: List[Dict[str, Any]], turn_messages: List[Dict[str, Any]],
                            turn_idx: int, step_offset: int) -> List[Step]:
        """
        Process tool calls into Step objects.

        :param tool_calls: List of tool call dictionaries
        :param turn_messages: All messages in the turn (to find tool responses)
        :param turn_idx: Index of the current turn
        :param step_offset: Current number of steps (for sequential numbering)
        :return: List of Step objects
        """
        steps = []

        for step_idx, tool_call in enumerate(tool_calls):
            tool_input_args = self._parse_tool_arguments(tool_call)
            tool_output = self._find_tool_output(tool_call.get("id"), turn_messages)
            parent_ids = self._get_parent_ids(step_offset + step_idx, steps)

            step = Step(
                id=f"turn_{turn_idx}_step_{step_offset + step_idx}",
                parent_ids=parent_ids,
                tool=self._extract_tool_name(tool_call),
                tool_input_args=tool_input_args,
                tool_output=tool_output,
                agent_thought=None,
                input_token_consumption=None,
                output_token_consumption=None,
                reasoning_token_consumption=None
            )
            steps.append(step)

        return steps

    def _parse_tool_arguments(self, tool_call: Dict[str, Any]) -> List[ToolInputParameter]:
        """
        Parse tool call arguments into ToolInputParameter objects.

        :param tool_call: Tool call dictionary
        :return: List of ToolInputParameter objects
        """
        tool_input_args = []

        if "function" in tool_call and "arguments" in tool_call["function"]:
            try:
                args_dict = json.loads(tool_call["function"]["arguments"])
                for key, value in args_dict.items():
                    tool_input_args.append(ToolInputParameter(name=key, value=value))
            except json.JSONDecodeError:
                # Fallback for invalid JSON
                tool_input_args.append(
                    ToolInputParameter(
                        name="arguments",
                        value=tool_call["function"]["arguments"]
                    )
                )

        return tool_input_args

    def _find_tool_output(self, tool_call_id: str, turn_messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find the tool output for a given tool call ID.

        :param tool_call_id: ID of the tool call
        :param turn_messages: All messages in the turn
        :return: Tool output content or None
        """
        for message in turn_messages:
            if (message.get("role") == "tool" and
                    message.get("tool_call_id") == tool_call_id):

                tool_output = message.get("content", "")
                tool_details = message.get("tool_details")

                if tool_details is not None:
                    tool_output += "\n" + str(tool_details)

                return tool_output if tool_output else None

        return None

    def _extract_tool_name(self, tool_call: Dict[str, Any]) -> Optional[str]:
        """
        Extract tool name from tool call.

        :param tool_call: Tool call dictionary
        :return: Tool name or None
        """
        return tool_call.get("function", {}).get("name")

    def _get_parent_ids(self, current_step_idx: int, existing_steps: List[Step]) -> List[str]:
        """
        Get parent IDs for the current step.

        :param current_step_idx: Index of current step within the turn
        :param existing_steps: List of already created steps
        :return: List of parent step IDs
        """
        if current_step_idx > 0 and existing_steps:
            return [existing_steps[-1].id]
        return []