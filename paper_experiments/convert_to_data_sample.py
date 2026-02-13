import json
import ast
import re

from agent_inspect.models.metrics import SubGoal, Conversation, ExpectedToolCall, ToolInputParameter, EvaluationSample

def parse_tool_call_string(tool_string: str):
    """
    Parse tool call from Python string format to structured data.
    
    Handles formats like:
    "[{'tool_code': 'get_user_details(user_id=ivan_rossi_8555)', 'output': '$AnyValue'}]"
    
    Returns:
        dict with 'tool' name and 'parameters' dict
    """
    #TODO: abstract duplicate code into another method to reduce duplicate code
    try:
        # First try to parse as JSON (in case format is already correct)
        parsed = json.loads(tool_string)
        if isinstance(parsed, list) and len(parsed) > 0:
            tool_dict = parsed[0]
            if 'tool_code' in tool_dict:
                # Extract tool name and parameters from tool_code
                tool_code = tool_dict['tool_code']
                return parse_tool_code(tool_code)
    except json.JSONDecodeError:
        pass
    
    # Try parsing as Python literal (handles single quotes)
    try:
        parsed = ast.literal_eval(tool_string)
        if isinstance(parsed, list) and len(parsed) > 0:
            tool_dict = parsed[0]
            if 'tool_code' in tool_dict:
                tool_code = tool_dict['tool_code']
                return parse_tool_code(tool_code)
    except (ValueError, SyntaxError):
        pass
    
    # If all else fails, try regex parsing
    return parse_tool_code_regex(tool_string)


def parse_tool_code(tool_code: str):
    """
    Parse tool_code string like 'get_user_details(user_id=ivan_rossi_8555)'
    
    Returns:
        dict with 'tool' name and 'parameters' dict
    """
    # Extract tool name
    match = re.match(r'(\w+)\((.*)\)', tool_code)
    if not match:
        return {"tool": tool_code, "parameters": {}}
    
    tool_name = match.group(1)
    params_str = match.group(2)
    
    # Parse parameters
    parameters = {}
    if params_str:
        # Split by comma, but respect nested structures
        param_pairs = re.findall(r'(\w+)=([^,]+(?:\[.*?\])?)', params_str)
        for key, value in param_pairs:
            # Clean up value
            value = value.strip()
            # Remove quotes if present
            if (value.startswith("'") and value.endswith("'")) or \
               (value.startswith('"') and value.endswith('"')):
                value = value[1:-1]
            parameters[key] = value
    
    return {"tool": tool_name, "parameters": parameters}


def parse_tool_code_regex(tool_string: str):
    """Fallback regex parser for malformed tool strings."""
    # Try to extract tool name
    tool_match = re.search(r"'tool_code':\s*'(\w+)", tool_string)
    if tool_match:
        return parse_tool_code(tool_match.group(0).split("'")[-1])
    
    return {"tool": "unknown", "parameters": {}}

    
def convert_sample_to_data_sample(dataset_json_obj):
    """Convert dataset JSON object to EvaluationSample."""
    
    sub_goals = []

    for sub_goal_json in dataset_json_obj["metadata"]["subgoals"]:
        sub_goals.append(
            SubGoal(
                type=sub_goal_json["type"],
                details=sub_goal_json["details"],
                turn=sub_goal_json["turn"]
            )
        )
    
    i = 0
    conversations = []
    if len(dataset_json_obj.get("target", "")) > 0:
        for input_item, target in zip(dataset_json_obj["input"], dataset_json_obj["target"]):
            conversations.append(
                Conversation(
                    turn_id=i, 
                    message=input_item["content"], 
                    expected_response=target
                )
            )
            i += 1

    expected_tool_calls = []
    user_instruction = dataset_json_obj["input"][0]["content"]

    # Parse expected tools
    if "expected_tools" in dataset_json_obj["metadata"]:
        for tool_string in dataset_json_obj["metadata"]["expected_tools"]:
            try:
                # Parse the tool string (handles both JSON and Python string formats)
                parsed_tool = parse_tool_call_string(tool_string)
                
                tool_name = parsed_tool["tool"]
                expected_parameters = []
                
                # Convert parameters to ToolInputParameter objects
                for key, value in parsed_tool.get("parameters", {}).items():
                    if value in ["$AnyValue", "No Remarks"] or key in ["expression"]:
                        # Parameters to be checked by LLM
                        expected_parameter = ToolInputParameter(name=key, check=value)
                    else:
                        expected_parameter = ToolInputParameter(name=key, value=value)
                    expected_parameters.append(expected_parameter)
                
                expected_tool_calls.append(
                    ExpectedToolCall(tool=tool_name, expected_parameters=expected_parameters)
                )
            except Exception as e:
                # Log error but continue processing
                print(f"Warning: Failed to parse tool string '{tool_string}': {e}")
                continue
    
    return EvaluationSample(
        conversation=conversations, 
        expected_tool_calls=expected_tool_calls, 
        sub_goals=sub_goals,
        user_instruction=user_instruction
    )
