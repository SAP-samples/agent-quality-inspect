"""
Constants for tool call validation explanation formats.

These format strings are used by ToolCallCompletionValidator to generate consistent explanations
and by DeterministicToolCallErrorAnalysis for regex pattern matching during error classification.
"""

# Tool name validation explanations
TOOL_NOT_FOUND_EXPLANATION = (
    'No matching tool name "{tool_name}" is found for expected tool in this agent turn'
)

# Argument validation explanations
ARGUMENT_PASSED_EXACT_MATCH_EXPLANATION = (
    'Argument "{param_name}" has passed exact match successfully.'
)
ARGUMENT_FAILED_EXACT_MATCH_EXPLANATION = 'Argument "{param_name}" has failed exact match. Expected value: {expected_value}, Expected type: {expected_type}. Actual value: {actual_value}, Actual type: {actual_type}.'
ARGUMENT_NOT_FOUND_EXPLANATION = 'Argument "{param_name}" not even found in actual tool call. Expected value: {expected_value}, Expected type: {expected_type}.'
ARGUMENT_PASSED_LLMJ_EXPLANATION = 'Argument "{param_name}" has passed llm-as-a-judge successfully.'
ARGUMENT_FAILED_LLMJ_EXPLANATION = 'Argument "{param_name}" has failed llm-as-a-judge.'

# Tool output validation explanations
OUTPUT_PASSED_EXACT_MATCH_EXPLANATION = "Tool output has passed exact match successfully."
OUTPUT_FAILED_EXACT_MATCH_EXPLANATION = "Tool output has failed exact match. Expected output: {expected_output}, Expected type: {expected_type}. Actual output: {actual_output}, Actual type: {actual_type}."
OUTPUT_PASSED_LLMJ_EXPLANATION = "Tool output has passed llm-as-a-judge successfully."
OUTPUT_FAILED_LLMJ_EXPLANATION = "Tool output has failed llm-as-a-judge."
