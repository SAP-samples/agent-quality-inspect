TOOL_CORRECTNESS_TEMPLATE = """
As a judge, your task is to assess whether the [Argument Value] used in the tool call is correct by comparing it with the expected [Ground Truth Value]. To make this assessment, consider the context provided in the [Agent Tool Call Step] as additional information, since [Argument Value] variable is extracted from this complete source.

When the [Argument Value] and [Ground Truth Value] semantically match completely, it is considered to be COMPLETE. However, note that minor differences such as style, grammar, capitalization, punctuation, or synonyms do not affect semantic matching. If the [Argument Value] includes additional information or details that are CONSISTENT with any other arguments in the [Agent Tool Call Step], it should also be considered a correct match and must be marked as COMPLETE. However, if the additional information contradicts directly or indirectly with any other arguments in the [Agent Tool Call Step], it must be marked as INCOMPLETE.

[BEGIN DATA]
************
[Argument Value]:
{arg_value}

************
[Ground Truth Value]:
{gt_value}

************
[Agent Tool Call Step]:
{agent_tool_step}

************
[END DATA]

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. End with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the [Argument Value] is correct. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE.
"""

DEFAULT_MODEL_GRADED_FACT_SINGLE_TURN_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL = """
You are provided with a sample containing a gold-standard user input [Gold User Input]. Gold Expert Answer is not provided. The actual agent response is provided in section [Agent Response Submission] and the [Agent Intermediate Trajectories] section details the steps taken by the agent.

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Response Submission] to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[Gold User Input]:
{questions}

************
[Ground Truth Subgoal]:
{subgoal}

************
[Agent Intermediate Trajectories]:
{trajectories}

************
[Agent Response Submission]:
{answers}

************
[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""

DEFAULT_MODEL_GRADED_FACT_MULTI_TURN_AT_CURRENT_TURN_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL = """
You are provided with a sample containing a gold-standard user input at the current conversational turn in the [Gold User Input] section which may include question, instruction, or response to the agent. Gold Expert Answer are not provided. The actual agent response at the current conversational turn is provided in section [Agent Response Submission] and the [Agent Intermediate Trajectories] section details the steps taken by the agent for the current conversational turn.

For additional context, user inputs, agent trajectories, and agent responses for all the past conversational turns are also provided in the sections [Past User Inputs], [Past Agent Trajectories], and [Past Agent Responses], respectively.

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved for the CURRENT turn. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Response Submission] that contain information of the CURRENT turn to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[Past User Inputs]:
{past_user_inputs}

************
[Past Agent Trajectories]:
{past_agent_trajectories}

************
[Past Agent Responses]:
{past_agent_responses}

************
[Gold User Input]:
{questions}

************
[Ground Truth Subgoal]:
{subgoal}

************
[Agent Intermediate Trajectories]:
{trajectories}

************
[Agent Response Submission]:
{answers}

************
[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""

DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL = """
You are provided with a sample that contains several key components centered around an interaction between an agent and a simulated user, referred to as the user proxy. The user proxy represents a human-in-the-loop, engaging with the agent by posing questions and guiding the conversation throughout the dialogue.

The [User Summary Instructions] section outlines the user’s goals, expectations, and the overall task the agent is expected to complete. The [Agent Responses Submission] section captures the agent’s actual responses to the user proxy at each turn of the interaction. The [Agent Intermediate Trajectories] section provides a detailed step-by-step reasoning and actions taken by the agent. Finally, the [Dynamic Dialogue] section presents the full conversation between the agent and the user proxy. 

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Responses Submission] to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[User Summary Instructions]:
{userTask}

************
[Ground Truth Subgoal]:
{subgoal}

************
[Agent Intermediate Trajectories]:
{trajectories}

************
[Agent Responses Submission]:
{answers}

************
[Dynamic Dialogue]:
{dynamicDialogue}
[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""

DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL = """
You are provided with a sample that contains several key components centered around an interaction between an agent and a simulated user, referred to as the user proxy. The user proxy represents a human-in-the-loop, engaging with the agent by posing questions and guiding the conversation throughout the dialogue.

The [Agent Responses Submission] section captures the agent’s actual responses to the user proxy at each turn of the interaction. The [Agent Intermediate Trajectories] section provides a detailed step-by-step reasoning and actions taken by the agent. Finally, the [Dynamic Dialogue] section presents the full conversation between the agent and the user proxy. 

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Responses Submission] to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[Ground Truth Subgoal]:
{subgoal}

************
[Agent Intermediate Trajectories]:
{trajectories}

************
[Agent Responses Submission]:
{answers}

************
[Dynamic Dialogue]:
{dynamicDialogue}
[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""
