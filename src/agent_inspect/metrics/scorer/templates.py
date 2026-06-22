DEFAULT_MODEL_GRADED_FACT_SINGLE_TURN_TEMPLATE_ONE_SUBGOAL = """
You are provided with a sample containing a gold-standard user input [Gold User Input]. Gold Expert Answer is not provided. The actual agent response is provided in section [Agent Response Submission] and the [Agent Intermediate Trajectories] section details the steps taken by the agent.

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Response Submission] to ascertain whether the subgoal has been successfully completed.

When the completion of a subgoal depends on the availability of a specific tool, and that TOOL is ABSENT, the subgoal must be marked as INCOMPLETE, REGARDLESS of the correctness of the agent's responses in [Agent Response Submission]. Tool availability is a hard constraint—no amount of valid reasoning can compensate for its absence. Suggesting or recommending a tool is acceptable only if the subgoal does not require the tool to be used. However, CONCEPTUALIZING or HALLUCINATING a tool call—i.e., falsely implying the tool was invoked—is NOT acceptable, and the subgoal must be marked as INCOMPLETE.

Additionally if the agent intermediate trajectory are left BLANK or DOES NOT contain relevant tools to the subgoal, it means that the ground truth subgoal is NOT MET and the subgoal is considered as INCOMPLETE, REGARDLESS of the correctness of the agent's response in [Agent Response Submission]. The thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE.

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

DEFAULT_MODEL_GRADED_FACT_MULTI_TURN_TEMPLATE_ONE_SUBGOAL = """
You are provided with a sample containing a sequence of gold-standard multi-turn user inputs (which may include questions, instructions, or responses) [Gold User Inputs] to the agent at each turn. Gold Expert Answers are not provided. The [Agent Responses Submission] section includes the agent's actual responses at every turn, and the [Agent Intermediate Trajectories] section details the steps taken by the agent.

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Responses Submission] to ascertain whether the subgoal has been successfully completed.

When the completion of a subgoal depends on the availability of a specific tool, and that TOOL is ABSENT, the subgoal must be marked as INCOMPLETE, REGARDLESS of the correctness of the agent's responses in [Agent Responses Submission]. Tool availability is a hard constraint—no amount of valid reasoning can compensate for its absence. Suggesting or recommending a tool is acceptable only if the subgoal does not require the tool to be used. However, CONCEPTUALIZING or HALLUCINATING a tool call—i.e., falsely implying the tool was invoked—is NOT acceptable, and the subgoal must be marked as INCOMPLETE.

Additionally if the agent intermediate trajectory are left BLANK or DOES NOT contain relevant tools to the subgoal, it means that the ground truth subgoal is NOT MET and the subgoal is considered as INCOMPLETE, REGARDLESS of the correctness of the agent's responses in [Agent Responses Submission]. The thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE.

[BEGIN DATA]
************
[Gold User Inputs]:
{questions}

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
[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""

DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_TEMPLATE_ONE_SUBGOAL = """
You are provided with a sample that contains several key components centered around an interaction between an agent and a simulated user, referred to as the user proxy. The user proxy represents a human-in-the-loop, engaging with the agent by posing questions and guiding the conversation throughout the dialogue.

The [User Summary Instructions] section outlines the user’s goals, expectations, and the overall task the agent is expected to complete. The [Agent Responses Submission] section captures the agent’s actual responses to the user proxy at each turn of the interaction. The [Agent Intermediate Trajectories] section provides a detailed step-by-step reasoning and actions taken by the agent. Finally, the [Dynamic Dialogue] section presents the full conversation between the agent and the user proxy.

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Responses Submission] to ascertain whether the subgoal has been successfully completed.

When the completion of a subgoal depends on the availability of a specific tool, and that TOOL is ABSENT, the subgoal must be marked as INCOMPLETE, REGARDLESS of the correctness of the agent's responses in [Agent Responses Submission]. Tool availability is a hard constraint—no amount of valid reasoning can compensate for its absence. Suggesting or recommending a tool is acceptable only if the subgoal does not require the tool to be used. However, CONCEPTUALIZING or HALLUCINATING a tool call—i.e., falsely implying the tool was invoked—is NOT acceptable, and the subgoal must be marked as INCOMPLETE.

Additionally if the agent intermediate trajectory are left BLANK or DOES NOT contain relevant tools to the subgoal, it means that the ground truth subgoal is NOT MET and the subgoal is considered as INCOMPLETE, REGARDLESS of the correctness of the agent's responses in [Agent Responses Submission]. The thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE.

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
[Dynamic Dialogue]
{dynamicDialogue}
[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""

DEFAULT_MODEL_GRADED_FACT_MULTI_TURN_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL = """
You are provided with a sample containing a sequence of gold-standard multi-turn user inputs (which may include questions, instructions, or responses) [Gold User Inputs] to the agent at each turn. Gold Expert Answers are not provided. The [Agent Responses Submission] section includes the agent's actual responses at every turn, and the [Agent Intermediate Trajectories] section details the steps taken by the agent.

As a judge, your task is to determine whether the subgoal specified in the [Ground Truth Subgoal] has been achieved. To make this assessment, evaluate the [Agent Intermediate Trajectories] and [Agent Responses Submission] to ascertain whether the subgoal has been successfully completed.

Additionally, the thought in [Agent Intermediate Trajectories]  CANNOT be considered as a substitute for a tool call and the ground truth subgoal is considered to be INCOMPLETE. Do NOT attempt to infer or reconcile differences between values when they are clearly different or potentially contain typographical errors. This rule does not apply to acceptable rounding differences or cases where the same value is represented in another form that is still semantically equivalent (e.g., different number formats or naming conventions that retain the same meaning).

[BEGIN DATA]
************
[Gold User Inputs]:
{questions}

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
[END DATA]

During assessment focus solely on the factual content and the goal completion while disregarding any differences in style, grammar, punctuation, or syntax.

Begin by presenting a concise argument to confirm the validity of your conclusion. Avoid simply stating the correct answers at the outset. Decide what type of tools is required and then end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I. Reply with 'GRADE: C' (without quotes) to indicate COMPLETE if the agent has successfully achieved the subgoal. Otherwise, reply with 'GRADE: I' (without quotes) to indicate INCOMPLETE if the agent did not achieved the subgoal.
"""
