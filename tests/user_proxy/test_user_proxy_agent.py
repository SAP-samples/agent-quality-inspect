import asyncio
from http import HTTPStatus
from unittest.mock import MagicMock, AsyncMock

import pytest

from agent_inspect.user_proxy.constants import (
    USE_EXPERT_AGENT,
    TERMINATING_CONDITION_MODE,
    USE_EXPERT_AGENT_DEFAULT,
    MULTIPLE,
)
from agent_inspect.exception import UserProxyError
from agent_inspect.models import LLMResponse
from agent_inspect.models.user_proxy import (
    ChatHistory,
    ConversationTurn,
    UserProxyMessage,
    ResponseFromAgent,
    TerminatingCondition,
)
from agent_inspect.models.user_proxy.terminating_condition import (
    DEFAULT_CHECK,
    DEFAULT_STOP_SEQUENCE,
    DEFAULT_DONE_STOP_SEQUENCE,
    DEFAULT_DELEGATED_STOP_SEQUENCE,
    DEFAULT_BLOCKED_STOP_SEQUENCE,
    TaskCompletedTerminatingCondition,
    TaskDelegatedTerminatingCondition,
    TaskBlockedTerminatingCondition,
)
from agent_inspect.user_proxy import UserProxyAgent


# Tests ensure_full_stop is actually called and stored
def test_init_appends_full_stop_to_task_summary():
    agent = UserProxyAgent(MagicMock(), "Do something", [TerminatingCondition()])
    assert agent.task_summary == "Do something."


def test_init_is_expert_defaults_to_default():
    agent = UserProxyAgent(MagicMock(), "Do something.", [TerminatingCondition()])
    assert agent.is_expert is USE_EXPERT_AGENT_DEFAULT


def test_init_is_expert_from_config():
    agent_false = UserProxyAgent(
        MagicMock(),
        "Do something.",
        [TerminatingCondition()],
        config={USE_EXPERT_AGENT: False},
    )
    agent_true = UserProxyAgent(
        MagicMock(),
        "Do something.",
        [TerminatingCondition()],
        config={USE_EXPERT_AGENT: True},
    )
    assert agent_false.is_expert is False
    assert agent_true.is_expert is True


def test_init_single_terminating_condition_mode_correct_assignment():
    agent = UserProxyAgent(MagicMock(), "Do something.", [TerminatingCondition()])
    assert agent.is_multi_terminating_conditions is False
    assert hasattr(agent, "terminating_condition")
    assert not hasattr(agent, "task_completed_terminating_condition")
    assert not hasattr(agent, "task_delegated_terminating_condition")
    assert not hasattr(agent, "task_blocked_terminating_condition")


def test_init_terminating_conditions_with_empty_fields_replaces_with_defaults():
    condition_with_empty_fields = TerminatingCondition(check="   ", stop_sequence="   ")
    agent = UserProxyAgent(MagicMock(), "Do something.", [condition_with_empty_fields])
    assert agent.terminating_condition.check == DEFAULT_CHECK
    assert agent.terminating_condition.stop_sequence == DEFAULT_STOP_SEQUENCE


def test_init_multi_terminating_conditions_mode_correct_assignment():
    agent = UserProxyAgent(
        MagicMock(),
        "Do something.",
        [
            TaskCompletedTerminatingCondition(),
            TaskDelegatedTerminatingCondition(),
            TaskBlockedTerminatingCondition(),
        ],
        config={TERMINATING_CONDITION_MODE: MULTIPLE},
    )
    assert agent.is_multi_terminating_conditions is True
    assert hasattr(agent, "task_completed_terminating_condition")
    assert hasattr(agent, "task_delegated_terminating_condition")
    assert hasattr(agent, "task_blocked_terminating_condition")
    assert not hasattr(agent, "terminating_condition")


def test_init_multi_terminating_conditions_mode_all_use_defaults_when_empty_list_given():
    # When TERMINATING_CONDITION_MODE is MULTIPLE and no conditions are
    # supplied, all three slots must be filled with their default instances.
    agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[],
        agent_description="Test Agent Description",
        config={TERMINATING_CONDITION_MODE: MULTIPLE},
    )
    assert agent.task_completed_terminating_condition == TaskCompletedTerminatingCondition()
    assert agent.task_delegated_terminating_condition == TaskDelegatedTerminatingCondition()
    assert agent.task_blocked_terminating_condition == TaskBlockedTerminatingCondition()


def test_init_multi_terminating_conditions_mode_uses_provided_and_defaults_for_remainder():
    # When only one typed condition is supplied, the other two slots must be
    # filled with their respective defaults.
    custom_completed = TaskCompletedTerminatingCondition(check="custom done check")
    agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[custom_completed],
        agent_description="Test Agent Description",
        config={TERMINATING_CONDITION_MODE: MULTIPLE},
    )
    assert agent.task_completed_terminating_condition == custom_completed
    assert agent.task_delegated_terminating_condition == TaskDelegatedTerminatingCondition()
    assert agent.task_blocked_terminating_condition == TaskBlockedTerminatingCondition()


def test_init_multi_terminating_conditions_mode_uses_all_provided_when_all_three_given():
    # When all three typed conditions are supplied, no defaults should be used.
    custom_completed = TaskCompletedTerminatingCondition(check="custom done")
    custom_delegated = TaskDelegatedTerminatingCondition(check="custom delegate")
    custom_blocked = TaskBlockedTerminatingCondition(check="custom blocked")
    agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[custom_completed, custom_delegated, custom_blocked],
        agent_description="Test Agent Description",
        config={TERMINATING_CONDITION_MODE: MULTIPLE},
    )
    assert agent.task_completed_terminating_condition == custom_completed
    assert agent.task_delegated_terminating_condition == custom_delegated
    assert agent.task_blocked_terminating_condition == custom_blocked


def test_get_system_prompt_expert_multi_terminating_conditions():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[
            TaskCompletedTerminatingCondition(check="Check 1."),
            TaskDelegatedTerminatingCondition(check="Check 2."),
            TaskBlockedTerminatingCondition(check="Check 3."),
        ],
        agent_description="Test Agent Description",
        config={USE_EXPERT_AGENT: True, TERMINATING_CONDITION_MODE: MULTIPLE},
    )

    expected_system_prompt = f"""
You are acting as an expert LLM-simulated user who fully understands the AI assistant system and goal. Always respond naturally in clear, concise language that fits the expert user role and goal. Provide complete and precise information in your responses. Generate one line at a time. Do not give away all the instructions at once. Only provide the information that is necessary for the current step.

You are provided with the following user task summary:
[user_task_summary]
Test Task Summary.

[Termination Statuses]:
The task must end with either ONE of the following status tokens:
1. {DEFAULT_DONE_STOP_SEQUENCE}
2. {DEFAULT_DELEGATED_STOP_SEQUENCE}
3. {DEFAULT_BLOCKED_STOP_SEQUENCE}

[Termination Conditions]:
1. Check 1.:
   {DEFAULT_DONE_STOP_SEQUENCE}

2. Check 2.:
   {DEFAULT_DELEGATED_STOP_SEQUENCE}

3. Check 3.:
   {DEFAULT_BLOCKED_STOP_SEQUENCE}

You understand the system well and will provide thorough, accurate responses using only the information provided in the [user_task_summary] section.

If the AI assistant returns output in JSON format, respond only to the content inside the JSON as if the format does not matter.

---
The following provides an overview of the AI assistant if available.
[AI Assistant Description] :
Test Agent Description


---
When you as an expert LLM-simulated user is analysing the real-time chat history, carry out a two-step process as the user: 
first, a Reflection Phase, followed by a Response Generation Phase.

"""

    actual_system_prompt = user_proxy_agent._get_system_prompt()
    assert actual_system_prompt == expected_system_prompt


def test_get_system_prompt_expert_single_terminating_conditions():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[TerminatingCondition(check="This is just a check.")],
        agent_description="Test Agent Description",
        config={USE_EXPERT_AGENT: True},
    )

    expected_system_prompt = """
You are acting as an expert LLM-simulated user who fully understands the AI assistant system and goal. Always respond naturally in clear, concise language that fits the expert user role and goal. Provide complete and precise information in your responses. Generate one line at a time. Do not give away all the instructions at once. Only provide the information that is necessary for the current step.

You are provided with the following user task summary:
[user_task_summary]
Test Task Summary. This is just a check.

You understand the system well and will provide thorough, accurate responses using only the information provided in the [user_task_summary] section.

If the AI assistant returns output in JSON format, respond only to the content inside the JSON as if the format does not matter.

---
The following provides an overview of the AI assistant if available.
[AI Assistant Description] :
Test Agent Description


---
When you as an expert LLM-simulated user is analysing the real-time chat history, carry out a two-step process as the user: 
first, a Reflection Phase, followed by a Response Generation Phase.
"""
    actual_system_prompt = user_proxy_agent._get_system_prompt()
    assert actual_system_prompt == expected_system_prompt


def test_get_system_prompt_non_expert_multi_terminating_conditions():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[
            TaskCompletedTerminatingCondition(check="Check 1."),
            TaskDelegatedTerminatingCondition(check="Check 2."),
            TaskBlockedTerminatingCondition(check="Check 3."),
        ],
        agent_description="Test Agent Description",
        config={USE_EXPERT_AGENT: False, TERMINATING_CONDITION_MODE: MULTIPLE},
    )

    expected_system_prompt = f"""
You are simulating a clueless, casual NON-expert user who is interacting with an AI assistant. You don’t fully understand how the AI system works, and you tend to give vague or incomplete instructions — often leaving out key steps or context.

When you respond:

Speak naturally, casually, like someone who's unsure how to talk to an AI.

Be brief and only provide part of the needed information.

Do not give a full picture unless the assistant directly asks for it.

Only share details that are directly related to what was just asked or prompted — not more.

Never proactively explain your reasoning or provide background info unless the assistant digs into it.

You are working toward the following general task:
[User Task Summary]
Test Task Summary.

[Termination Statuses]:
The task must end with either ONE of the following status tokens:
1. {DEFAULT_DONE_STOP_SEQUENCE}
2. {DEFAULT_DELEGATED_STOP_SEQUENCE}
3. {DEFAULT_BLOCKED_STOP_SEQUENCE}

[Termination Conditions]:
1. Check 1.:
   {DEFAULT_DONE_STOP_SEQUENCE}

2. Check 2.:
   {DEFAULT_DELEGATED_STOP_SEQUENCE}

3. Check 3.:
   {DEFAULT_BLOCKED_STOP_SEQUENCE}

But since you’re not an expert, you’ll just sort of "feel your way through it" and leave lots of gaps in your instructions. NEVER provide COMPLETE instructions. ALWAYS OMIT some variables and missing key context.
If the assistant returns something in structured formats like JSON, you can just react casually to the content. Treat the format like it doesn’t matter.

---
The following provides an overview of the AI assistant if available.
[AI Assistant Description] :
Test Agent Description

---
When you as a clueless, casual NON-expert user is analysing the real-time chat history, carry out a two-step process as the user: 
first, a Reflection Phase, followed by a Response Generation Phase.

When simulating your process during the conversation:
You go through two internal steps each time:

1. Reflection Phase (internal thought):
Take a quick look at the current chat history. Think to yourself:
“Okay, what did the assistant just say or ask? What should I probably say next without overexplaining?”
Remember: you're not confident in how this system works, so don’t try to be precise.

2. Response Generation Phase (your reply):
Now write a short, casual message that gives only partial information based on what the assistant asked. Leave things unclear unless the assistant is persistent.


"""

    actual_system_prompt = user_proxy_agent._get_system_prompt()
    assert actual_system_prompt == expected_system_prompt


def test_get_system_prompt_non_expert_single_terminating_conditions():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[TerminatingCondition(check="This is just a check.")],
        agent_description="Test Agent Description",
        config={USE_EXPERT_AGENT: False},
    )

    expected_system_prompt = """
You are simulating a clueless, casual NON-expert user who is interacting with an AI assistant. You don’t fully understand how the AI system works, and you tend to give vague or incomplete instructions — often leaving out key steps or context.

When you respond:

Speak naturally, casually, like someone who's unsure how to talk to an AI.

Be brief and only provide part of the needed information.

Do not give a full picture unless the assistant directly asks for it.

Only share details that are directly related to what was just asked or prompted — not more.

Never proactively explain your reasoning or provide background info unless the assistant digs into it.

You are working toward the following general task:
[User Task Summary]
Test Task Summary. This is just a check.

But since you’re not an expert, you’ll just sort of "feel your way through it" and leave lots of gaps in your instructions. NEVER provide COMPLETE instructions. ALWAYS OMIT some variables and missing key context.
If the assistant returns something in structured formats like JSON, you can just react casually to the content. Treat the format like it doesn’t matter.

---
The following provides an overview of the AI assistant if available.
[AI Assistant Description] :
Test Agent Description

---
When you as a clueless, casual NON-expert user is analysing the real-time chat history, carry out a two-step process as the user: 
first, a Reflection Phase, followed by a Response Generation Phase.

When simulating your process during the conversation:
You go through two internal steps each time:

1. Reflection Phase (internal thought):
Take a quick look at the current chat history. Think to yourself:
“Okay, what did the assistant just say or ask? What should I probably say next without overexplaining?”
Remember: you're not confident in how this system works, so don’t try to be precise.

2. Response Generation Phase (your reply):
Now write a short, casual message that gives only partial information based on what the assistant asked. Leave things unclear unless the assistant is persistent.


"""
    actual_system_prompt = user_proxy_agent._get_system_prompt()
    assert actual_system_prompt == expected_system_prompt


def test_get_chat_history_str_from_chat_history_single():
    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(message_str="Hello, what is the weather today"),
                agent_responses=[ResponseFromAgent(response_str="Hi, How can i help you?")],
            )
        ],
    )
    expected_chat_history_str = """[LLM-simulated user start]:
Hello, what is the weather today
[LLM-simulated user end]
[AI assistant start]:
Hi, How can i help you?
[AI assistant end]
"""
    actual_chat_history_str = UserProxyAgent._get_chat_history_str_from_chat_history(chat_history)
    assert actual_chat_history_str == expected_chat_history_str


def test_get_chat_history_str_from_chat_history_multiple():
    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(message_str="Hello, what is the weather today"),
                agent_responses=[ResponseFromAgent(response_str="Hi, How can i help you?")],
            ),
            ConversationTurn(
                id="test_id_2",
                user_message=UserProxyMessage(message_str="Can you tell me a joke?"),
                agent_responses=[
                    ResponseFromAgent(response_str="Sure! Why did the scarecrow win an award?"),
                    ResponseFromAgent(response_str="Because he was outstanding in his field!"),
                ],
            ),
        ],
    )
    expected_chat_history_str = """[LLM-simulated user start]:
Hello, what is the weather today
[LLM-simulated user end]
[AI assistant start]:
Hi, How can i help you?
[AI assistant end]
[LLM-simulated user start]:
Can you tell me a joke?
[LLM-simulated user end]
[AI assistant start]:
Sure! Why did the scarecrow win an award?
[AI assistant end]
[AI assistant start]:
Because he was outstanding in his field!
[AI assistant end]
"""
    actual_chat_history_str = UserProxyAgent._get_chat_history_str_from_chat_history(chat_history)
    assert actual_chat_history_str == expected_chat_history_str


def test_get_given_or_default_termination_condition():
    assert (
        UserProxyAgent._get_given_or_default_terminating_condition(
            [], TaskCompletedTerminatingCondition
        )
        == TaskCompletedTerminatingCondition()
    )
    assert (
        UserProxyAgent._get_given_or_default_terminating_condition(
            [], TaskDelegatedTerminatingCondition
        )
        == TaskDelegatedTerminatingCondition()
    )
    assert (
        UserProxyAgent._get_given_or_default_terminating_condition(
            [], TaskBlockedTerminatingCondition
        )
        == TaskBlockedTerminatingCondition()
    )


def test_get_user_message_reflection_prompt_single_terminating_condition_returns_expected_content():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[TerminatingCondition(check="This is a just check.")],
        agent_description="Test Agent Description",
        config={},
    )
    actual_prompt = user_proxy_agent._get_user_message_reflection_prompt(
        chat_history_str="This is chat history string."
    )
    assert (
        actual_prompt
        == f"""

---
The following [Chat History] (if available) provides context and indicates the CURRENT stage of your conversation as a LLM-simulated user with the AI assistant.
[Chat History]
This is chat history string.
---

Step 1: Reflection Phase

Given the [Chat History] REFLECT carefully on the AI assistant’s last response and what the LLM-simulated user is trying to accomplish based on the [user_task_summary].

Briefly address:
- Your role as the LLM-simulated user.
- The current stage of the conversation. You SHOULD NOT skip any user instructions as mentioned in the [user_task_summary].
- The assistant’s last reply in the [Chat History].

IMPORTANT CLARIFICATION:
- Review the entire [Chat History] and the [user_task_summary] and see what should be your next response as a LLM-simulated user.
- At times, the AI assistant’s last message may overlap with or anticipate a future user turn. In such cases, treat it strictly as the AI assistant response, not a replacement of the user message 

Do NOT generate the LLM-simulated user response yet. RESPOND only with a REFLECTION.
**IMPORTANT** remember your user persona as written in the system prompt (eg: expert user or non-expert) and respond with appropriate reflection.

TERMINATE ONLY IF the conversation is at its FINAL STAGE where the agent has completed all the tasks wanted by the user as shown in the [user_task_summary].
If the conversation has concluded, prepare to respond with {DEFAULT_STOP_SEQUENCE} in the next response generation phase.
Otherwise, DO NOT consider termination if the current conversation is not at its final stage.
        """
    )


def test_get_user_message_reflection_prompt_multi_terminating_condition_returns_expected_content():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[
            TaskCompletedTerminatingCondition(stop_sequence="STOP_DONE"),
            TaskDelegatedTerminatingCondition(stop_sequence="STOP_DELEGATED"),
            TaskBlockedTerminatingCondition(stop_sequence="STOP_BLOCKED"),
        ],
        agent_description="Test Agent Description",
        config={TERMINATING_CONDITION_MODE: MULTIPLE},
    )
    actual_prompt = user_proxy_agent._get_user_message_reflection_prompt(
        chat_history_str="This is chat history string."
    )
    assert (
        actual_prompt
        == """

---
The following [Chat History] (if available) provides context and indicates the CURRENT stage of your conversation as a LLM-simulated user with the AI assistant.
[Chat History]
This is chat history string.
---

Step 1: Reflection Phase

Given the [Chat History] REFLECT carefully on the AI assistant’s last response and what the LLM-simulated user is trying to accomplish based on the [user_task_summary].

Briefly address:
- Your role as the LLM-simulated user.
- The current stage of the conversation. You SHOULD NOT skip any user instructions as mentioned in the [user_task_summary].
- The assistant’s last reply in the [Chat History].

IMPORTANT CLARIFICATION:
- Review the entire [Chat History] and the [user_task_summary] and see what should be your next response as a LLM-simulated user.
- At times, the AI assistant’s last message may overlap with or anticipate a future user turn. In such cases, treat it strictly as the AI assistant response, not a replacement of the user message 

Do NOT generate the LLM-simulated user response yet. RESPOND only with a REFLECTION.
**IMPORTANT** remember your user persona as written in the system prompt (eg: expert user or non-expert) and respond with appropriate reflection.

TERMINATE ONLY IF the conversation is at its FINAL STAGE according to one of the stated [Termination Conditions].
If the conversation has concluded, prepare to respond with either the stop sequence STOP_DONE, STOP_DELEGATED, or STOP_BLOCKED in the next response generation phase.
Otherwise, DO NOT consider termination if the current conversation is not at its final stage.
"""
    )


def test_contains_stop_sequence_single():
    agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[
            TerminatingCondition(check="some check", stop_sequence="STOP_SEQUENCE")
        ],
        agent_description="Test Agent Description",
    )
    assert agent._contains_stop_sequence("STOP_SEQUENCE") == "STOP_SEQUENCE"
    assert agent._contains_stop_sequence("Some prefix STOP_SEQUENCE and suffix") == "STOP_SEQUENCE"
    assert agent._contains_stop_sequence("This is a response with no stop sequence") is None


def test_contains_stop_sequence_multi():
    agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[
            TaskCompletedTerminatingCondition(stop_sequence="STOP_DONE"),
            TaskDelegatedTerminatingCondition(stop_sequence="STOP_DELEGATED"),
            TaskBlockedTerminatingCondition(stop_sequence="STOP_BLOCKED"),
        ],
        agent_description="Test Agent Description",
        config={TERMINATING_CONDITION_MODE: MULTIPLE},
    )
    assert agent._contains_stop_sequence("STOP_DONE is included here") == "STOP_DONE"
    assert agent._contains_stop_sequence("STOP_DELEGATED") == "STOP_DELEGATED"
    assert agent._contains_stop_sequence("This is the message STOP_BLOCKED") == "STOP_BLOCKED"
    assert agent._contains_stop_sequence("This is a response with no stop sequence") is None


def test_get_user_proxy_reply_generation_prompt_single():
    agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[
            TerminatingCondition(check="some check", stop_sequence="STOP_SEQUENCE")
        ],
        agent_description="Test Agent Description",
    )
    actual = agent._get_user_proxy_reply_generation_prompt(
        chat_history_str="This is chat history string.",
        user_message_reflection="This is a just reflection.",
    )
    assert (
        actual
        == """

---
The following [Chat History] (if available) provides context and indicates the CURRENT stage of your conversation as a LLM-simulated user with the AI assistant.
[Chat History]
This is chat history string.
---

The following is the LLM-simulated user reflection.
[Reflection]
This is a just reflection.

---
Step 2: Response Generation Phase

Given the [Chat History] and [Reflection], GENERATE the LLM-simulated user NEXT RESPONSE that:

i) Naturally continues the conversation WITHOUT ADDING NEW TASK that is NOT found in the [user_task_summary]. You SHOULD NOT skip any tasks for the LLM-simulated user.
ii) Avoids revealing or repeating the AI assistant’s answers.
iv) Responds appropriately to the assistant’s actual reply, even if vague or off-track. If the AI assistant’s last message echoes or resembles any part of a user message, it’s the AI assistant response, NOT a new user turn. Note that suggestions or recommendations by the AI assistant should NEVER be MISTAKEN for actual actions taken.

GENERATE the LLM-simulated USER RESPONSE based on the [Reflection]. Return ONLY the LLM-simulated user response.

**IMPORTANT** remember your user persona as written in the system prompt (eg: expert user or non-expert) and respond with appropriate response.

TERMINATE ONLY IF the conversation is at its FINAL STAGE where the agent has completed all the tasks wanted by the user as shown in the [user_task_summary].
If the conversation has concluded, prepare to respond with STOP_SEQUENCE in the next response generation phase.
Otherwise, DO NOT consider termination if the current conversation is not at its final stage.
"""
    )


def test_get_user_proxy_reply_generation_prompt_multi():
    agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[
            TaskCompletedTerminatingCondition(stop_sequence="STOP_DONE"),
            TaskDelegatedTerminatingCondition(stop_sequence="STOP_DELEGATED"),
            TaskBlockedTerminatingCondition(stop_sequence="STOP_BLOCKED"),
        ],
        agent_description="Test Agent Description",
        config={TERMINATING_CONDITION_MODE: MULTIPLE},
    )
    actual = agent._get_user_proxy_reply_generation_prompt(
        chat_history_str="This is chat history string.",
        user_message_reflection="This is a just reflection.",
    )
    assert (
        actual
        == """

---
The following [Chat History] (if available) provides context and indicates the CURRENT stage of your conversation as a LLM-simulated user with the AI assistant.
[Chat History]
This is chat history string.
---

The following is the LLM-simulated user reflection.
[Reflection]
This is a just reflection.

---
Step 2: Response Generation Phase

Given the [Chat History] and [Reflection], GENERATE the LLM-simulated user NEXT RESPONSE that:

i) Naturally continues the conversation WITHOUT ADDING NEW TASK that is NOT found in the [user_task_summary]. You SHOULD NOT skip any tasks for the LLM-simulated user.
ii) Avoids revealing or repeating the AI assistant's answers.
iii) Responds appropriately to the assistant's actual reply, even if vague or off-track. If the AI assistant's last message echoes or resembles any part of a user message, it's the AI assistant response, NOT a new user turn. Note that suggestions or recommendations by the AI assistant should NEVER be MISTAKEN for actual actions taken.

GENERATE the LLM-simulated USER RESPONSE based on the [Reflection]. Return ONLY the LLM-simulated user response.

**IMPORTANT** remember your user persona as written in the system prompt (eg: expert user or non-expert) and respond with appropriate response.

TERMINATE ONLY IF the conversation is at its FINAL STAGE according to one of the stated [Termination Conditions].
If the conversation has concluded, prepare to respond with either the stop sequence STOP_DONE, STOP_DELEGATED, or STOP_BLOCKED in the next response generation phase.
Otherwise, DO NOT consider termination if the current conversation is not at its final stage.
"""
    )


def test_get_check_from_stop_sequence_single():
    agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[
            TerminatingCondition(check="some check", stop_sequence="STOP_SEQUENCE")
        ],
        agent_description="Test Agent Description",
    )
    assert agent._get_check_from_stop_sequence("STOP_SEQUENCE") == "some check"
    with pytest.raises(UserProxyError, match="does not match the defined terminating condition"):
        agent._get_check_from_stop_sequence("UNKNOWN_SEQUENCE")


def test_get_check_from_stop_sequence_multi_done():
    agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[
            TaskCompletedTerminatingCondition(check="check_done", stop_sequence="STOP_DONE"),
            TaskDelegatedTerminatingCondition(
                check="check_delegate", stop_sequence="STOP_DELEGATED"
            ),
            TaskBlockedTerminatingCondition(check="check_blocked", stop_sequence="STOP_BLOCKED"),
        ],
        agent_description="Test Agent Description",
        config={TERMINATING_CONDITION_MODE: MULTIPLE},
    )
    assert agent._get_check_from_stop_sequence("STOP_DONE") == "check_done"
    assert agent._get_check_from_stop_sequence("STOP_DELEGATED") == "check_delegate"
    assert agent._get_check_from_stop_sequence("STOP_BLOCKED") == "check_blocked"
    with pytest.raises(
        UserProxyError, match="does not match any of the defined terminating conditions"
    ):
        agent._get_check_from_stop_sequence("UNKNOWN_SEQUENCE")


def test_generate_message_from_chat_history_reflection_llm_call_error():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response = LLMResponse(
        status=HTTPStatus.INTERNAL_SERVER_ERROR,
        completion="",
        error_message="Internal Server Error",
    )

    mock_llm_client.make_request_with_payload.return_value = mock_llm_response
    terminating_conditions = [TerminatingCondition(check="This is a just check.")]
    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={},
    )

    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(message_str="Hello, what is the weather today"),
                agent_responses=[ResponseFromAgent(response_str="Hi, How can i help you?")],
            )
        ],
    )

    with pytest.raises(
        UserProxyError,
        match="Internal Code: 060010, Error Message: Unable to get user message reflection due to status: 500 from LLM client.",
    ):
        asyncio.run(user_proxy_agent.generate_message_from_chat_history(chat_history))


def test_generate_message_from_chat_history_200_ok_not_terminated():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response_reflection = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is a just reflection.",
        error_message=None,
    )

    mock_llm_response_reply = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is the user proxy message response.",
        error_message=None,
    )

    mock_llm_client.make_request_with_payload.side_effect = [
        mock_llm_response_reflection,
        mock_llm_response_reply,
    ]

    terminating_conditions = [TerminatingCondition(check="check 1")]

    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={},
    )

    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(
                    message_str="Hello, what is the weather today",
                ),
                agent_responses=[ResponseFromAgent(response_str="Hi, How can i help you?")],
            )
        ],
    )

    actual_user_proxy_message = asyncio.run(
        user_proxy_agent.generate_message_from_chat_history(chat_history)
    )
    expected_user_proxy_message = UserProxyMessage(
        message_str="This is the user proxy message response."
    )
    assert actual_user_proxy_message == expected_user_proxy_message
    assert mock_llm_client.make_request_with_payload.call_count == 2


def test_generate_message_from_chat_history_200_ok_terminated():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response_reflection = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is a just reflection.",
        error_message=None,
    )

    mock_llm_response_reply = LLMResponse(
        status=HTTPStatus.OK, completion=DEFAULT_STOP_SEQUENCE, error_message=None
    )

    mock_llm_client.make_request_with_payload.side_effect = [
        mock_llm_response_reflection,
        mock_llm_response_reply,
    ]

    terminating_conditions = [TerminatingCondition(check="This is just a check.")]
    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={},
    )

    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(message_str="Hello, what is the weather today"),
                agent_responses=[ResponseFromAgent(response_str="Hi, How can i help you?")],
            )
        ],
    )

    actual_user_proxy_message = asyncio.run(
        user_proxy_agent.generate_message_from_chat_history(chat_history)
    )
    assert DEFAULT_STOP_SEQUENCE in actual_user_proxy_message.message_str
    assert actual_user_proxy_message.check == "This is just a check."
    assert mock_llm_client.make_request_with_payload.call_count == 2


def test_generate_message_from_chat_history_200_ok_terminated_with_additional_message():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response_reflection = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is a just reflection.",
        error_message=None,
    )

    mock_llm_response_reply = LLMResponse(
        status=HTTPStatus.OK,
        completion=DEFAULT_STOP_SEQUENCE
        + " This is some additional message after the stop sequence.",
        error_message=None,
    )

    mock_llm_client.make_request_with_payload.side_effect = [
        mock_llm_response_reflection,
        mock_llm_response_reply,
    ]

    terminating_conditions = [
        TerminatingCondition(check="This is just a check for the user proxy terminating.")
    ]
    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={},
    )

    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(message_str="Hello, what is the weather today"),
                agent_responses=[ResponseFromAgent(response_str="Hi, How can i help you?")],
            )
        ],
    )

    actual_user_proxy_message = asyncio.run(
        user_proxy_agent.generate_message_from_chat_history(chat_history)
    )
    assert DEFAULT_STOP_SEQUENCE in actual_user_proxy_message.message_str
    assert actual_user_proxy_message.check == "This is just a check for the user proxy terminating."
    assert mock_llm_client.make_request_with_payload.call_count == 2


def test_generate_message_from_empty_chat_history():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[TerminatingCondition(check="This is just a check.")],
        agent_description="Test Agent Description",
        config={},
        initial_message="This is the initial message.",
    )

    actual_user_proxy_message = asyncio.run(
        user_proxy_agent.generate_message_from_chat_history(chat_history=None)
    )
    expected_user_proxy_message = UserProxyMessage(
        message_str="This is the initial message.",
    )
    assert actual_user_proxy_message == expected_user_proxy_message


def test_generate_message_from_chat_history_reflection_error():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response_reflection = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is a just reflection.",
        error_message=None,
    )

    mock_llm_response_reply = LLMResponse(
        status=HTTPStatus.INTERNAL_SERVER_ERROR,
        completion="",
        error_message="This is an internal server error.",
    )

    mock_llm_client.make_request_with_payload.side_effect = [
        mock_llm_response_reflection,
        mock_llm_response_reply,
    ]

    terminating_conditions = [TerminatingCondition(check="check 1")]

    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={},
    )

    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(message_str="Hello, what is the weather today"),
                agent_responses=[ResponseFromAgent(response_str="Hi, How can i help you?")],
            )
        ],
    )

    with pytest.raises(
        UserProxyError,
        match="Internal Code: 060011, Error Message: Unable to generate user proxy message due to status: 500 from LLM client.",
    ):
        asyncio.run(user_proxy_agent.generate_message_from_chat_history(chat_history))
