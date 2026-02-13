from http import HTTPStatus
from typing import Optional, Dict, List, Any

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.metrics.constants import USE_EXPERT_AGENT, \
    USE_EXPERT_AGENT_DEFAULT, DEFAULT_STOP_SEQUENCE
from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import UserProxyError
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.metrics.utils.metrics_utils import get_config_or_default
from agent_inspect.models.user_proxy.chat import ChatHistory, UserProxyMessage
from agent_inspect.models.user_proxy.terminating_condition import TerminatingCondition
from agent_inspect.user_proxy.utils import ensure_full_stop
from agent_inspect.user_proxy.llm_proxy_agent import LLMProxyAgent
from agent_inspect.user_proxy.templates import USER_PROXY_REFLECTION_GEN_TEMPLATE, USER_PROXY_REPLY_GEN_TEMPLATE, \
    EXPERT_PERSONA_TEMPLATE, NONEXPERT_PERSONA_TEMPLATE
from agent_inspect.metrics.utils.user_proxy_validators import UserProxyInputValidator


class UserProxyAgent(LLMProxyAgent):
    """
    User proxy (a.k.a. simulated user) class which generates the user utterances during a dynamic conversation with the AI agent. The dynamic user utterances are generated based on the user task instruction and user persona (e.g., expert or non-expert) prompt templates via a two-step process—reflection followed by response generation. 

    :param llm_client: the connection to the llm client for user utterances generation.
    :param task_summary: a user task instruction describing the summary of the task user wants the AI agent to complete.
    :param terminating_conditions: a :obj:`~typing.List` [:obj:`~agent_inspect.models.user_proxy.terminating_condition.TerminatingCondition`] object where each element in the list is a terminating condition for the user proxy to exit the user-agent conversation early. Currently supports only one terminating condition.
    :param agent_description: the description of the AI agent that will interact with the user proxy, provided as additional context for the user proxy. Default to empty string.
    :param initial_message: a static message that is used as the user proxy's initial message (if available) to the AI agent. Default to empty string.
    :param config: Default to ``None``. Configuration options:

        - **use_expert_agent**: a :obj:`~typing.bool` flag to indicate whether the user proxy should use an expert persona; otherwise, it uses a non-expert persona. Default to ``True`` to use an expert persona.

    """

    def __init__(
            self,
            llm_client: LLMClient,
            task_summary: str,
            terminating_conditions: List[TerminatingCondition],
            agent_description: str = "",
            initial_message: str = "",
            config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(llm_client, config)

        UserProxyInputValidator.validate_task_summary(task_summary)
        UserProxyInputValidator.validate_terminating_condition(terminating_conditions)

        self.initial_message = initial_message
        self.task_summary = ensure_full_stop(task_summary)
        self.terminating_conditions = terminating_conditions
        self.agent_description = agent_description
        self.is_expert = get_config_or_default(config=self.config, config_key=USE_EXPERT_AGENT, default=USE_EXPERT_AGENT_DEFAULT)


    def get_system_prompt(self) -> str:
        if self.is_expert:
            return EXPERT_PERSONA_TEMPLATE.format(
                task_summary=self.task_summary,
                check=self.terminating_conditions[0].check,
                agent_desc=self.agent_description
            )
        else:
            return NONEXPERT_PERSONA_TEMPLATE.format(
                task_summary=self.task_summary,
                check=self.terminating_conditions[0].check,
                agent_desc=self.agent_description
            )


    @staticmethod
    def get_chat_history_str_from_chat_history(chat_history: ChatHistory) -> str:
        chat_history_str = ""
        if not UserProxyAgent.contains_chat_history(chat_history):
            return chat_history_str
        for past_conversation in chat_history.conversations:
            chat_history_str += f"[LLM-simulated user start]:\n{past_conversation.user_message.message_str.strip()}\n[LLM-simulated user end]\n" if past_conversation.user_message.message_str else ""
            for agent_response in past_conversation.agent_responses:
                if agent_response:
                    chat_history_str += f"[AI assistant start]:\n{agent_response.response_str.strip()}\n[AI assistant end]\n" if agent_response.response_str else ""
        return chat_history_str


    @staticmethod
    def contains_chat_history(chat_history: Optional[ChatHistory]) -> bool:
        return chat_history is not None and bool(chat_history.conversations)


    async def get_user_message_reflection(self, chat_history_str: str, stop_sequence: str, system_prompt: str) -> str:
        user_message_reflection_prompt = USER_PROXY_REFLECTION_GEN_TEMPLATE.format(
            chat_history_str=chat_history_str,
            stop_sequence=stop_sequence
        )
        llm_response = await self.llm_client.make_request_with_payload(
            LLMPayload(
                user_prompt=user_message_reflection_prompt,
                system_prompt=system_prompt
            )
        )

        if llm_response.status == HTTPStatus.OK:
            return llm_response.completion.strip()
        raise UserProxyError(internal_code=ErrorCode.INVALID_USER_MESSAGE_REFLECTION.value, message=f"Unable to get user message reflection due to status: {llm_response.status} from LLM client.")

    @staticmethod
    def _contains_stop_sequence(user_proxy_message_str: str) -> bool:
        return DEFAULT_STOP_SEQUENCE in user_proxy_message_str

    async def generate_message_from_chat_history(self, chat_history: Optional[ChatHistory]) -> UserProxyMessage:
        """
        Generates the next user utterance given a :obj:`~agent_inspect.models.user_proxy.chat.ChatHistory` object containing the user-agent conversation history as input.

        :param chat_history: a :obj:`~agent_inspect.models.user_proxy.chat.ChatHistory` object containing the user-agent conversation history. 
        :return: a :obj:`~agent_inspect.models.user_proxy.chat.UserProxyMessage` object containing the next user utterance. For the first user utterance, if the input variable ``chat_history`` is ``None`` and ``self.initial_message`` is not an empty string,  the method returns the ``self.initial_message`` as the initial user utterance, otherwise it generates the user utterance based on the conversation history. All subsequent user utterances are generated based on the conversation history.

        Example:

        >>> from agent_inspect.user_proxy import UserProxyAgent
        >>> from agent_inspect.models.user_proxy import ChatHistory TerminatingCondition
        >>> from agent_inspect.metrics.constants import USE_EXPERT_AGENT
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>> from uuid import uuid4
        >>> import asyncio
        >>>
        >>> user_instruct, term_condition = load_user_instruct_term(sample_path) # Load user instruction and terminating condition
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create llm client for user proxy
        >>> user = UserProxyAgent(
        ...     llm_client=client,
        ...     task_summary=user_instruct,
        ...     terminating_conditions=[
        ...         TerminatingCondition(check=term_condition)
        ...     ],
        ...     config={USE_EXPERT_AGENT: True}
        ... )
        >>> chat_history = ChatHistory(id=str(uuid4()), conversations=[]) # start from an empty conversation
        >>> user_response = asyncio.run(user.generate_message_from_chat_history(chat_history))
        >>> print(user_response.message_str)
        """


        if (not UserProxyAgent.contains_chat_history(chat_history)) and self.initial_message:
            return UserProxyMessage(
                message_str=self.initial_message
            )

        system_prompt = self.get_system_prompt()
        chat_history_str = self.get_chat_history_str_from_chat_history(chat_history)
        stop_sequence = DEFAULT_STOP_SEQUENCE
        user_message_reflection = await self.get_user_message_reflection(chat_history_str, stop_sequence, system_prompt)

        user_proxy_reply_gen_prompt = USER_PROXY_REPLY_GEN_TEMPLATE.format(
            chat_history_str=chat_history_str,
            user_message_reflection=user_message_reflection,
            stop_sequence=stop_sequence
        )

        llm_response = await self.llm_client.make_request_with_payload(
            LLMPayload(
                user_prompt=user_proxy_reply_gen_prompt,
                system_prompt=system_prompt
            )
        )

        if llm_response.status != HTTPStatus.OK:
            raise UserProxyError(internal_code=ErrorCode.INVALID_USER_PROXY_RESPONSE.value,
                                 message=f"Unable to generate user proxy message due to status: {llm_response.status} from LLM client.")

        user_proxy_message_str = llm_response.completion.strip()
        is_terminated = self._contains_stop_sequence(user_proxy_message_str)

        user_proxy_message = UserProxyMessage(
            message_str=user_proxy_message_str,
            check=self.terminating_conditions[0].check if is_terminated else None
        )
        return user_proxy_message
