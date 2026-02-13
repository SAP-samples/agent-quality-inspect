from abc import ABC, abstractmethod
from agent_inspect.models.llm_response import LLMResponse
from agent_inspect.models.llm_payload import LLMPayload

class LLMClient(ABC):
    """
    This is a base abstract class that should be extended for actual implementations to connect to llm-as-a-judge model.
    """

    @abstractmethod
    async def make_llm_request(self, prompt: str) -> LLMResponse:
        """
        This is an abstract method and should be implemented for concrete class to make LLM request to the LLM model.

        :param prompt: the user provided prompt to send to the model.
        :return: :obj:`~agent_inspect.models.llm_response.LLMResponse` object containing status code, completion and error message.
        """
        ...

    @abstractmethod
    async def make_llm_requests(self, prompts: list[str]) -> list[LLMResponse]:
        """
        This is an abstract method and should be implemented for concrete class to make multiple LLM requests to the LLM model.

        :param prompts: the user provided prompts to send to the model.
        :return: a :obj:`~typing.List` [:obj:`~agent_inspect.models.llm_response.LLMResponse`] object containing status codes, completions and error messages.
        """
        ...

    @abstractmethod
    async def make_request_with_payload(self, payload: LLMPayload) -> LLMResponse:
        """
        This is an abstract method and should be implemented for concrete class to make LLM request to the LLM model with LLMPayload.

        :param payload: the user provided LLMPayload to send to the model.
        :return: :obj:`~agent_inspect.models.llm_response.LLMResponse` object containing status code, completion and error message.
        """
        ...
