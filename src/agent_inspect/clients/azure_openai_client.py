import os
import requests
import asyncio
import backoff

from openai import AzureOpenAI, APIStatusError, RateLimitError
from typing import Any, Dict
import logging

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.models.llm_response import LLMResponse

from agent_inspect.metrics.constants import STATUS_200, STATUS_500, STATUS_429, MAX_RETRY_ATTEMPTS_EXCEEDED

logger = logging.getLogger(__name__)


def backoff_handler(details):
    """
    Custom handler for backoff events to log messages with extra attributes.
    """
    message = (
        f"Backing off {details['target'].__name__} "
        f"(args={details['args']}, kwargs={details['kwargs']}) "
        f"for {details['wait']:.1f}s "
        f"(tries={details['tries']}, elapsed={details['elapsed']:.1f}s, "
        f"exception: {details['exception']})"
    )

    logger.warning(message)


def give_up_handler(details):
    """
    Custom handler for when max retries are reached.
    """
    message = (
        f"Max retries reached for {details['target'].__name__} "
        f"(args={details['args']}, kwargs={details['kwargs']}, "
        f"elapsed={details['elapsed']:.1f}s, exception: {details['exception']})"
    )
    logger.error(message)


class AzureOpenAIClient(LLMClient):
    """
    Client class providing connection to the Azure OpenAI Service. Need to set the following environment variables: ``AZURE_API_VERSION``, ``AZURE_API_BASE``, ``AZURE_API_KEY``.

    :param model: the selected Azure OpenAI model which will receive the prompt. This is the deployment name in Azure.
    :param max_tokens: the maximum number of tokens allowed for the LLM to generate.
    :param temperature: the temperature setting for LLM model. Default to ``0``.
    """

    def __init__(self, model: str, max_tokens: int, temperature: float = 0):
        self.model = model
        self.max_tokens = max_tokens
        self.chat_client = AzureOpenAI(
            api_version=os.environ['AZURE_API_VERSION'],
            azure_endpoint=os.environ['AZURE_API_BASE'],
            api_key=os.environ['AZURE_API_KEY'],
        )
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo,
        (APIStatusError, RateLimitError, requests.exceptions.Timeout, requests.exceptions.ConnectionError,
         requests.exceptions.RequestException),
        max_tries=10,
        max_time=300,
        jitter=None,
        on_backoff=backoff_handler,
        on_giveup=give_up_handler,
        giveup=lambda e: isinstance(e, APIStatusError) and e.status_code in [400, 401, 403, 404]
    )
    async def make_llm_request_with_retry(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        azure_openai_response = self.chat_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return azure_openai_response


    async def make_llm_request(self, prompt: str) -> LLMResponse:
        """
        Returns LLM completion after sending a prompt to the selected the model.
        Uses an exponential backoff retry mechanism for transient failures.

        :param prompt: the provided prompt to send to the model.
        :return: :obj:`~agent_inspect.models.llm_response.LLMResponse` object containing status code, completion and error message.
        """

        # Backoff decorator handles retries; this try-catch handles final failures
        try:
            azure_openai_response = await self.make_llm_request_with_retry(prompt)
            return LLMResponse(status=STATUS_200, completion=azure_openai_response.choices[0].message.content)
        except RateLimitError:
            return LLMResponse(status=STATUS_429, completion="", error_message=MAX_RETRY_ATTEMPTS_EXCEEDED)
        except APIStatusError as e:
            # Non-retryable errors (4xx) or final failure after all retries
            return LLMResponse(
                status=e.status_code,
                completion="",
                error_message=f"Azure OpenAI API Error: {e.message}"
            )
        except Exception as e:
            # Catch any other unexpected errors
            return LLMResponse(
                status=STATUS_500,
                completion="",
                error_message=f"Unexpected error: {str(e)}"
            )

    async def make_llm_requests(self, prompts: list[str]) -> list[LLMResponse]:
        """
        Returns LLM completion after sending a prompt to the selected the model.

        :param prompts: a list of provided prompts to send to the model.
        :return: a :obj:`~typing.List` [:obj:`~agent_inspect.models.llm_response.LLMResponse`] object containing status codes, completions and error messages.
        """

        responses = await asyncio.gather(*(self.make_llm_request(prompt) for prompt in prompts))
        return list(responses)
    
    def convert_payload_to_raw_request(self, payload: LLMPayload) -> Dict[str, Any]:
        raw_request: Dict[str, Any] = {}
        messages = []
        if payload.system_prompt:
            messages.append({"role": "system", "content": payload.system_prompt})
        messages.append({"role": "user", "content": payload.user_prompt})
        raw_request["model"] = payload.model if payload.model else self.model
        raw_request["messages"] = messages
        raw_request["temperature"] = payload.temperature if payload.temperature else self.temperature
        raw_request["max_tokens"] = payload.max_tokens if payload.max_tokens else self.max_tokens
        if payload.structured_output:
            raw_request["response_format"] = payload.structured_output
        return raw_request

    @backoff.on_exception(
        backoff.expo,
        (APIStatusError, requests.exceptions.Timeout, requests.exceptions.ConnectionError,
         requests.exceptions.RequestException),
        max_tries=10,
        max_time=300,
        jitter=None,
        on_backoff=backoff_handler,
        on_giveup=give_up_handler,
        giveup=lambda e: isinstance(e, APIStatusError) and e.status_code in [400, 401, 403, 404]
    )
    async def make_request_with_payload_using_retry(self, payload: LLMPayload):
        raw_request = self.convert_payload_to_raw_request(payload)
        azure_response = self.chat_client.chat.completions.create(**raw_request)
        return azure_response
    
    async def make_request_with_payload(self, payload: LLMPayload) -> LLMResponse:
        """
        Returns LLM completion after sending a payload to the selected the model.

        :param payload: the provided payload to send to the model.
        :return: :obj:`~agent_inspect.models.llm_response.LLMResponse` object containing status code, completion and error message.
        """
        try:
            azure_response = await self.make_request_with_payload_using_retry(payload)
            response = LLMResponse(status=STATUS_200, completion=azure_response.choices[0].message.content)
        except RateLimitError:
            response = LLMResponse(status=STATUS_429, completion="", error_message=MAX_RETRY_ATTEMPTS_EXCEEDED)
        except APIStatusError as e:
            response = LLMResponse(status=e.status_code, completion="", error_message=e.message)
        except Exception as e:
            response = LLMResponse(status=STATUS_500, completion="", error_message=str(e))
        return response
