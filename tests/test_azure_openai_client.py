import asyncio
import os
from http import HTTPStatus
from unittest.mock import patch, MagicMock, Mock, AsyncMock

import pytest
import requests
from openai import RateLimitError

from agent_inspect.metrics.constants import (
    STATUS_200,
    STATUS_404,
    STATUS_429,
    MAX_RETRY_ATTEMPTS_EXCEEDED,
    STATUS_500,
)
from agent_inspect.models import LLMPayload

import httpx
from openai import APIStatusError

from agent_inspect.clients.azure_openai_client import (
    AzureOpenAIClient,
    backoff_handler,
    give_up_handler,
)


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_llm_request_successful_response(mock_azure_openai):
    mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content="Generated response"))])

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0.1)
    result = asyncio.run(client.make_llm_request("Test prompt"))

    assert result.status == STATUS_200
    assert result.completion == "Generated response"


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_llm_requests_api_status_error(mock_azure_openai):
    mock_response = httpx.Response(
        status_code=HTTPStatus.NOT_FOUND,
        request=httpx.Request("POST", "https://test.openai.azure.com"),
        json={"error": {"message": "Invalid input format"}},
    )

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.side_effect = APIStatusError(
        "This is a bad request", response=mock_response, body=None
    )
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0)
    prompt = "Prompt 1"
    result = asyncio.run(client.make_llm_request(prompt))

    assert result.status == STATUS_404
    assert result.completion == ""
    assert result.error_message == "Azure OpenAI API Error: This is a bad request"


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_llm_request_unknown_exception(mock_azure_openai):
    mock_client_instance = MagicMock()
    exception = Exception("Unknown error occurred")
    exception.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    mock_client_instance.chat.completions.create.side_effect = exception
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0)
    prompt = "Prompt causing unknown error"
    result = asyncio.run(client.make_llm_request(prompt))

    assert result.status == STATUS_500
    assert result.completion == ""
    assert result.error_message == "Unexpected error: Unknown error occurred"


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_llm_requests_successful_responses(mock_azure_openai):
    mock_response_1 = MagicMock(choices=[MagicMock(message=MagicMock(content="Response 1"))])
    mock_response_2 = MagicMock(choices=[MagicMock(message=MagicMock(content="Response 2"))])

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.side_effect = [
        mock_response_1,
        mock_response_2,
    ]
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0)
    prompts = ["Prompt 1", "Prompt 2"]
    results = asyncio.run(client.make_llm_requests(prompts))

    assert len(results) == 2
    assert results[0].status == STATUS_200
    assert results[0].completion == "Response 1"
    assert results[1].status == STATUS_200
    assert results[1].completion == "Response 2"


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_llm_requests_max_retry_reached(mock_azure_openai):
    mock_response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://test.openai.azure.com"),
        json={"error": {"message": "Rate limited!"}},
    )

    mock_client_instance = MagicMock()
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0)
    client.make_llm_request_with_retry = AsyncMock()
    client.make_llm_request_with_retry.side_effect = APIStatusError(
        "Rate limit exceeded", response=mock_response, body=None
    )

    prompts = ["Prompt causing rate limit error"]
    result = asyncio.run(client.make_llm_request(prompts[0]))

    assert result.status == STATUS_429
    assert result.completion == ""
    assert result.error_message == "Azure OpenAI API Error: Rate limit exceeded"


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_llm_requests_rate_limit_error(mock_azure_openai):
    mock_client_instance = MagicMock()
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0)
    client.make_llm_request_with_retry = AsyncMock()
    client.make_llm_request_with_retry.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=Mock(status_code=429, request=Mock()),
        body=None,
    )

    prompts = ["Prompt causing rate limit error"]
    result = asyncio.run(client.make_llm_request(prompts[0]))

    assert result.status == STATUS_429
    assert result.completion == ""
    assert result.error_message == MAX_RETRY_ATTEMPTS_EXCEEDED


@patch("agent_inspect.clients.azure_openai_client.logger")
def test_backoff_handler(mock_logger):
    """Test backoff handler logs warning correctly"""
    details = {
        "target": Mock(__name__="test_function"),
        "args": ("arg1",),
        "kwargs": {"key": "value"},
        "wait": 2.5,
        "tries": 3,
        "elapsed": 5.2,
        "exception": Exception("Test exception"),
    }

    backoff_handler(details)

    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args[0][0]
    assert "Backing off test_function" in call_args
    assert "2.5s" in call_args
    assert "tries=3" in call_args


@patch("agent_inspect.clients.azure_openai_client.logger")
def test_give_up_handler(mock_logger):
    """Test give up handler logs error correctly"""
    details = {
        "target": Mock(__name__="test_function"),
        "args": ("arg1",),
        "kwargs": {"key": "value"},
        "elapsed": 300.5,
        "exception": Exception("Final exception"),
    }

    give_up_handler(details)

    mock_logger.error.assert_called_once()
    call_args = mock_logger.error.call_args[0][0]
    assert "Max retries reached" in call_args
    assert "test_function" in call_args
    assert "300.5s" in call_args
    assert "Final exception" in call_args


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_convert_payload_to_raw_request_with_all_fields(mock_azure_openai):
    client = AzureOpenAIClient("default-model", 1000, 0.5)

    payload = LLMPayload(
        user_prompt="What is AI?",
        model="custom-model",
        system_prompt="You are a helpful assistant.",
        temperature=0.8,
        max_tokens=2000,
        structured_output={"type": "json_object"},
    )

    raw_request = client.convert_payload_to_raw_request(payload)

    assert raw_request["model"] == "custom-model"
    assert raw_request["temperature"] == 0.8
    assert raw_request["max_tokens"] == 2000
    assert len(raw_request["messages"]) == 2
    assert raw_request["messages"][0] == {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    assert raw_request["messages"][1] == {"role": "user", "content": "What is AI?"}
    assert raw_request["response_format"] == {"type": "json_object"}


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_request_with_payload_successful(mock_azure_openai):
    """Test successful request with payload using mocked make_request_with_retry"""
    mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content="AI response"))])
    mock_client_instance = MagicMock()
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0.1)

    payload = LLMPayload(user_prompt="Test question")

    with patch.object(
        client,
        "make_request_with_payload_using_retry",
        new=AsyncMock(return_value=mock_response),
    ) as mock_retry:
        result = asyncio.run(client.make_request_with_payload(payload))

    assert result.status == STATUS_200
    assert result.completion == "AI response"
    mock_retry.assert_awaited_once_with(payload)


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_request_with_payload_using_retry_successful(mock_azure_openai):
    mock_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Successful payload response"))]
    )
    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0.1)
    payload = LLMPayload(user_prompt="Valid payload")

    result = asyncio.run(client.make_request_with_payload_using_retry(payload))

    assert result.choices[0].message.content == "Successful payload response"


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_request_with_payload_using_retry_non_retryable_error(mock_azure_openai):
    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.side_effect = APIStatusError(
        "Non-retryable error", response=Mock(status_code=400), body=None
    )
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0.1)
    payload = LLMPayload(user_prompt="Payload causing non-retryable error")

    with pytest.raises(APIStatusError, match="Non-retryable error"):
        asyncio.run(client.make_request_with_payload_using_retry(payload))


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_request_with_payload_using_retry_transient_error(mock_azure_openai):
    mock_client_instance = MagicMock()
    mock_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Successful payload response"))]
    )
    mock_client_instance.chat.completions.create.side_effect = [
        requests.exceptions.Timeout("Timeout occurred"),
        requests.exceptions.Timeout("Timeout occurred"),
        requests.exceptions.Timeout("Timeout occurred"),
        mock_response,
    ]
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0.1)
    payload = LLMPayload(user_prompt="Payload causing transient error resolved after 3 retries")

    result = asyncio.run(client.make_request_with_payload_using_retry(payload))
    assert result.choices[0].message.content == "Successful payload response"


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_request_with_payload_successful_response(mock_azure_openai):
    mock_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Successful payload response"))]
    )
    mock_client_instance = MagicMock()
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0.1)
    payload = LLMPayload(user_prompt="Valid payload")

    with patch.object(
        client,
        "make_request_with_payload_using_retry",
        new=AsyncMock(return_value=mock_response),
    ):
        result = asyncio.run(client.make_request_with_payload(payload))

    assert result.status == STATUS_200
    assert result.completion == "Successful payload response"


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_request_with_payload_rate_limit_error(mock_azure_openai):
    mock_client_instance = MagicMock()
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0.1)
    payload = LLMPayload(user_prompt="Rate limit test")

    with patch.object(
        client,
        "make_request_with_payload_using_retry",
        new=AsyncMock(
            side_effect=RateLimitError(
                message="Rate limit exceeded",
                response=Mock(status_code=429, request=Mock()),
                body=None,
            )
        ),
    ):
        result = asyncio.run(client.make_request_with_payload(payload))

    assert result.status == STATUS_429
    assert result.completion == ""
    assert result.error_message == MAX_RETRY_ATTEMPTS_EXCEEDED


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_request_with_payload_api_status_error(mock_azure_openai):
    mock_client_instance = MagicMock()
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0.1)
    payload = LLMPayload(user_prompt="API status error test")

    with patch.object(
        client,
        "make_request_with_payload_using_retry",
        new=AsyncMock(
            side_effect=APIStatusError("API error", response=Mock(status_code=403), body=None)
        ),
    ):
        result = asyncio.run(client.make_request_with_payload(payload))

    assert result.status == 403
    assert result.completion == ""
    assert result.error_message == "API error"


@patch.dict(
    os.environ,
    {
        "AZURE_API_VERSION": "2024-02-01",
        "AZURE_API_BASE": "https://test.openai.azure.com",
        "AZURE_API_KEY": "test-key",
    },
)
@patch("agent_inspect.clients.azure_openai_client.AzureOpenAI")
def test_make_request_with_payload_unexpected_error(mock_azure_openai):
    mock_client_instance = MagicMock()
    mock_azure_openai.return_value = mock_client_instance

    client = AzureOpenAIClient("test-model", 100, 0.1)
    payload = LLMPayload(user_prompt="Unexpected error test")

    with patch.object(
        client,
        "make_request_with_payload_using_retry",
        new=AsyncMock(side_effect=Exception("Unexpected error")),
    ):
        result = asyncio.run(client.make_request_with_payload(payload))

    assert result.status == STATUS_500
    assert result.completion == ""
    assert result.error_message == "Unexpected error"
