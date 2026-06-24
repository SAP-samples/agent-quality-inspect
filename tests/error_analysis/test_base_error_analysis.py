"""
Unit tests for ErrorAnalysis.
Tests cover:
- validate_user_defined_clusters: cluster validation logic
- _request_and_parse_json_with_retry: LLM request retry logic
- __init__: configuration management
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_inspect.clients import LLMClient
from agent_inspect.metrics.constants import STATUS_200
from agent_inspect.exception import ToolError
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.models.llm_response import LLMResponse
from agent_inspect.tools.error_analysis.base_error_analysis import ErrorAnalysis


class ConcreteErrorAnalysis(ErrorAnalysis):
    """Concrete subclass for testing abstract ErrorAnalysis."""

    async def analyze_batch_async(self, data_samples):
        return None


@pytest.fixture
def analysis():
    return ConcreteErrorAnalysis()


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.make_request_with_payload = AsyncMock()
    return client


# ==================== Tests for __init__ ====================


def test_init_with_default_config():
    analysis = ConcreteErrorAnalysis()
    assert analysis.config == {}


def test_init_with_custom_max_workers():
    analysis = ConcreteErrorAnalysis(config={"max_workers": 5})
    assert analysis.config == {"max_workers": 5}


# ==================== Tests for _request_and_parse_json_with_retry ====================


@pytest.mark.asyncio
async def test_request_and_parse_json_returns_parsed_response_on_success(analysis, mock_llm_client):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=STATUS_200, completion=json.dumps({"key": "value"}))

    mock_llm_client.make_request_with_payload.return_value = mock_response
    result = await analysis._request_and_parse_json_with_retry(mock_llm_client, payload)

    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_request_and_parse_json_raises_tool_error_on_non_200_status(
    analysis, mock_llm_client
):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=500, completion=None, error_message="Error")

    mock_llm_client.make_request_with_payload.return_value = mock_response

    with pytest.raises(ToolError, match="LLM request failed with status 500 and error: Error"):
        await analysis._request_and_parse_json_with_retry(mock_llm_client, payload)


@pytest.mark.asyncio
async def test_request_and_parse_json_retries_on_json_decode_error(analysis, mock_llm_client):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=STATUS_200, completion="invalid json")
    mock_llm_client.make_request_with_payload.return_value = mock_response

    with pytest.raises(ToolError, match="Maximum retry attempts exceeded for JSON decode error."):
        await analysis._request_and_parse_json_with_retry(mock_llm_client, payload)


@pytest.mark.asyncio
async def test_request_and_parse_json_logs_warning_on_decode_error(
    analysis, mock_llm_client, caplog
):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=STATUS_200, completion="invalid json")
    mock_llm_client.make_request_with_payload.return_value = mock_response

    with pytest.raises(ToolError):
        await analysis._request_and_parse_json_with_retry(mock_llm_client, payload)

    assert "JSON decode error on attempt" in caplog.text


@pytest.mark.asyncio
async def test_request_and_parse_json_raises_when_completion_is_none(analysis, mock_llm_client):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=STATUS_200, completion=None)

    mock_llm_client.make_request_with_payload.return_value = mock_response

    with pytest.raises(
        ToolError,
        match="Internal Code: 080014, Error Message: Maximum retry attempts exceeded for JSON decode error.",
    ):
        await analysis._request_and_parse_json_with_retry(mock_llm_client, payload)


@pytest.mark.asyncio
async def test_request_and_parse_json_succeeds_on_third_attempt(analysis, mock_llm_client):
    """Test that retry succeeds after 2 failed attempts with JSON decode errors on the 3rd attempt."""
    payload = LLMPayload(user_prompt="prompt", structured_output={})

    invalid_response = LLMResponse(status=STATUS_200, completion="invalid json")
    valid_response = LLMResponse(status=STATUS_200, completion=json.dumps({"key": "value"}))

    mock_llm_client.make_request_with_payload.side_effect = [
        invalid_response,
        invalid_response,
        valid_response,
    ]

    result = await analysis._request_and_parse_json_with_retry(mock_llm_client, payload)

    assert result == {"key": "value"}
    assert mock_llm_client.make_request_with_payload.call_count == 3


@pytest.mark.asyncio
async def test_request_and_parse_json_succeeds_on_fourth_attempt(analysis, mock_llm_client):
    """Test that retry succeeds after 3 failed attempts with JSON decode errors on the 4th attempt."""
    payload = LLMPayload(user_prompt="prompt", structured_output={})

    invalid_response = LLMResponse(status=STATUS_200, completion="invalid json")
    valid_response = LLMResponse(status=STATUS_200, completion=json.dumps({"result": "success"}))

    mock_llm_client.make_request_with_payload.side_effect = [
        invalid_response,
        invalid_response,
        invalid_response,
        valid_response,
    ]

    result = await analysis._request_and_parse_json_with_retry(mock_llm_client, payload)

    assert result == {"result": "success"}
    assert mock_llm_client.make_request_with_payload.call_count == 4


@pytest.mark.asyncio
async def test_request_and_parse_json_exhausts_all_retries_and_fails(
    analysis, mock_llm_client, caplog
):
    """Test that retry exhausts all attempts and raises ToolError with 'Maximum retry attempts exceeded' message."""
    payload = LLMPayload(user_prompt="prompt", structured_output={})

    invalid_response = LLMResponse(status=STATUS_200, completion="invalid json")
    mock_llm_client.make_request_with_payload.return_value = invalid_response

    with pytest.raises(ToolError, match="Maximum retry attempts exceeded for JSON decode error."):
        await analysis._request_and_parse_json_with_retry(mock_llm_client, payload)

    # Verify that all retry attempts were made (default is 5 from MAX_RETRY_JSON_DECODE_ERROR constant)
    assert mock_llm_client.make_request_with_payload.call_count == 5

    # Verify that warnings were logged for each failed attempt
    assert "JSON decode error on attempt 1/5" in caplog.text
    assert "JSON decode error on attempt 2/5" in caplog.text
    assert "JSON decode error on attempt 3/5" in caplog.text
    assert "JSON decode error on attempt 4/5" in caplog.text
    assert "JSON decode error on attempt 5/5" in caplog.text
