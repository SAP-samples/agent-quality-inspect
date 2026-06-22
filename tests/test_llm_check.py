from http import HTTPStatus
from unittest.mock import AsyncMock

import pytest

from agent_inspect.exception.error_codes import ErrorCode, EvaluationComponent
from agent_inspect.exception import EvaluationError
from agent_inspect.models import LLMResponse
from agent_inspect.metrics.validator import llm_check


@pytest.mark.asyncio
async def test_llm_check_returns_true_for_successful_response():
    mock_client = AsyncMock()
    mock_response = LLMResponse(status=HTTPStatus.OK, completion="dummy completion")
    mock_client.make_llm_request.return_value = mock_response

    def post_process(llm_response):
        return True

    result = await llm_check(mock_client, {"key": "value"}, "template {key}", post_process)
    assert result is True


@pytest.mark.asyncio
async def test_llm_check_raises_error_for_non_200_status():
    mock_client = AsyncMock()
    mock_response = LLMResponse(
        status=HTTPStatus.INTERNAL_SERVER_ERROR, error_message="Internal Server Error"
    )
    mock_client.make_llm_request.return_value = mock_response

    def post_process(llm_response):
        return True

    with pytest.raises(EvaluationError) as exc_info:
        await llm_check(mock_client, {"key": "value"}, "template {key}", post_process)
    assert (
        exc_info.value.internal_code
        == EvaluationComponent.EVALUATION_ERROR_CODE.value
        + ErrorCode.INVALID_LLM_JUDGE_RESULT_ERROR.value
    )
