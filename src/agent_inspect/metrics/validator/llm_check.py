from typing import Dict, Callable

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.metrics.constants import STATUS_200
from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import EvaluationError
from agent_inspect.models.llm_response import LLMResponse

async def llm_check(client: LLMClient, variables: Dict, template: str, post_process: Callable[[LLMResponse], bool]) -> bool:
    prompt = template.format(**variables)
    response = await client.make_llm_request(prompt)
    if response.status != STATUS_200:
        raise EvaluationError(internal_code=ErrorCode.INVALID_LLM_JUDGE_RESULT_ERROR.value,
                              message=f"LLM request failed with status: {response.status}, "
                                      f"error message: {response.error_message}")
    return post_process(response)
