"""
Error Analysis Module - Class Hierarchy
========================================

This module defines the base class for error analysis implementations.

Class Hierarchy Diagram:

.. code-block:: text

                                    ErrorAnalysis (ABC)
                                          |
                                          | (abstract base)
                    ______________________|_______________________
                   |                                              |
            ToolCallErrorAnalysis (ABC)              SubgoalErrorAnalysis (ABC)
            (tool call errors)                       (subgoal errors)
                   |                                              |
         __________|__________                           _________|_________
        |                     |                         |                   |
    SemisupervisedTool      DeterministicTool       UnsupervisedSubgoal     SemisupervisedSubgoal
    CallErrorAnalysis       CallErrorAnalysis       ErrorAnalysis           ErrorAnalysis
    (LLM-based              (Regex-based            (Dynamic clustering     (Predefined clusters
    classification)         classification)         with LLM)               with LLM)


Algorithm Types:

1. **Unsupervised**: Discovers error patterns dynamically without predefined categories
   - Uses LLM for error summarization and semantic clustering
   - Generates error categories from data

2. **Semi-supervised**: Classifies errors into predefined error clusters
   - Uses LLM with structured prompts for classification
   - More consistent and interpretable results
   - Supports custom error cluster definitions

3. **Deterministic**: Rule-based classification using regex patterns
   - No LLM required (faster, cheaper)
   - Works only for tool call errors with specific explanation formats
"""

import asyncio
import json
import logging
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import ToolError
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.metrics.constants import (
    STATUS_200,
    MAX_RETRY_JSON_DECODE_ERROR,
    MAX_WORKERS_KEY,
    DEFAULT_MAX_WORKERS,
)
from agent_inspect.core.utils import get_config_or_default


class ErrorAnalysis(ABC):
    """Abstract base class for all error analysis implementations.

    This class provides common functionality for error analysis operations:

    - Configuration management via config dictionary
    - Thread pool executor for concurrent processing
    - Retry logic for LLM requests with JSON decode errors

    :param config: Optional configuration dictionary. Supported keys:

        - **max_workers**: Maximum number of concurrent workers (default: 20)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        max_workers = get_config_or_default(self.config, MAX_WORKERS_KEY, DEFAULT_MAX_WORKERS)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def _request_and_parse_json_with_retry(
        self, llm_client: LLMClient, payload: LLMPayload
    ) -> Any:
        """Make an LLM request with retry logic for JSON decode errors."""
        max_retry_json_decode_error = MAX_RETRY_JSON_DECODE_ERROR
        attempt = 0
        while attempt < max_retry_json_decode_error:
            try:
                # Run blocking LLM call in thread pool to enable concurrent execution
                response = await asyncio.to_thread(
                    lambda: asyncio.run(llm_client.make_request_with_payload(payload))
                )
                if response.status != STATUS_200:
                    raise ToolError(
                        internal_code=ErrorCode.CLIENT_REQUEST_ERROR.value,
                        message=f"LLM request failed with status {response.status} and error: {response.error_message}",
                    )
                if not response.completion:
                    logging.warning(
                        f"JSON decode error on attempt {attempt + 1}/{max_retry_json_decode_error}: "
                        f"Empty completion received."
                    )
                    attempt += 1
                else:
                    # output the response token count for monitoring purposes
                    # TODO: Should check actual tokens rather than splitting on whitespace
                    input_token_count = len(payload.user_prompt)
                    completion_token_count = len(response.completion.split())
                    logging.info(
                        f"LLM response received with {completion_token_count} tokens "
                        f"(input prompt had {input_token_count} tokens). "
                        f"Total tokens: {input_token_count + completion_token_count}"
                    )
                    return json.loads(response.completion, strict=False)
            except json.JSONDecodeError as e:
                logging.warning(
                    f"JSON decode error on attempt {attempt + 1}/{max_retry_json_decode_error}: {e}"
                )
                attempt += 1
        raise ToolError(
            internal_code=ErrorCode.INVALID_JSON_DECODE_ERROR.value,
            message="Maximum retry attempts exceeded for JSON decode error.",
        )
