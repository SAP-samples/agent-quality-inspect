import asyncio
import json
from typing import List, Dict, Any, Optional

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import ToolError
from agent_inspect.models.tools.analysis_models import AnalyzedSubgoalValidation
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.tools.error_analysis.subgoal_error_analysis import (
    SubgoalErrorAnalysis,
)
from agent_inspect.tools.error_analysis.llm_templates import (
    MAJORITY_VOTE_PROMPT_TEMPLATE,
    UNSUPERVISED_CLUSTERING_PROMPT_TEMPLATE,
)
from agent_inspect.tools.error_analysis.llm_output_schemas import (
    MAJORITY_VOTE_OUTPUT_SCHEMA,
    UNSUPERVISED_CLUSTERING_OUTPUT_SCHEMA,
)


class UnsupervisedSubgoalErrorAnalysis(SubgoalErrorAnalysis):
    """Unsupervised error analysis for subgoal validations.

    This class implements an unsupervised learning approach to error analysis that:
    1) Identifies low-level errors from individual subgoal validation failures
    2) Dynamically clusters similar low-level errors into error categories
    3) Error clusters are generated on-the-fly based on the low-level errors, without relying on any predefined error categories.

    The analysis follows a two-step process:

    - **Step 1**: For each failed subgoal validation, summarize the issues raised by
      the judge trials into a concise base error description.
    - **Step 2**: Cluster all base errors semantically using an LLM to identify
      common failure patterns.

    This approach is useful when you want to discover error patterns without prior
    assumptions about what kinds of errors might occur.

    :param llm_client: The :class:`~agent_inspect.clients.llm_client.LLMClient` for performing error analysis operations.
    :param config: Optional configuration dictionary. Supported keys:

        - **max_workers**: Maximum number of concurrent workers (default: 20)

    Example:
        >>> from agent_inspect.models.tools import SubgoalErrorAnalysisDataSample
        >>> from agent_inspect.tools.error_analysis import UnsupervisedSubgoalErrorAnalysis
        >>> from agent_inspect.clients.azure_openai_client import AzureOpenAIClient
        >>>
        >>> # Prepare your data samples
        >>> data_samples: List[SubgoalErrorAnalysisDataSample] = [...]
        >>>
        >>> # Initialize the analyzer
        >>> llm_client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096)
        >>> analyzer = UnsupervisedSubgoalErrorAnalysis(
        ...     llm_client=llm_client,
        ...     config={"max_workers": 10}
        ... )
        >>>
        >>> # Run error analysis
        >>> results = analyzer.analyze_batch(data_samples)
        >>> error_categories = list(results.analyzed_validations_clustered_by_errors.keys())
        >>> print(f"Discovered error categories: {error_categories}")
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_client, config)

    # ==================== Algorithm-Specific Methods (implementing abstract methods) ====================

    async def _summarize_errors_into_base_error(
        self, subgoal_validation: SubGoalValidationResult
    ) -> Optional[str]:
        """Summarize judge trial explanations into a single base error description.

        This method uses the unsupervised approach:

        - If subgoal is completed: returns None (no error)
        - If all judge trials failed: uses first judge trial explanation
        - If mixed results: generates error summary for each judge trial and performs majority voting to determine the most likely base error

        :param subgoal_validation: a :obj:`~agent_inspect.models.metrics.validation_result.SubGoalValidationResult` object to analyze.
        :return: A concise base error description, or None if the subgoal was completed.
        """
        if subgoal_validation.is_completed:
            return None

        judge_trial_explanations = self._get_judge_trial_explanations_from_subgoal_validation(
            subgoal_validation
        )

        if self._has_failed_consistently(subgoal_validation):
            # All judge trials failed - take first judge trial explanation as representative
            base_error = await self._summarize_error(
                judge_trial_explanations[0], subgoal_validation.sub_goal.details
            )
        else:
            # Mixed results - use unsupervised majority voting approach
            # Get error summary for every judge trial explanation
            tasks = [
                self._summarize_error(judge_trial_explanation, subgoal_validation.sub_goal.details)
                for judge_trial_explanation in judge_trial_explanations
            ]
            errors = await asyncio.gather(*tasks)
            # Perform majority voting to determine most likely error
            base_error = await self._perform_majority_voting(errors)

        return base_error

    async def _cluster_errors(
        self, analyzed_subgoal_validations: List[AnalyzedSubgoalValidation]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster analyzed subgoal validations based on their base errors.

        Uses unsupervised clustering to dynamically generate error categories from
        the base errors. The LLM identifies similar errors and groups them together.

        :param analyzed_subgoal_validations: a :obj:`~typing.List` of :obj:`~agent_inspect.models.tools.analysis_models.AnalyzedSubgoalValidation` objects with base errors.
        :return: Dictionary containing clustering results with the following structure::

                {
                    "clusters": [
                        {
                            "cluster_label": "<error category name>",
                            "description": "<description of this error type>",
                            "error_ids": ["0", "3", "5", ...]  # indices into analyzed_subgoal_validations
                        },
                        ...
                    ]
                }
        """
        # Build index -> base_error mapping for LLM prompt
        index_to_error_mapping = {
            str(idx): analyzed_subgoal_validation.base_error
            for idx, analyzed_subgoal_validation in enumerate(analyzed_subgoal_validations)
        }

        # Get all unique subgoals to provide context for clustering
        unique_subgoals = list(
            dict.fromkeys(
                analysed_subgoal_validation.subgoal_validation.sub_goal.details
                for analysed_subgoal_validation in analyzed_subgoal_validations
            )
        )

        # Create LLM payload for unsupervised clustering
        payload = LLMPayload(
            user_prompt=UNSUPERVISED_CLUSTERING_PROMPT_TEMPLATE.format(
                error_types=json.dumps(index_to_error_mapping, indent=2),
                subgoals=json.dumps(unique_subgoals, indent=2),
            ),
            structured_output=UNSUPERVISED_CLUSTERING_OUTPUT_SCHEMA,
        )

        return await self._request_and_parse_json_with_retry(self.llm_client, payload)

    # ==================== Unsupervised-Specific Helper Methods ====================

    async def _perform_majority_voting(self, errors: List[str]) -> str:
        """Perform majority voting on multiple error descriptions.

        Uses an LLM to determine the most common or representative error from a list
        of error descriptions.

        :param errors: List of error descriptions to vote on.
        :return: The most probable error description according to the LLM.
        :raises ToolError: If the LLM fails to perform majority voting.
        """
        payload = LLMPayload(
            user_prompt=MAJORITY_VOTE_PROMPT_TEMPLATE.format(
                error_type_list=json.dumps(errors, indent=2)
            ),
            structured_output=MAJORITY_VOTE_OUTPUT_SCHEMA,
        )

        response_dict = await self._request_and_parse_json_with_retry(self.llm_client, payload)
        if "most_probable_error_type" in response_dict:
            return response_dict["most_probable_error_type"].strip()
        else:
            raise ToolError(
                internal_code=ErrorCode.UNSUCCESSFUL_MAJORITY_VOTING.value,
                message=f"LLM majority voting request failed as no most_probable_error_type found in response: {response_dict}",
            )
