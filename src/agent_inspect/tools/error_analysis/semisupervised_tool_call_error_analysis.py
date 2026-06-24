import json
from typing import List, Dict, Any, Optional, Tuple

from agent_inspect.models.metrics.validation_result import ToolCallValidationResult
from agent_inspect.tools.error_analysis.tool_call_error_analysis import (
    ToolCallErrorAnalysis,
)
from agent_inspect.models.tools.analysis_models import AnalyzedToolValidation
from agent_inspect.models.tools.error_cluster import (
    ErrorCluster,
    DEFAULT_TOOL_CALL_ERROR_CLUSTERS,
)
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.tools.error_analysis.llm_templates import (
    TOOL_VALIDATION_CLASSIFICATION_PROMPT_TEMPLATE,
)
from agent_inspect.tools.error_analysis.llm_output_schemas import (
    TOOL_VALIDATION_CLASSIFICATION_OUTPUT_SCHEMA,
)
from agent_inspect.tools.error_analysis.utils.validator import (
    validate_user_defined_clusters,
)
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.exception.error_codes import ErrorCode, ToolComponent


class SemisupervisedToolCallErrorAnalysis(ToolCallErrorAnalysis):
    """Performs semi-supervised error analysis on tool call validations using LLMs.

    This implementation uses a direct classification approach where each validation error
    is classified into predefined error clusters using an LLM judge. The validation
    explanation serves as the base error description, and the LLM assigns the most
    appropriate error cluster label.

    Unlike unsupervised methods that discover patterns through clustering, this semi-supervised
    approach leverages predefined error categories to provide more interpretable and consistent
    error classification across evaluations. When a validation error does not fit well into any of the predefined clusters, it will be labeled as "Unclassified". No new clusters will be created in this approach.

    **Default predefined clusters**:

        - :obj:`~agent_inspect.models.tools.error_cluster.IncorrectToolInputCluster`
        - :obj:`~agent_inspect.models.tools.error_cluster.IncorrectToolOutputHandlingCluster`
        - :obj:`~agent_inspect.models.tools.error_cluster.WrongToolSelectionCluster`

    :param llm_client: The :class:`~agent_inspect.clients.LLMClient` to use for error classification. Must implement the :class:`~agent_inspect.clients.LLMClient` interface for making LLM requests.
    :param config: Optional configuration dictionary. Supported keys:

        - **max_workers**: Maximum number of concurrent workers for processing validations (default: 20)

    :param error_clusters: Optional list of error cluster definitions. If not provided, uses the default tool call error clusters.
    :raises InvalidInputValueError: If llm_client is None or if error_clusters are invalid.

    Example:
        >>> from agent_inspect.clients.azure_openai_client import AzureOpenAIClient
        >>> from agent_inspect.models.tools.analysis_models import ToolCallErrorAnalysisDataSample
        >>>
        >>> # Initialize with LLM client
        >>> llm_client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096)
        >>> analyzer = SemisupervisedToolCallErrorAnalysis(
        ...     llm_client=llm_client,
        ...     config={"max_workers": 10}
        ... )
        >>>
        >>> # Prepare data samples with tool call validations
        >>> data_samples: List[ToolCallErrorAnalysisDataSample] = [...]
        >>>
        >>> # Run error analysis with default clusters
        >>> results = analyzer.analyze_batch(data_samples)
        >>>
        >>> # Or provide custom error clusters
        >>> from agent_inspect.models.tools.error_cluster import ErrorCluster
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass(frozen=True)
        >>> class CustomErrorCluster(ErrorCluster):
        ...     cluster_label: str = "Custom Error Type"
        ...     description: str = "Description of when this error occurs"
        >>>
        >>> custom_clusters = [CustomErrorCluster()]
        >>> custom_analyzer = SemisupervisedToolCallErrorAnalysis(
        ...     llm_client=llm_client,
        ...     error_clusters=custom_clusters
        ... )
        >>> results = custom_analyzer.analyze_batch(data_samples)
        >>>
        >>> # Inspect results
        >>> for cluster_label, validations in results.analyzed_validations_clustered_by_errors.items():
        ...     print(f"{cluster_label}: {len(validations)} errors")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None,
        error_clusters: Optional[List[ErrorCluster]] = None,
    ):
        if llm_client is None:
            raise InvalidInputValueError(
                internal_code=ErrorCode.MISSING_VALUE.value,
                message="LLM client must be provided for SemisupervisedToolCallErrorAnalysis.",
                component_code=ToolComponent.TOOL_ERROR_CODE.value,
            )

        super().__init__(config)
        self.llm_client = llm_client

        # Default to default error clusters if clusters not provided or is empty list
        if error_clusters is None or (
            isinstance(error_clusters, list) and len(error_clusters) == 0
        ):
            error_clusters = DEFAULT_TOOL_CALL_ERROR_CLUSTERS
        else:
            validate_user_defined_clusters(error_clusters)

        self.error_clusters = error_clusters

    # ==================== Algorithm-Specific Methods (implementing abstract methods) ====================

    async def _classify_single_validation(
        self,
        validation_result: ToolCallValidationResult,
        data_sample_id: int,
        agent_run_id: Optional[int],
    ) -> Tuple[AnalyzedToolValidation, str]:
        """Classify a single tool validation error into a predefined cluster.

        Uses the validation explanation as the base error and makes an LLM request
        to assign the most appropriate cluster label based on the expected tool
        call details.

        :param validation_result: a :obj:`~agent_inspect.models.metrics.validation_result.ToolCallValidationResult` object to classify.
        :param data_sample_id: ID of the data sample this validation belongs to.
        :param agent_run_id: Optional ID of the agent run.
        :return: Tuple of (:obj:`~agent_inspect.models.tools.analysis_models.AnalyzedToolValidation`, cluster_label) where:

            - :obj:`~agent_inspect.models.tools.analysis_models.AnalyzedToolValidation` contains the validation result and base error
            - cluster_label is the assigned error category

        Note:
            Uses self.error_clusters from initialization for classification.
        """
        # Use validation explanation as base_error
        base_error = "\n".join(validation_result.explanations)
        expected_tool = validation_result.expected_tool_call

        # Convert error clusters to dict format for LLM prompt
        clusters_as_dicts = [c.to_dict() for c in self.error_clusters]

        # Build classification prompt
        payload = LLMPayload(
            user_prompt=TOOL_VALIDATION_CLASSIFICATION_PROMPT_TEMPLATE.format(
                predefined_clusters=json.dumps(clusters_as_dicts, indent=2),
                tool_name=expected_tool.tool,
                expected_parameters=expected_tool.expected_parameters,
                expected_output=expected_tool.expected_output,
                explanation=base_error,
            ),
            structured_output=TOOL_VALIDATION_CLASSIFICATION_OUTPUT_SCHEMA,
        )

        # Call LLM for classification
        response_dict = await self._request_and_parse_json_with_retry(self.llm_client, payload)
        cluster_label = response_dict.get("cluster_label")

        return (
            AnalyzedToolValidation(
                tool_call_validation=validation_result,
                data_sample_id=data_sample_id,
                base_error=base_error,
                agent_run_id=agent_run_id,
            ),
            cluster_label,
        )
