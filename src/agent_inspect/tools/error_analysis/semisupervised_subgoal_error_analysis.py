import json
from typing import List, Dict, Any, Optional

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.exception.exception import InvalidInputValueError
from agent_inspect.exception.error_codes import ErrorCode, ToolComponent
from agent_inspect.models.tools.analysis_models import AnalyzedSubgoalValidation
from agent_inspect.models.tools.error_cluster import (
    ErrorCluster,
    DEFAULT_SUBGOAL_ERROR_CLUSTERS,
)
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.core.utils import match_to_int
from agent_inspect.tools.error_analysis.subgoal_error_analysis import (
    SubgoalErrorAnalysis,
)
from agent_inspect.tools.error_analysis.utils.validator import (
    validate_user_defined_clusters,
)
from agent_inspect.tools.error_analysis.llm_templates import (
    SEMI_SUPERVISED_CLUSTERING_PROMPT_TEMPLATE,
)
from agent_inspect.tools.error_analysis.llm_output_schemas import (
    SEMI_SUPERVISED_CLUSTERING_OUTPUT_SCHEMA,
)

from agent_inspect.metrics.constants import (
    COMPLETE_INCOMPLETE_GRADE_PATTERN,
    COMPLETE_INCOMPLETE_PAIR,
)


class SemisupervisedSubgoalErrorAnalysis(SubgoalErrorAnalysis):
    """Semi-supervised algorithm for subgoal-based error analysis.

    This implementation uses predefined error clusters to classify failures, making it
    more efficient and consistent than unsupervised approaches. The algorithm:

    1. **Error Summarization**: For each failed subgoal validation, generates a concise
       error description using LLM analysis of judge trial explanations
    2. **Semi-Supervised Clustering**: Classifies errors into predefined categories using
       semantic matching. When error doesn't fit existing categories, it will be assigned to "Unclassified". No new clusters will be created in this approach.
    3. **Result Organization**: Groups analyzed validations by error cluster

    Notes: There is a small efficiency optimization compared with the unsupervised approach:
    When subgoal validations have mixed judge scores (some complete, some incomplete),
    only analyzes the first incomplete judge trial explanation rather than performing majority voting across all trials.

    **Default predefined clusters**:

            - :obj:`~agent_inspect.models.tools.error_cluster.IncorrectToolInputCluster`
            - :obj:`~agent_inspect.models.tools.error_cluster.IncorrectToolOutputHandlingCluster`
            - :obj:`~agent_inspect.models.tools.error_cluster.WrongToolSelectionCluster`
            - :obj:`~agent_inspect.models.tools.error_cluster.MissedToolCallCluster`
            - :obj:`~agent_inspect.models.tools.error_cluster.InstructionFollowingErrorCluster`
            - :obj:`~agent_inspect.models.tools.error_cluster.IncompleteCommunicationCluster`
            - :obj:`~agent_inspect.models.tools.error_cluster.FaithfulnessErrorCluster`
            - :obj:`~agent_inspect.models.tools.error_cluster.LogicalReasoningErrorCluster`
            - :obj:`~agent_inspect.models.tools.error_cluster.UnclassifiedCluster`

    :param llm_client: The :class:`~agent_inspect.clients.LLMClient` for performing error analysis and clustering.
    :param config: Optional configuration dictionary. Supported keys:

        - **max_workers**: Maximum number of concurrent workers (default: 20)

    :param error_clusters: Optional list of predefined error clusters. If not provided, uses the default clusters.

    :raises InvalidInputValueError: If llm_client is None or error_clusters are invalid (wrong format, empty list, etc.)

    Example:
        >>> from agent_inspect.clients.azure_openai_client import AzureOpenAIClient
        >>> from agent_inspect.models.tools import SubgoalErrorAnalysisDataSample
        >>> from agent_inspect.tools import SemisupervisedSubgoalErrorAnalysis
        >>>
        >>> llm_client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096)
        >>>
        >>> # Prepare data samples with subgoal validations
        >>> data_samples: List[SubgoalErrorAnalysisDataSample] = [...]
        >>>
        >>> # Run analysis with default clusters
        >>> analyzer = SemisupervisedSubgoalErrorAnalysis(llm_client)
        >>> result = analyzer.analyze_batch(data_samples)
        >>>
        >>> # Or use custom clusters
        >>> from agent_inspect.models.tools.error_cluster import ErrorCluster
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass(frozen=True)
        ... class APIRateLimitCluster(ErrorCluster):
        ...     cluster_label: str = "API Rate Limiting"
        ...     description: str = "Agent hit rate limits on external APIs"
        >>>
        >>> custom_clusters = [APIRateLimitCluster()]
        >>> analyzer = SemisupervisedSubgoalErrorAnalysis(llm_client, error_clusters=custom_clusters)
        >>> result = analyzer.analyze_batch(data_samples)
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
                message="LLM client must be provided for SemisupervisedSubgoalErrorAnalysis.",
                component_code=ToolComponent.TOOL_ERROR_CODE.value,
            )

        super().__init__(llm_client, config)

        # Default to DEFAULT_SUBGOAL_ERROR_CLUSTERS if not provided or is empty list
        if error_clusters is None or (
            isinstance(error_clusters, list) and len(error_clusters) == 0
        ):
            error_clusters = DEFAULT_SUBGOAL_ERROR_CLUSTERS
        else:
            validate_user_defined_clusters(error_clusters)

        self.error_clusters = error_clusters

    # ==================== Algorithm-Specific Methods (implementing abstract methods) ====================

    async def _summarize_errors_into_base_error(
        self, subgoal_validation: SubGoalValidationResult
    ) -> str | None:
        """Analyze a single subgoal validation to summarize the issues raised by judge trials.

        Semi-supervised optimization: When judge trials have mixed scores (some complete,
        some incomplete), uses only the first incomplete judge trial explanation rather
        than performing majority voting across all trials.

        :param subgoal_validation: a :obj:`~agent_inspect.models.metrics.validation_result.SubGoalValidationResult` object to analyze.
        :return: A concise error description if the subgoal is incomplete, None if completed.

        Behavior:

        - If subgoal is completed: Returns None (no error)
        - If all judge trials failed consistently: Summarizes first judge trial explanation
        - If mixed judge scores: Uses first incomplete judge trial explanation (optimization)
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
            # Majority says incomplete, use first incomplete judge trial explanation
            base_error = None
            for explanation in judge_trial_explanations:
                # Use match_to_int to check if this judge explanation trial is incomplete (score = 0)
                score = match_to_int(
                    explanation,
                    COMPLETE_INCOMPLETE_GRADE_PATTERN,
                    COMPLETE_INCOMPLETE_PAIR,
                )
                if score == 0:  # Incomplete
                    base_error = await self._summarize_error(
                        explanation, subgoal_validation.sub_goal.details
                    )
                    break

        return base_error

    async def _cluster_errors(
        self, analyzed_subgoal_validations: List[AnalyzedSubgoalValidation]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster analyzed subgoal validations using semi-supervised approach.

        Uses predefined error clusters to classify base errors via semantic matching.
        The LLM assigns each error to the most appropriate predefined cluster or creates
        new clusters only when errors don't fit existing categories.

        :param analyzed_subgoal_validations: a :obj:`~typing.List` of :obj:`~agent_inspect.models.tools.analysis_models.AnalyzedSubgoalValidation` objects with base errors.
        :return: Dictionary mapping cluster labels to lists of analyzed subgoal validations.
        :raises ToolError: If LLM clustering request fails or produces invalid output.
        """
        # Build index -> base_error mapping, and get all unique subgoals to prepare LLM prompt
        index_to_error_mapping = {
            str(idx): analyzed_subgoal_validation.base_error
            for idx, analyzed_subgoal_validation in enumerate(analyzed_subgoal_validations)
        }
        unique_subgoals = list(
            dict.fromkeys(
                analysed_subgoal_validation.subgoal_validation.sub_goal.details
                for analysed_subgoal_validation in analyzed_subgoal_validations
            )
        )

        # Convert error cluster dataclasses to dicts for LLM prompt
        predefined_clusters_dicts = [cluster.to_dict() for cluster in self.error_clusters]

        # Create LLM payload for semi-supervised clustering
        payload = LLMPayload(
            user_prompt=SEMI_SUPERVISED_CLUSTERING_PROMPT_TEMPLATE.format(
                predefined_clusters=json.dumps(predefined_clusters_dicts, indent=2),
                error_types=json.dumps(index_to_error_mapping, indent=2),
                subgoals=json.dumps(unique_subgoals, indent=2),
            ),
            structured_output=SEMI_SUPERVISED_CLUSTERING_OUTPUT_SCHEMA,
        )

        return await self._request_and_parse_json_with_retry(self.llm_client, payload)
