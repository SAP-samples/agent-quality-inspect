import re
from typing import List, Dict, Optional, Tuple, Any

from agent_inspect.models.metrics.validation_result import ToolCallValidationResult
from agent_inspect.exception import InvalidInputValueError, ErrorCode
from agent_inspect.tools.error_analysis.tool_call_error_analysis import (
    ToolCallErrorAnalysis,
)
from agent_inspect.models.tools.analysis_models import AnalyzedToolValidation
from agent_inspect.models.tools.error_cluster import (
    WrongToolSelectionCluster,
    IncorrectToolInputCluster,
    IncorrectToolOutputHandlingCluster,
    DEFAULT_TOOL_CALL_ERROR_CLUSTERS,
)
from agent_inspect.metrics.validator.constants import (
    TOOL_NOT_FOUND_EXPLANATION,
    ARGUMENT_FAILED_EXACT_MATCH_EXPLANATION,
    ARGUMENT_NOT_FOUND_EXPLANATION,
    ARGUMENT_FAILED_LLMJ_EXPLANATION,
    OUTPUT_FAILED_EXACT_MATCH_EXPLANATION,
    OUTPUT_FAILED_LLMJ_EXPLANATION,
)


class DeterministicToolCallErrorAnalysis(ToolCallErrorAnalysis):
    """
    Deterministic (rule-based) error analysis for tool call validations.

    This class classifies tool validation failures into predefined error clusters
    using regex pattern matching on validation explanations. It provides a fast,
    LLM-free alternative to :class:`~agent_inspect.tools.error_analysis.SemisupervisedToolCallErrorAnalysis`.

    **Cluster Labels (in priority order):**

        - :obj:`~agent_inspect.models.tools.error_cluster.IncorrectToolInputCluster`
        - :obj:`~agent_inspect.models.tools.error_cluster.IncorrectToolOutputHandlingCluster`
        - :obj:`~agent_inspect.models.tools.error_cluster.WrongToolSelectionCluster`

    **Priority Behavior:**
    When multiple failures occur (e.g., both argument AND output fail),
    the highest priority label is assigned based on the order above.

    :param config: Optional configuration dictionary. Supported keys:

        - **max_workers**: Maximum number of concurrent workers (default: 20)

    Example:

    >>> from agent_inspect.tools import DeterministicToolCallErrorAnalysis
    >>> from agent_inspect.models.tools import ToolCallErrorAnalysisDataSample
    >>>
    >>> analyzer = DeterministicToolCallErrorAnalysis(config={"max_workers": 1})
    >>> result = analyzer.analyze_batch(tool_validation_data_samples)
    >>> for cluster_label, validations in result.analyzed_validations_clustered_by_errors.items():
    ...     print(f"{cluster_label}: {len(validations)} errors")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.error_clusters = DEFAULT_TOOL_CALL_ERROR_CLUSTERS
        self._PATTERNS = self._build_patterns_from_constants()

    # ==================== Algorithm-Specific Methods (implementing abstract methods) ====================

    async def _classify_single_validation(
        self,
        validation_result: ToolCallValidationResult,
        data_sample_id: int,
        agent_run_id: Optional[int],
    ) -> Tuple[AnalyzedToolValidation, str]:
        """Classify a single failed tool validation result using deterministic regex matching.

        :param validation_result: a :obj:`~agent_inspect.models.metrics.validation_result.ToolCallValidationResult` object to classify.
        :param data_sample_id: ID of the data sample.
        :param agent_run_id: Optional agent run ID.
        :return: Tuple of (:obj:`~agent_inspect.models.tools.analysis_models.AnalyzedToolValidation`, cluster_label).
        """
        cluster_label = self._classify_from_explanations(validation_result.explanations)
        base_error = "\n".join(validation_result.explanations)

        return (
            AnalyzedToolValidation(
                tool_call_validation=validation_result,
                data_sample_id=data_sample_id,
                base_error=base_error,
                agent_run_id=agent_run_id,
            ),
            cluster_label,
        )

    # ==================== Deterministic Classification Helpers ====================

    @staticmethod
    def _build_patterns_from_constants() -> Dict[str, re.Pattern]:
        """Build all regex patterns from the explanation format constants."""

        # Helper to build pattern from format string, matching up to first sentence
        def build(fmt_str):
            # Split on '. ' to get first sentence, then convert to regex
            first_sentence = fmt_str.split(". ")[0]
            # Inline the regex conversion logic to avoid referencing the class
            pattern = re.escape(first_sentence)
            pattern = re.sub(r"\\{[^}]+\\}", "(.+)", pattern)
            return re.compile(pattern)

        return {
            "tool_not_found": build(TOOL_NOT_FOUND_EXPLANATION),
            "arg_failed_exact": build(ARGUMENT_FAILED_EXACT_MATCH_EXPLANATION),
            "arg_not_found": build(ARGUMENT_NOT_FOUND_EXPLANATION),
            "arg_failed_llmj": build(ARGUMENT_FAILED_LLMJ_EXPLANATION),
            "output_failed_exact": build(OUTPUT_FAILED_EXACT_MATCH_EXPLANATION),
            "output_failed_llmj": build(OUTPUT_FAILED_LLMJ_EXPLANATION),
        }

    def _classify_from_explanations(self, explanations: List[str]) -> str:
        """Classify validation failure from explanation strings.

        Priority order:

        1. Tool not found → "Wrong Tool Selection"
        2. Argument failed → "Incorrect Tool Input"
        3. Output failed → "Incorrect Tool Output Handling"

        :param explanations: List of validation explanation strings.
        :return: Cluster label string.
        """
        tool_not_found = False
        arg_failed = False
        output_failed = False

        if explanations is None or len(explanations) == 0:
            raise InvalidInputValueError(
                internal_code=ErrorCode.EMPTY_VALIDATION_RESULT.value,
                message="Validation explanations cannot be empty for failed validation",
            )

        for explanation in explanations:
            if self._PATTERNS["tool_not_found"].search(explanation):
                tool_not_found = True

            elif (
                self._PATTERNS["arg_failed_exact"].search(explanation)
                or self._PATTERNS["arg_not_found"].search(explanation)
                or self._PATTERNS["arg_failed_llmj"].search(explanation)
            ):
                arg_failed = True

            elif self._PATTERNS["output_failed_exact"].search(explanation) or self._PATTERNS[
                "output_failed_llmj"
            ].search(explanation):
                output_failed = True

        # Return highest priority error type found
        if tool_not_found:
            return WrongToolSelectionCluster().cluster_label
        if arg_failed:
            return IncorrectToolInputCluster().cluster_label
        if output_failed:
            return IncorrectToolOutputHandlingCluster().cluster_label

        # Fallback for unexpected validation format
        raise InvalidInputValueError(
            internal_code=ErrorCode.INVALID_JUDGE_RESPONSE_FORMAT_ERROR.value,
            message=f"Unable to classify validation failure from llm judge explanations: {explanations}. The reason may be that the explanation format has changed and the regex patterns need to be updated accordingly.",
        )
