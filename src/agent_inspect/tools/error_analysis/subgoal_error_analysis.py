import asyncio
import logging
from abc import abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from agent_inspect.tools.error_analysis.base_error_analysis import ErrorAnalysis
from agent_inspect.models.tools.analysis_models import (
    SubgoalErrorAnalysisDataSample,
    SubgoalErrorAnalysisResult,
    AnalyzedSubgoalValidation,
)
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.core.utils import tally_votes
from agent_inspect.exception.error_codes import ErrorCode, ToolComponent
from agent_inspect.exception import ToolError, InvalidInputValueError
from agent_inspect.tools.error_analysis.llm_templates import (
    ERROR_SUMMARIZATION_PROMPT_TEMPLATE,
)
from agent_inspect.tools.error_analysis.llm_output_schemas import (
    ERROR_SUMMARIZATION_OUTPUT_SCHEMA,
)

from agent_inspect.metrics.constants import (
    COMPLETE_INCOMPLETE_GRADE_PATTERN,
    COMPLETE_INCOMPLETE_PAIR,
)


class SubgoalErrorAnalysis(ErrorAnalysis):
    """Abstract base class for subgoal-based error analysis implementations.

    This class provides common functionality for analyzing errors in subgoal validations, including:

    - LLM client management
    - Helper methods for processing judge trial explanations
    - Splitting validations by completeness status
    - Concurrent processing of data samples using thread pool executors

    Subclasses implement specific algorithms (unsupervised, semi-supervised) for
    identifying and clustering errors.

    :param llm_client: The LLM client used for error analysis operations.
    :param config: Optional configuration dictionary. Supported keys:

        - **max_workers**: Maximum number of concurrent workers (default: 20)
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm_client = llm_client

    # ==================== Public API ====================

    async def analyze_batch_async(
        self, data_samples: List[SubgoalErrorAnalysisDataSample]
    ) -> SubgoalErrorAnalysisResult:
        """Performs error analysis on a batch of data samples.

        Use this method when calling from an async context (i.e., when an event loop is already running).
        If you are calling from a synchronous context, use the :meth:`~agent_inspect.models.tools.analysis_models.ErrorAnalysis.analyze_batch` method instead

        Returns the :obj:`~agent_inspect.models.tools.analysis_models.SubgoalErrorAnalysisResult` containing the clustered error types with their associated subgoal validations and the rest of subgoal validations that don't have errors.

        :param data_samples: a :obj:`~typing.List` of :obj:`~agent_inspect.models.tools.analysis_models.SubgoalErrorAnalysisDataSample` objects to perform error analysis on. Each data sample contains multiple subgoal validations.
        :return: an :obj:`~agent_inspect.models.tools.analysis_models.SubgoalErrorAnalysisResult` object containing the error analysis results:

            - :obj:`~agent_inspect.models.tools.analysis_models.SubgoalErrorAnalysisResult.analyzed_validations_clustered_by_errors`: Dictionary mapping clustered error types to lists of incomplete subgoal validations exhibiting those errors
            - :obj:`~agent_inspect.models.tools.analysis_models.SubgoalErrorAnalysisResult.completed_subgoal_validations`: List of subgoal validations that were successfully completed without errors
        """
        # Process all data samples concurrently
        logging.info(f"Starting subgoal error analysis for {len(data_samples)} data samples.")

        loop = asyncio.get_running_loop()
        loop.set_default_executor(self.executor)
        tasks = [self._analyze(data_sample) for data_sample in data_samples]

        all_analyzed_subgoal_validations = await asyncio.gather(*tasks)

        # Separate completed/incomplete validations
        logging.info("Separating analyzed subgoal validations into completed and incomplete.")
        complete_subgoal_validations, incomplete_subgoal_validations = (
            self._split_analysed_subgoal_validations_by_completeness(
                all_analyzed_subgoal_validations
            )
        )

        # Cluster errors (pass clusters if available for semi-supervised)
        logging.info("Clustering analyzed subgoal validations based on base errors.")
        llm_clusterings = await self._cluster_errors(incomplete_subgoal_validations)

        logging.info("Building final clustered error analysis result.")
        analyzed_validations_clustered_by_errors = self._build_clustered_result(
            llm_clusterings, incomplete_subgoal_validations
        )

        final_result = SubgoalErrorAnalysisResult(
            analyzed_validations_clustered_by_errors=analyzed_validations_clustered_by_errors,
            completed_subgoal_validations=complete_subgoal_validations,
        )

        return final_result

    def analyze_batch(
        self, data_samples: List[SubgoalErrorAnalysisDataSample]
    ) -> SubgoalErrorAnalysisResult:
        """Analyze a batch of data samples and return results.

        This is a synchronous wrapper around :meth:`analyze_batch_async`. Use this method
        when calling from a synchronous context. If you're already in an async context
        (i.e., inside an async function with an event loop running), use :meth:`analyze_batch_async` instead.

        :param data_samples: a :obj:`~typing.List` of :obj:`~agent_inspect.models.tools.analysis_models.SubgoalErrorAnalysisDataSample` objects to analyze.
        :return: an :obj:`~agent_inspect.models.tools.analysis_models.SubgoalErrorAnalysisResult` object with clustered errors.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.analyze_batch_async(data_samples))
        else:
            raise RuntimeError(
                "analyze_batch cannot be called from an async context. Use analyze_batch_async instead."
            )

    # ==================== Private Helper Methods (in execution order) ====================

    async def _analyze(
        self, data_sample: SubgoalErrorAnalysisDataSample
    ) -> List[AnalyzedSubgoalValidation]:
        """Analyze a single data sample to obtain base errors for each subgoal validation."""
        logging.info(f"Analyzing data sample with ID: {data_sample.data_sample_id}")
        tasks = [
            self._summarize_errors_into_base_error(subgoal_validation)
            for subgoal_validation in data_sample.subgoal_validations
        ]
        base_errors: List[Optional[str]] = await asyncio.gather(*tasks)

        analyzed_subgoal_validations = [
            AnalyzedSubgoalValidation(
                subgoal_validation=subgoal_validation,
                data_sample_id=data_sample.data_sample_id,
                base_error=base_error,
                agent_run_id=data_sample.agent_run_id,
            )
            for subgoal_validation, base_error in zip(data_sample.subgoal_validations, base_errors)
        ]

        return analyzed_subgoal_validations

    async def _summarize_error(self, judge_trial_explanation: str, subgoal: str) -> str:
        """Summarize a single judge trial explanation into a concise error description using LLM."""
        payload = LLMPayload(
            user_prompt=ERROR_SUMMARIZATION_PROMPT_TEMPLATE.format(
                subgoals=subgoal, explanation=judge_trial_explanation
            ),
            structured_output=ERROR_SUMMARIZATION_OUTPUT_SCHEMA,
        )

        response_dict = await self._request_and_parse_json_with_retry(self.llm_client, payload)
        if "error_type" in response_dict:
            return response_dict["error_type"].strip()
        else:
            raise ToolError(
                internal_code=ErrorCode.UNSUCCESSFUL_LLM_SUMMARIZATION.value,
                message=f"LLM error summarization request failed as no error_type found in response: {response_dict}",
            )

    def _split_analysed_subgoal_validations_by_completeness(
        self, analysed_subgoal_validation_list: List[List[AnalyzedSubgoalValidation]]
    ) -> Tuple[List[AnalyzedSubgoalValidation], List[AnalyzedSubgoalValidation]]:
        """Split analyzed subgoal validations into completed and incomplete lists."""
        complete_subgoal_analysed_subgoal_validations = []
        incomplete_subgoal_analysed_subgoal_validations = []
        for sublist in analysed_subgoal_validation_list:
            for analysed_subgoal_validation in sublist:
                if analysed_subgoal_validation.base_error is None:
                    complete_subgoal_analysed_subgoal_validations.append(
                        analysed_subgoal_validation
                    )
                else:
                    incomplete_subgoal_analysed_subgoal_validations.append(
                        analysed_subgoal_validation
                    )

        return (
            complete_subgoal_analysed_subgoal_validations,
            incomplete_subgoal_analysed_subgoal_validations,
        )

    def _build_clustered_result(
        self,
        llm_clustering: Dict[str, List[Dict[str, Any]]],
        analyzed_subgoal_validations: List[AnalyzedSubgoalValidation],
    ) -> Dict[str, List[AnalyzedSubgoalValidation]]:
        """Build final result by mapping cluster labels to analyzed subgoal validations."""
        analyzed_validations_clustered_by_errors = {}

        # Track which error_ids have been assigned to clusters
        assigned_error_ids = set()

        for cluster in llm_clustering["clusters"]:
            logging.info(
                f"Processing cluster: {cluster['cluster_label']} with error_ids: {cluster['error_ids']}"
            )
            cluster_label = cluster["cluster_label"]
            error_ids = cluster["error_ids"]

            analyzed_validations_clustered_by_errors[cluster_label] = [
                analyzed_subgoal_validations[int(error_id)]
                for error_id in error_ids
                if int(error_id) < len(analyzed_subgoal_validations)
            ]

            # Track assigned error_ids
            assigned_error_ids.update(
                int(error_id)
                for error_id in error_ids
                if int(error_id) < len(analyzed_subgoal_validations)
            )

        # Check for missing error_ids and log warning
        total_errors = len(analyzed_subgoal_validations)
        if len(assigned_error_ids) < total_errors:
            missing_count = total_errors - len(assigned_error_ids)
            missing_ids = set(range(total_errors)) - assigned_error_ids
            logging.warning(
                f"LLM clustering missed {missing_count} error(s) out of {total_errors}. "
                f"Missing error_ids: {sorted(missing_ids)[:10]}{'...' if len(missing_ids) > 10 else ''}"
            )

            # Add unclustered errors to a separate None cluster
            if missing_ids:
                analyzed_validations_clustered_by_errors[None] = [
                    analyzed_subgoal_validations[error_id] for error_id in sorted(missing_ids)
                ]

        return analyzed_validations_clustered_by_errors

    # ==================== Utility Methods ====================

    def _get_judge_trial_explanations_from_subgoal_validation(
        self, subgoal_validation: SubGoalValidationResult
    ) -> List[str]:
        """Extract judge trial explanations from a subgoal validation result (excluding overall explanation at index 0)."""
        if len(subgoal_validation.explanations) <= 1:
            raise InvalidInputValueError(
                internal_code=ErrorCode.INVALID_VALUE.value,
                message="Invalid SubGoalValidationResult.explanation format. "
                "ErrorAnalysis expects SubGoalValidationResult.explanation to have an overall explanation in index 0, "
                "and judge trial explanations from index 1 onwards.",
                component_code=ToolComponent.TOOL_ERROR_CODE.value,
            )
        # Strip out the overall explanation to get a list of judge trial explanations
        return subgoal_validation.explanations[1::]

    def _has_failed_consistently(self, subgoal_validation: SubGoalValidationResult) -> bool:
        """Check if all judge trials for the subgoal validation have failed."""
        judge_trials_explanations = self._get_judge_trial_explanations_from_subgoal_validation(
            subgoal_validation
        )
        complete_cnt, _, invalid_cnt = tally_votes(
            0,
            0,
            0,
            judge_trials_explanations,
            COMPLETE_INCOMPLETE_GRADE_PATTERN,
            COMPLETE_INCOMPLETE_PAIR,
        )

        # Error analysis expects all judge trials to be valid. Raise error if any invalid judge trial responses are encountered
        if invalid_cnt > 0:
            raise InvalidInputValueError(
                internal_code=ErrorCode.INVALID_VALUE.value,
                message=f"Subgoal error analysis encountered {invalid_cnt} invalid judge response(s) "
                f"in subgoal validation with subgoal: '{subgoal_validation.sub_goal}' and explanations: {subgoal_validation.explanations}. "
                f"Error analysis expects all judge trial explanations to be valid.",
            )

        return complete_cnt == 0

    # ==================== Abstract Methods (to be implemented by subclasses) ====================

    @abstractmethod
    async def _summarize_errors_into_base_error(
        self, subgoal_validation: SubGoalValidationResult
    ) -> Optional[str]:
        """Summarize judge trial explanations into a single base error description.

        This method is algorithm-specific:

        - Unsupervised: Uses majority voting when judge trials have mixed results
        - Semi-supervised: Uses first incomplete judge trial for efficiency

        :param subgoal_validation: a :obj:`~agent_inspect.models.metrics.validation_result.SubGoalValidationResult` object to analyze.
        :return: A concise base error description, or None if the subgoal was completed.
        """
        pass

    @abstractmethod
    async def _cluster_errors(
        self, analyzed_subgoal_validations: List[AnalyzedSubgoalValidation]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster analyzed subgoal validations based on their base errors.

        :param analyzed_subgoal_validations: a :obj:`~typing.List` of :obj:`~agent_inspect.models.tools.analysis_models.AnalyzedSubgoalValidation` objects with base errors.
        :return: Dictionary containing clustering results with cluster labels and error_ids.
        """
        pass
