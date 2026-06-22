import asyncio
import logging
from abc import abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from agent_inspect.tools.error_analysis.base_error_analysis import ErrorAnalysis
from agent_inspect.models.tools.analysis_models import (
    ToolCallErrorAnalysisDataSample,
    ToolCallErrorAnalysisResult,
    AnalyzedToolValidation,
)


class ToolCallErrorAnalysis(ErrorAnalysis):
    """Abstract base class for tool call-based error analysis implementations.

    This class provides common functionality for analyzing errors in tool call validations.
    Tool call analysis focuses on lower-level tool execution errors such as:

    - Wrong tool selection
    - Incorrect tool inputs
    - Incorrect tool output handling

    Subclasses implement specific algorithms (semi-supervised, deterministic) for
    identifying and clustering these errors.

    :param config: Optional configuration dictionary. Supported keys:

        - **max_workers**: Maximum number of concurrent workers (default: 20)

    :param error_clusters: Optional list of predefined error clusters for classification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ==================== Public API ====================

    async def analyze_batch_async(
        self, data_samples: List[ToolCallErrorAnalysisDataSample]
    ) -> ToolCallErrorAnalysisResult:
        """Async version of :meth:`~agent_inspect.tools.error_analysis.tool_call_error_analysis.ErrorAnalysis.analyze_batch`. Performs error analysis on a batch of data samples with tool call validations.

        Use this method when calling from an async context (i.e., when an event loop is already running).

        Returns the :obj:`~agent_inspect.models.tools.analysis_models.ToolCallErrorAnalysisResult` containing the clustered error types with their associated tool call validations.

        :param data_samples: a :obj:`~typing.List` of :obj:`~agent_inspect.models.tools.analysis_models.ToolCallErrorAnalysisDataSample` objects to perform error analysis on. Each data sample contains multiple tool call validations.
        :return: an :obj:`~agent_inspect.models.tools.analysis_models.ToolCallErrorAnalysisResult` object containing the error analysis results:

            - :obj:`~agent_inspect.models.tools.analysis_models.ToolCallErrorAnalysisResult.analyzed_validations_clustered_by_errors`: Dictionary mapping clustered error types to lists of incomplete tool call validations exhibiting those errors.
        """
        logging.info(f"Starting tool call error analysis for {len(data_samples)} data samples.")

        # Process all tool validations concurrently (async pattern)
        loop = asyncio.get_running_loop()
        loop.set_default_executor(self.executor)
        tasks = []
        for data_sample in data_samples:
            for validation_result in data_sample.tool_call_validations:
                # Skip completed validations
                if validation_result.is_completed:
                    continue
                tasks.append(
                    self._classify_single_validation(
                        validation_result,
                        data_sample.data_sample_id,
                        data_sample.agent_run_id,
                    )
                )

        all_analyzed_validations = await asyncio.gather(*tasks)

        # Group by cluster_label
        clustered_by_errors: Dict[str, List[AnalyzedToolValidation]] = {}
        for analyzed, cluster_label in all_analyzed_validations:
            if cluster_label not in clustered_by_errors:
                clustered_by_errors[cluster_label] = []
            clustered_by_errors[cluster_label].append(analyzed)

        logging.info(
            f"Tool call error analysis complete. Found {len(clustered_by_errors)} error clusters."
        )

        return ToolCallErrorAnalysisResult(
            analyzed_validations_clustered_by_errors=clustered_by_errors
        )

    def analyze_batch(
        self, data_samples: List[ToolCallErrorAnalysisDataSample]
    ) -> ToolCallErrorAnalysisResult:
        """Analyze a batch of data samples and return results.

        This is a synchronous wrapper around :meth:`analyze_batch_async`. Use this method
        when calling from a synchronous context. If you're already in an async context
        (i.e., inside an async function with an event loop running), use :meth:`analyze_batch_async` instead.

        :param data_samples: a :obj:`~typing.List` of :obj:`~agent_inspect.models.tools.analysis_models.ToolCallErrorAnalysisDataSample` objects to analyze.
        :return: an :obj:`~agent_inspect.models.tools.analysis_models.ToolCallErrorAnalysisResult` object with clustered errors.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.analyze_batch_async(data_samples))
        else:
            raise RuntimeError(
                "analyze_batch cannot be called from an async context. Use analyze_batch_async instead."
            )

    # ==================== Abstract Methods (to be implemented by subclasses) ====================

    @abstractmethod
    async def _classify_single_validation(
        self, validation_result, data_sample_id: int, agent_run_id: Optional[int]
    ) -> Tuple[AnalyzedToolValidation, str]:
        """Classify a single tool validation into an error cluster.

        This method is algorithm-specific:

        - Semi-supervised: Uses LLM to classify into predefined clusters
        - Deterministic: Uses regex patterns to classify deterministically

        :param validation_result: Tool call validation result to classify.
        :param data_sample_id: ID of the data sample.
        :param agent_run_id: Optional agent run ID.
        :return: Tuple of (:obj:`~agent_inspect.models.tools.analysis_models.AnalyzedToolValidation`, cluster_label) where cluster_label is the error category string.
        """
        pass
