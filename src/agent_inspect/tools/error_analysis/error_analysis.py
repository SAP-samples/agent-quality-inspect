import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import ToolError
from agent_inspect.models.tools.analysis_models import ErrorAnalysisDataSample, AnalyzedSubgoalValidation, ErrorAnalysisResult
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult
from agent_inspect.models.llm_payload import LLMPayload
from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.metrics.constants import STATUS_200, MAX_RETRY_JSON_DECODE_ERROR
from agent_inspect.metrics.utils.metrics_utils import tally_votes
from agent_inspect.tools.error_analysis.llm_constants import (
    UNSUPERVISED_ERROR_SUMMARIZATION_PROMPT_TEMPLATE,
    ERROR_SUMMARIZATION_OUTPUT_SCHEMA,
    MAJORITY_VOTE_PROMPT_TEMPLATE,
    MAJORITY_VOTE_OUTPUT_SCHEMA,
    CLUSTERING_PROMPT_TEMPLATE,
    CLUSTERING_OUTPUT_SCHEMA
)


class ErrorAnalysis:
    #TODO: 20 workers is an arbitrary choice need to further tune based on performance testing
    """
    Method to perform error analysis across multiple data samples using LLMs in order for developers to easily identify and understand common errors of agents. The method is based on subgoal validations and will execute a two-step unsupervised learning process: 1) low-level error identification, 2) semantic clustering of error types.
    
    :param llm_client: the client which allows connection to the LLM model for performing error analysis.
    :param max_workers: Maximum number of concurrent workers for processing data samples. Default to ``20``.
    """
    def __init__(self, llm_client: LLMClient, max_workers: int = 20):
        self.llm_client = llm_client
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def analyze_batch(self, data_samples: List[ErrorAnalysisDataSample]) -> ErrorAnalysisResult:
        """
        Performs error analysis on a batch of data samples (usually across the entire dataset samples).
        Returns the :obj:`~agent_inspect.models.tools.analysis_models.ErrorAnalysisResult` containing the clustered error types with their associated subgoal validations and the rest of subgoal validations that don't have errors.
        
        :param data_samples: List of data samples to perform error analysis on. Each data sample contains multiple subgoal validations.
        :return: an :obj:`~agent_inspect.models.tools.analysis_models.ErrorAnalysisResult` containing the error analysis results:
        
            - :obj:`~agent_inspect.models.tools.analysis_models.ErrorAnalysisResult.analyzed_validations_clustered_by_errors`: Dictionary mapping clustered error types to lists of incomplete subgoal validations exhibiting those errors
            - :obj:`~agent_inspect.models.tools.analysis_models.ErrorAnalysisResult.completed_subgoal_validations`: List of subgoal validations that were successfully completed without errors
            
        Example:
        
        >>> from agent_inspect.models.tools import ErrorAnalysisDataSample
        >>> from agent_inspect.tools import ErrorAnalysis 
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> # Prepare your error_analysis_data_samples: List[ErrorAnalysisDataSample] = [...]
        >>> error_analysis_data_samples: List[ErrorAnalysisDataSample] = [...] 
        >>> llm_client = AzureOpenAIClient(
        ...     model="gpt-4.1", 
        ...     max_tokens=4096
        ...     )
        >>> error_analyzer = ErrorAnalysis(
        ...     llm_client=llm_client, 
        ...     max_workers=10
        ...     )
        >>>
        >>> test_error_analysis_results = error_analyzer.analyze_batch(error_analysis_data_samples)
        >>> test_error_categories = list(test_error_analysis_results.analyzed_validations_clustered_by_errors.keys())
        >>> print(f"Identified error categories: {test_error_categories}")
        """
        # Process all data samples concurrently
        logging.info(f"Starting error analysis for {len(data_samples)} data samples.")
        async def runner():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(self.executor)
            tasks = [self._analyze(data_sample) for data_sample in data_samples]
            return await asyncio.gather(*tasks)

        all_analyzed_subgoal_validations = asyncio.run(runner())
        
        # TODO: revisit the clustering method (probably need to put it as a separate function)
        # Seperate the results
        logging.info("Separating analyzed subgoal validations from all data samples into completed and not completed.")
        complete_subgoal_analyzed_subgoal_validations, incomplete_subgoal_analyzed_subgoal_validations =\
            self._split_analysed_subgoal_validations_by_completeness(all_analyzed_subgoal_validations)

        # Cluster the errors
        logging.info("Clustering analyzed subgoal validations based on base errors.")
        llm_clusterings = asyncio.run(self._cluster_errors(incomplete_subgoal_analyzed_subgoal_validations))
        
        logging.info("Building final clustered error analysis result.")
        analyzed_validations_clustered_by_errors = self._build_clustered_result(
            llm_clusterings,
            incomplete_subgoal_analyzed_subgoal_validations
        )

        final_result = ErrorAnalysisResult(
            analyzed_validations_clustered_by_errors=analyzed_validations_clustered_by_errors,
            completed_subgoal_validations=complete_subgoal_analyzed_subgoal_validations
        )
        
        return final_result
    
    async def _analyze(self, data_sample: ErrorAnalysisDataSample) -> List[AnalyzedSubgoalValidation]:
        """
        Analyze an error analysis data sample, obtaining a summarized BASE error for each subgoal validation.
        Processes all subgoal validations within the sample concurrently.
        """
        logging.info(f"Analyzing data sample with ID: {data_sample.data_sample_id}")
        tasks = [
            self._summarize_errors_into_base_error(subgoal_validation)
            for subgoal_validation in data_sample.subgoal_validations
        ]
        base_errors: List[str] = await asyncio.gather(*tasks)
        
        analyzed_subgoal_validations = [
            AnalyzedSubgoalValidation(
                subgoal_validation=subgoal_validation,
                data_sample_id=data_sample.data_sample_id,
                base_error=base_error,
                agent_run_id=data_sample.agent_run_id
            )
            for subgoal_validation, base_error in zip(data_sample.subgoal_validations, base_errors)
        ]
        
        return analyzed_subgoal_validations
    
    async def _summarize_errors_into_base_error(self, subgoal_validation: SubGoalValidationResult) -> str | None:
        """
        Analyze a single subgoal validation to summarize the issues raised by the judge trials.

        If the subgoal validation shows the subgoal as completed, no error summarization is performed, and None is returned.
        If all judge trials show incompletion, a single error summary is generated from the first judge trial explanation.
        If there is a mix of completed and incompleted judge trials, error summaries are generated for each judge trial explanation,
        and majority voting is performed to determine the most likely base error.
        """
        if subgoal_validation.is_completed:
            return None

        judge_trial_explanations = self._get_judge_trial_explanations_from_subgoal_validation(subgoal_validation)
        if self._has_failed_consistently(subgoal_validation):
            # Take first judge trial explanation as representative
            base_error = await self._summarize_error(
                judge_trial_explanations[0],
                subgoal_validation.sub_goal.details
            )
        else:
            # Get error for every judge trial explanation and do majority voting
            tasks = [
                self._summarize_error(judge_trial_explanation, subgoal_validation.sub_goal.details)
                for judge_trial_explanation in judge_trial_explanations
            ]
            errors = await asyncio.gather(*tasks)
            base_error = await self._perform_majority_voting(errors)
        
        return base_error
    
    def _get_judge_trial_explanations_from_subgoal_validation(self, subgoal_validation: SubGoalValidationResult) -> List[str]:
        if len( subgoal_validation.explanations) <= 1:
            raise ValueError("Invalid SubGoalValidationResult.explanation format."
                            "ErrorAnalysis expects SubGoalValidationResult.explanation to have an overall explanation in index 0,"
                            "and judge trial explanations from index 1 onwards.")
        # Strip out the overall explanation to get a list of judge trial explanations
        return subgoal_validation.explanations[1::]
    
    def _has_failed_consistently(self, subgoal_validation: SubGoalValidationResult) -> bool:
        """
        Check if all judge trials for the subgoal validation have failed.
        """
        judge_trials_explanations = self._get_judge_trial_explanations_from_subgoal_validation(subgoal_validation)
        complete_cnt, _, _ = tally_votes(0, 0, 0, judge_trials_explanations)
        # TODO: Add error handling for any instances of invalid_cnt > 0
        return complete_cnt == 0

    async def _summarize_error(self, judge_trial_explanation: str, subgoal: str) -> str:
        """
        Summarize a single judge trial explanation into a concise error description using the LLM.
        """
        payload = LLMPayload(
            user_prompt=UNSUPERVISED_ERROR_SUMMARIZATION_PROMPT_TEMPLATE.format(
                subgoals=subgoal,
                explanation=judge_trial_explanation
            ),
            structured_output=ERROR_SUMMARIZATION_OUTPUT_SCHEMA
        )
        
        response_dict = await self._retry_if_json_decode_error(payload)
        if 'error_type' in response_dict:
            return response_dict['error_type'].strip()
        else:
            raise ToolError(internal_code=ErrorCode.UNSUCCESSFUL_LLM_SUMMARIZATION.value,
                            message=f"LLM error summarization request failed as no error_type found in response: {response_dict}")

    async def _perform_majority_voting(self, errors: List[str]) -> str:
        """
        Perform majority voting on a list of error descriptions to determine the most common error.
        """
        payload = LLMPayload(
            user_prompt=MAJORITY_VOTE_PROMPT_TEMPLATE.format(
                error_type_list=json.dumps(errors, indent=2)
            ),
            structured_output=MAJORITY_VOTE_OUTPUT_SCHEMA
        )
        
        response_dict = await self._retry_if_json_decode_error(payload)
        if 'most_probable_error_type' in response_dict:
            return response_dict['most_probable_error_type'].strip()
        else:
            raise ToolError(internal_code=ErrorCode.UNSUCCESSFUL_MAJORITY_VOTING.value,
                            message=f"LLM majority voting request failed as no most_probable_error_type found in response: {response_dict}")

    def _split_analysed_subgoal_validations_by_completeness(
        self,
        analysed_subgoal_validation_list: List[List[AnalyzedSubgoalValidation]]
    ) -> Tuple[List[AnalyzedSubgoalValidation], List[AnalyzedSubgoalValidation]]:
        """
        Split a list of AnalyzedSubgoalValidation into completed and incompleted lists based on base errors.
        """
        complete_subgoal_analysed_subgoal_validations = []
        incomplete_subgoal_analysed_subgoal_validations = []
        for sublist in analysed_subgoal_validation_list:
            for analysed_subgoal_validation in sublist:
                if analysed_subgoal_validation.base_error is None:
                    complete_subgoal_analysed_subgoal_validations.append(analysed_subgoal_validation)
                else:
                    incomplete_subgoal_analysed_subgoal_validations.append(analysed_subgoal_validation)
        
        return complete_subgoal_analysed_subgoal_validations, incomplete_subgoal_analysed_subgoal_validations

    async def _cluster_errors(self, analyzed_subgoal_validations: List[AnalyzedSubgoalValidation]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Cluster analyzed subgoal validations based on their base errors.
        Returns a mapping from cluster labels to lists of analyzed subgoal validations.
        """
        # Build index -> base_error mapping, and get all unique subgoals to prepare LLM prompt
        index_to_error_mapping = {
            str(idx): analyzed_subgoal_validation.base_error for idx, analyzed_subgoal_validation in enumerate(analyzed_subgoal_validations)
        }
        unique_subgoals = list(dict.fromkeys(
            analysed_subgoal_validation.subgoal_validation.sub_goal.details for analysed_subgoal_validation in analyzed_subgoal_validations
        ))
        
        # Create LLM payload
        payload = LLMPayload(
            user_prompt=CLUSTERING_PROMPT_TEMPLATE.format(
                error_types=json.dumps(index_to_error_mapping, indent=2),
                subgoals=json.dumps(unique_subgoals, indent=2)
            ),
            structured_output=CLUSTERING_OUTPUT_SCHEMA
        )
        
        return await self._retry_if_json_decode_error(payload)

    async def _retry_if_json_decode_error(self, payload: LLMPayload) -> Any:
        """
        Make an LLM request with retry logic for JSON decode errors.
        Runs the LLM call in a thread pool to enable concurrent execution in async contexts.
        
        :param payload: The LLM payload containing the prompt and structured output schema
        :return: Parsed JSON response as a dictionary
        :raises ToolError: If the request fails or max retries are exceeded
        """
        max_retry_json_decode_error = MAX_RETRY_JSON_DECODE_ERROR
        attempt = 0
        while attempt < max_retry_json_decode_error:
            try:
                # Run blocking LLM call in thread pool to enable concurrent execution
                response = await asyncio.to_thread(
                    lambda: asyncio.run(self.llm_client.make_request_with_payload(payload))
                )
                if response.status != STATUS_200:
                    raise ToolError(
                        internal_code=ErrorCode.CLIENT_REQUEST_ERROR.value,
                        message=f"LLM request failed with status {response.status} and error: {response.error_message}"
                    )
                if not response.completion:
                    logging.warning(f"JSON decode error on attempt {attempt + 1}/{max_retry_json_decode_error}: Empty completion received.")
                    attempt += 1
                else:
                    return json.loads(response.completion, strict=False)
            except json.JSONDecodeError as e:
                logging.warning(f"JSON decode error on attempt {attempt + 1}/{max_retry_json_decode_error}: {e}")
                attempt += 1
        raise ToolError(
            internal_code=ErrorCode.INVALID_JSON_DECODE_ERROR.value,
            message="Maximum retry attempts exceeded for JSON decode error."
        )


    def _build_clustered_result(
            self,
            llm_clustering: Dict[str, List[Dict[str, Any]]],
            analyzed_subgoal_validations: List[AnalyzedSubgoalValidation]
        ) -> Dict[str, List[AnalyzedSubgoalValidation]]:
        """
        Given the LLM clustering output and the list of analyzed subgoal validations, 
        maps cluster labels to list of analyzed subgoal validations using the index found in LLM clustering output.
        """
        analyzed_validations_clustered_by_errors = {}
        
        for cluster in llm_clustering['clusters']:
            cluster_label = cluster['cluster_label']
            error_ids = cluster['error_ids']
            
            analyzed_validations_clustered_by_errors[cluster_label] = [
                analyzed_subgoal_validations[int(error_id)]
                for error_id in error_ids
                if int(error_id) < len(analyzed_subgoal_validations)
            ]
        
        return analyzed_validations_clustered_by_errors
