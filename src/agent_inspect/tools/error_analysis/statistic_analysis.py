import numpy as np

from agent_inspect.exception.error_codes import ErrorCode, ToolComponent
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.models.tools.analysis_models import ErrorAnalysisDataSample, StatisticAnalysisResult
from agent_inspect.metrics.utils.metrics_utils import map_subgoal_validations_to_binary_matrix



class StatisticAnalysis:
    """
    Method to compute expectation and variance of agent's progress across multiple LLM-as-a-judge runs.

    For each subgoal :math:`g_{i,j} \in G_i`, we define a binary random variable :math:`Z_{i,j}`, where :math:`Z_{i,j}=1` if the agent achieves the :math:`j`-th subgoal (under the given trajectory), and :math:`0` otherwise.

    Let the probability of achieving the subgoal :math:`g_{i,j}` be :math:`Pr(Z_{i,j}=1)=z_{i,j}`. Then, for a given sample with multiple subgoals :math:`(i, G_i)` and an agent trajectory :math:`\\tau_i`, we define the progress as the proportion of subgoals successfully achieved :math:`progress(i, G_i, \\tau_i) = \\frac{\sum_j Z_{i,j}}{|G_i|}`.

    So the expectation and variance of agent's progress are measured by:
    
    .. math::

        E[{progress}(i, G_i, \\tau_i)] = \\frac{\sum_j z_{i,j}}{|G_i|} \ ; \quad
        {Var}[{progress}(i, G_i, \\tau_i)] = \\frac{\sum_j z_{i,j} (1 - z_{i,j})}{|G_i|^2}


    where :math:`z_{i,j}= \\frac{1}{Q}\sum_{q=1}^Q z_{i,j}^{(q)}` is estimated by averaging over :math:`Q` judge runs per subgoal, generalizing the single binary judge output to a probabilistic estimate.
    """

    @staticmethod
    def calculate_probabilities(binary_matrix: list[list[int]]) -> list[float]:
        """
        Calculate probability for each subgoal by computing the mean of each inner list.
        
        Args:
            binary_matrix: 2D list where inner lists cannot have different lengths.
                          Each inner list represents judge scores for a subgoal.
        
        Returns:
            List of probabilities (means) for each subgoal.
        
        Example:
            >>> binary_matrix = [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0]]
            >>> calculate_probabilities(binary_matrix)
            [1.0, 0.6, 0.0]
        """
        # Check if binary_matrix is empty
        if not binary_matrix:
            raise InvalidInputValueError(
                    internal_code=ErrorCode.MISSING_VALUE.value,
                    component_code=ToolComponent.TOOL_ERROR_CODE.value,
                    message="Binary matrix cannot be empty."
                )
        
        # Check if matrix is homogeneous (all inner lists have same length)
        first_length = len(binary_matrix[0])
        for i, subgoal_scores in enumerate(binary_matrix):
            if len(subgoal_scores) != first_length:
                raise InvalidInputValueError(
                    internal_code=ErrorCode.INVALID_VALUE.value,
                    component_code=ToolComponent.TOOL_ERROR_CODE.value,
                    message="Binary matrix must be homogeneous. Expected all inner lists to have same length. Ensure that you run our metrics by disabling judge optimization."
                )
        
        probabilities = []
        for subgoal_scores in binary_matrix:
            if len(subgoal_scores) == 0:
                raise InvalidInputValueError(
                    internal_code=ErrorCode.MISSING_VALUE.value,
                    component_code=ToolComponent.TOOL_ERROR_CODE.value,
                    message="Subgoal must have at least 1 judge scores to compute probability."
                )
            else:
                probability = sum(subgoal_scores) / len(subgoal_scores)
                probabilities.append(probability)
        return probabilities

    @staticmethod
    def compute_statistic_analysis_result(data_sample: ErrorAnalysisDataSample) -> StatisticAnalysisResult:
        """
        Returns the judge expectation and variance on a single data sample.
        
        :param data_sample: The data sample containing subgoal validations.
        :return: an :obj:`~agent_inspect.models.tools.analysis_models.StatisticAnalysisResult` containing judge expectation and variance.
        
        Example:
        
        >>> from agent_inspect.models.tools import ErrorAnalysisDataSample
        >>> from agent_inspect.models.metrics import SubGoalValidationResult
        >>> from agent_inspect.tools import StatisticAnalysis
        >>> data_sample = ErrorAnalysisDataSample(
        ...     data_sample_id=1,
        ...     agent_run_id=101,
        ...     subgoal_validations=[
        ...         # The first element is the summarized explanation, skipped in computation. The rest are truncated judge explanations for demo, which only contain "Grade: I" or "Grade: C".
        ...         SubGoalValidationResult(
        ...             explanations=["Check: {subgoal 1} has failed.'", "Grade: I", "Grade: I", "Grade: I", "Grade: I", "Grade: C"]
        ...         ),
        ...         SubGoalValidationResult(
        ...             explanations=["Check: {subgoal 2} has failed.'", "Grade: I", "Grade: I", "Grade: I", "Grade: I", "Grade: I"]
        ...         )
        ...     ]
        ... )
        >>> stat_result = StatisticAnalysis.compute_statistic_analysis_result(data_sample)
        >>> stat_result.judge_expectation
        0.1
        >>> stat_result.judge_std
        0.2
        """
        judger_binary_matrix = []
        if not data_sample.subgoal_validations:
            return StatisticAnalysisResult(
                data_sample_id=data_sample.data_sample_id,
                agent_run_id=data_sample.agent_run_id,
                subgoal_validations=data_sample.subgoal_validations,
                judge_expectation=None,
                judge_std=None,
            )
        for subgoal_validation in data_sample.subgoal_validations:
            if len(subgoal_validation.explanations) < 2:
                raise InvalidInputValueError(internal_code=ErrorCode.MISSING_VALUE.value, component_code=ToolComponent.TOOL_ERROR_CODE.value, message="Each SubGoalValidationResult must contain at least one judge explanation besides the summarized one.")
            one_subgoal_judge_score = map_subgoal_validations_to_binary_matrix(subgoal_validation.explanations[1::]) # Skip the first explanation which is the summarized one
            judger_binary_matrix.append(one_subgoal_judge_score)
        
        # Convert to numpy array for easier computation
        # Shape: (num_subgoals, num_judges) [[1, 2], [1, 2, 3]]
        subgoal_means = StatisticAnalysis.calculate_probabilities(judger_binary_matrix)
        # Calculate expectation (mean) across all judges for each subgoal
        # Then take the mean across all subgoals
        overall_expectation = np.mean(subgoal_means)  # Mean across all subgoals subgoal len of outer array(binary matrix) 
        # Calculate aggregated standard deviation
        # Following the formula: sqrt(sum(pi * (1 - pi)) / n^2)
        # where pi are the mean across judges for each subgoal and n is the total count
        def aggregate_sd(pi):
            n = len(pi)
            std = np.sqrt(np.sum(pi * (1 - pi)) / (n**2))
            return std
        aggregated_std = aggregate_sd(np.array(subgoal_means))
        return StatisticAnalysisResult(
            data_sample_id=data_sample.data_sample_id,
            agent_run_id=data_sample.agent_run_id,
            subgoal_validations=data_sample.subgoal_validations,
            judge_expectation=float(overall_expectation),
            judge_std=float(aggregated_std),
        )
    
    
