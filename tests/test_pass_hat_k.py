import pytest
from agent_inspect.exception import EvaluationError
from agent_inspect.metrics.constants import K_VALUE, NO_OF_TRIALS
from agent_inspect.models.metrics import NumericalScore
from agent_inspect.metrics.multi_samples import PassHatK


# --- PassHatK Tests ---
def test_pass_hat_k_all_success():
    metric = PassHatK(config={K_VALUE: 2, NO_OF_TRIALS: 4})
    success_scores = [NumericalScore(1) for _ in range(4)]
    result = metric.compute(success_scores)
    assert result.score == 1.0


def test_pass_hat_k_none_success():
    metric = PassHatK(config={K_VALUE: 2, NO_OF_TRIALS: 4})
    success_scores = [NumericalScore(0) for _ in range(4)]
    result = metric.compute(success_scores)
    assert result.score == 0.0


def test_pass_hat_k_no_num_of_trials_given():
    with pytest.raises(EvaluationError, match="num_trials .* must be provided"):
        PassHatK()


def test_pass_hat_k_no_k_value_given():
    metric = PassHatK(config={NO_OF_TRIALS: 5})
    success_scores = [NumericalScore(1) for _ in range(5)]
    result = metric.compute(success_scores)
    # When k is not given, should default to k=num_trials, so all successes means score is 1.0
    assert result.score == 1.0


def test_pass_hat_k_some_success():
    metric = PassHatK(config={K_VALUE: 2, NO_OF_TRIALS: 4})
    success_scores = [NumericalScore(x) for x in [1, 0, 1, 0]]
    result = metric.compute(success_scores)
    assert abs(result.score - (1 / 6)) < 1e-6


def test_pass_hat_k_typical():
    metric = PassHatK(config={K_VALUE: 3, NO_OF_TRIALS: 5})
    success_scores = [NumericalScore(x) for x in [1, 1, 0, 1, 0]]
    result = metric.compute(success_scores)
    assert abs(result.score - 0.1) < 1e-6


def test_pass_hat_k_error_k_too_large():
    with pytest.raises(EvaluationError, match="k_value .* cannot be greater than num_trials .*"):
        PassHatK(config={K_VALUE: 5, NO_OF_TRIALS: 3})


def test_pass_hat_k_error_success_less_than_k():
    metric = PassHatK(config={K_VALUE: 3, NO_OF_TRIALS: 3})
    success_scores = [NumericalScore(x) for x in [1, 0, 1]]
    result = metric.compute(success_scores)
    assert result.score == 0.0


def test_pass_hat_k_empty_success_flags():
    metric = PassHatK(config={K_VALUE: 2, NO_OF_TRIALS: 4})
    success_scores = []
    with pytest.raises(
        EvaluationError,
        match="Success scores should have the same length as num_trials .*, but got .*",
    ):
        metric.compute(success_scores)


def test_pass_hat_k_error_k_zero():
    with pytest.raises(EvaluationError, match="k_value .* must be greater than 0"):
        PassHatK(config={K_VALUE: 0, NO_OF_TRIALS: 4})


def test_pass_hat_k_error_n_trials_zero():
    with pytest.raises(EvaluationError, match="num_trials .* must be provided"):
        PassHatK(config={K_VALUE: 2, NO_OF_TRIALS: 0})
