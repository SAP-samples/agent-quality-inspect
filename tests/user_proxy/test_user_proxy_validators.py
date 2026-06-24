import pytest

from agent_inspect.exception import InvalidInputValueError
from agent_inspect.models.user_proxy.terminating_condition import (
    TerminatingCondition,
    TaskCompletedTerminatingCondition,
    TaskDelegatedTerminatingCondition,
    TaskBlockedTerminatingCondition,
    DEFAULT_CHECK,
    DEFAULT_STOP_SEQUENCE,
    DEFAULT_BLOCKED_CHECK,
)
from agent_inspect.user_proxy.utils.user_proxy_validators import UserProxyInputValidator


def test_replace_empty_strings_fields_with_defaults_replaces_empty_check_and_stop_sequence():
    condition = TerminatingCondition(check="   ", stop_sequence="   ")
    UserProxyInputValidator.replace_empty_strings_fields_with_defaults(condition)
    assert condition.check == condition.__class__.__dataclass_fields__["check"].default
    assert (
        condition.stop_sequence == condition.__class__.__dataclass_fields__["stop_sequence"].default
    )


def test_replace_empty_strings_fields_with_defaults_does_not_modify_non_empty_fields():
    condition = TerminatingCondition(check="valid check", stop_sequence="valid stop sequence")
    UserProxyInputValidator.replace_empty_strings_fields_with_defaults(condition)
    assert condition.check == "valid check"
    assert condition.stop_sequence == "valid stop sequence"


def test_validate_task_summary_raises_when_none():
    with pytest.raises(InvalidInputValueError) as exc:
        UserProxyInputValidator.validate_task_summary(None)  # type: ignore[arg-type]
    assert "Task summary cannot be empty" in str(exc.value)


def test_validate_task_summary_raises_when_empty_string():
    with pytest.raises(InvalidInputValueError):
        UserProxyInputValidator.validate_task_summary("")


def test_validate_task_summary_raises_when_whitespace_only():
    with pytest.raises(InvalidInputValueError):
        UserProxyInputValidator.validate_task_summary("   ")


def test_validate_task_summary_passes_for_valid_summary():
    UserProxyInputValidator.validate_task_summary("Summarize the user goal")


def test_validate_single_terminating_condition_raises_when_zero_conditions():
    with pytest.raises(InvalidInputValueError) as exc:
        UserProxyInputValidator.validate_single_terminating_condition([])
    assert "Exactly one terminating condition" in str(exc.value)


def test_validate_single_terminating_condition_raises_when_multiple_conditions():
    condition1 = TerminatingCondition(check="c1")
    condition2 = TerminatingCondition(check="c2")
    with pytest.raises(InvalidInputValueError) as exc:
        UserProxyInputValidator.validate_single_terminating_condition([condition1, condition2])
    assert "Exactly one terminating condition" in str(exc.value)


def test_validate_single_terminating_condition_raises_when_subclass_given():
    condition = TaskCompletedTerminatingCondition()
    with pytest.raises(InvalidInputValueError) as exc:
        UserProxyInputValidator.validate_single_terminating_condition([condition])
    assert (
        "Terminating condition must be of type TerminatingCondition and not its subclasses"
        in str(exc.value)
    )


def test_validate_single_terminating_condition_passes_when_single_condition_has_empty_check():
    condition = TerminatingCondition(check=" ")
    UserProxyInputValidator.validate_single_terminating_condition([condition])
    assert condition.check == DEFAULT_CHECK


def test_validate_single_terminating_condition_passes_when_single_condition_has_empty_stop_sequence():
    condition = TerminatingCondition(stop_sequence="")
    UserProxyInputValidator.validate_single_terminating_condition([condition])
    assert condition.stop_sequence == DEFAULT_STOP_SEQUENCE


def test_validate_single_terminating_condition_passes_for_single_valid_condition():
    condition = TerminatingCondition(check="stop when complete")
    UserProxyInputValidator.validate_single_terminating_condition([condition])


def test_validate_multiple_terminating_conditions_raises_when_multiple_completed_conditions():
    c1 = TaskCompletedTerminatingCondition(check="done")
    c2 = TaskCompletedTerminatingCondition(check="also done")
    with pytest.raises(InvalidInputValueError) as exc:
        UserProxyInputValidator.validate_multiple_terminating_conditions([c1, c2])
    assert "Only one task completed terminating condition" in str(exc.value)


def test_validate_multiple_terminating_conditions_raises_when_multiple_delegated_conditions():
    d1 = TaskDelegatedTerminatingCondition(check="delegate 1")
    d2 = TaskDelegatedTerminatingCondition(check="delegate 2")
    with pytest.raises(InvalidInputValueError) as exc:
        UserProxyInputValidator.validate_multiple_terminating_conditions([d1, d2])
    assert "Only one task delegated terminating condition" in str(exc.value)


def test_validate_multiple_terminating_conditions_raises_when_multiple_blocked_conditions():
    b1 = TaskBlockedTerminatingCondition(check="blocked 1")
    b2 = TaskBlockedTerminatingCondition(check="blocked 2")
    with pytest.raises(InvalidInputValueError) as exc:
        UserProxyInputValidator.validate_multiple_terminating_conditions([b1, b2])
    assert "Only one task blocked terminating condition" in str(exc.value)


def test_validate_multiple_terminating_conditions_passes_when_any_condition_has_empty_check():
    valid_completed = TaskCompletedTerminatingCondition(check="done")
    invalid_blocked = TaskBlockedTerminatingCondition(check="  ")
    UserProxyInputValidator.validate_multiple_terminating_conditions(
        [valid_completed, invalid_blocked]
    )
    assert invalid_blocked.check == DEFAULT_BLOCKED_CHECK


def test_validate_multiple_terminating_conditions_passes_for_distinct_types_with_valid_checks():
    completed = TaskCompletedTerminatingCondition(check="done")
    delegated = TaskDelegatedTerminatingCondition(check="delegated")
    blocked = TaskBlockedTerminatingCondition(check="blocked")
    UserProxyInputValidator.validate_multiple_terminating_conditions(
        [completed, delegated, blocked]
    )


def test_validate_multiple_terminating_conditions_raises_when_only_base_condition_given():
    base = TerminatingCondition(check="some check")
    with pytest.raises(InvalidInputValueError) as exc:
        UserProxyInputValidator.validate_multiple_terminating_conditions([base])
    assert (
        "Terminating conditions must be of type TaskCompletedTerminatingCondition, "
        "TaskDelegatedTerminatingCondition, or TaskBlockedTerminatingCondition" in str(exc.value)
    )


def test_validate_multiple_terminating_conditions_raises_when_base_condition_mixed_with_subclasses():
    base = TerminatingCondition(check="some check")
    completed = TaskCompletedTerminatingCondition()
    with pytest.raises(InvalidInputValueError) as exc:
        UserProxyInputValidator.validate_multiple_terminating_conditions([base, completed])
    assert (
        "Terminating conditions must be of type TaskCompletedTerminatingCondition, "
        "TaskDelegatedTerminatingCondition, or TaskBlockedTerminatingCondition" in str(exc.value)
    )


def test_validate_multiple_terminating_conditions_passes_with_empty_list():
    UserProxyInputValidator.validate_multiple_terminating_conditions([])


def test_validate_terminating_conditions_uses_single_validation_when_allow_multiple_false():
    condition = TerminatingCondition(check="single condition")
    UserProxyInputValidator.validate_terminating_conditions([condition], allow_multiple=False)


def test_validate_terminating_conditions_uses_multiple_validation_when_allow_multiple_true():
    completed = TaskCompletedTerminatingCondition(check="done")
    delegated = TaskDelegatedTerminatingCondition(check="delegated")
    UserProxyInputValidator.validate_terminating_conditions(
        [completed, delegated], allow_multiple=True
    )
