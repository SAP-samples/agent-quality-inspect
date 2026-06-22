from agent_inspect.core.utils import (
    get_config_or_default,
    tally_votes,
)
from agent_inspect.metrics.constants import (
    COMPLETE_INCOMPLETE_GRADE_PATTERN,
    COMPLETE_INCOMPLETE_PAIR,
)


def test_config_or_default_returns_config_value_when_key_exists():
    config = {"key1": "value1", "key2": "value2"}
    result = get_config_or_default(config, "key1", "default_value")
    assert result == "value1"


def test_config_or_default_returns_default_when_key_does_not_exist():
    config = {"key1": "value1"}
    result = get_config_or_default(config, "key2", "default_value")
    assert result == "default_value"


def test_tally_votes_counts_complete_incomplete_and_invalid():
    completions = ["Grade: C", "Grade: I", "Grade: C", "Invalid Grade"]
    complete_cnt, incomplete_cnt, invalid_cnt = tally_votes(
        0,
        0,
        0,
        completions,
        COMPLETE_INCOMPLETE_GRADE_PATTERN,
        COMPLETE_INCOMPLETE_PAIR,
    )
    assert complete_cnt == 2
    assert incomplete_cnt == 1
    assert invalid_cnt == 1


def test_tally_votes_with_existing_counts():
    completions = ["Grade: C", "Grade: I"]
    complete_cnt, incomplete_cnt, invalid_cnt = tally_votes(
        5,
        3,
        2,
        completions,
        COMPLETE_INCOMPLETE_GRADE_PATTERN,
        COMPLETE_INCOMPLETE_PAIR,
    )
    assert complete_cnt == 6
    assert incomplete_cnt == 4
    assert invalid_cnt == 2
