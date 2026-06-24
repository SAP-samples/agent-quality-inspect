from agent_inspect.metrics.validator import regex_match


def test_regex_match_returns_true_for_full_match():
    assert regex_match("hello", "hello", {"mode": "full"}) is True


def test_regex_match_returns_false_for_partial_match_in_full_mode():
    assert regex_match("hello world", "hello", {"mode": "full"}) is False


def test_regex_match_returns_true_for_substring_match():
    assert regex_match("hello world", "hello", {"mode": "substring"}) is True


def test_regex_match_returns_false_for_no_match():
    assert regex_match("goodbye", "hello") is False


def test_regex_match_trims_candidate_string_by_default():
    assert regex_match("  hello  ", "hello") is True


def test_regex_match_handles_empty_candidate_string():
    assert regex_match("", "hello") is False


def test_regex_match_handles_empty_pattern():
    assert regex_match("hello", "") is True


def test_regex_match_handles_empty_pattern_mode_full():
    assert regex_match("hello", "", {"mode": "full"}) is False


def test_regex_match_handles_both_empty_strings():
    assert regex_match("", "") is True
