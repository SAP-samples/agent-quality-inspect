from agent_inspect.metrics.validator import exact_match


def test_exact_match_returns_false_for_whitespace_and_empty_string():
    assert exact_match("   ", "", {"trim": True}) is True


def test_exact_match_returns_true_for_case_insensitive_match():
    assert exact_match("HELLO", "hello", {"case_sensitive": False}) is True


def test_exact_match_returns_false_for_case_sensitive_mismatch():
    assert exact_match("HELLO", "hello", {"case_sensitive": True}) is False


def test_exact_match_returns_true_for_trimmed_strings():
    assert exact_match("  hello  ", "hello", {"trim": True}) is True


def test_exact_match_returns_false_for_untrimmed_strings():
    assert exact_match("  hello  ", "hello", {"trim": False}) is False
