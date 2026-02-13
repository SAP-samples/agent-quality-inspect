from typing import Optional, Dict, Any

from agent_inspect.metrics.utils.metrics_utils import get_config_or_default

def exact_match(candidate: str, ground_truth: str, config: Optional[Dict[str, Any]] = None) -> bool:
    trim = get_config_or_default(config, "trim", True)
    case_sensitive = get_config_or_default(config, "case_sensitive", True)
    if trim:
        candidate = candidate.strip()
        ground_truth = ground_truth.strip()
    if not case_sensitive:
        candidate = candidate.lower()
        ground_truth = ground_truth.lower()
    return candidate == ground_truth



