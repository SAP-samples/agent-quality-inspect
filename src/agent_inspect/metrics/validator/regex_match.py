from typing import Optional, Dict, Any
import re

from agent_inspect.metrics.utils.metrics_utils import get_config_or_default

def regex_match(candidate: str, pattern: str, config: Optional[Dict[str, Any]] = None) -> bool:
    mode = get_config_or_default(config, "mode", "substring")
    flags = 0
    if mode == "full":
        return re.fullmatch(pattern, candidate, flags) is not None
    else:
        return re.search(pattern, candidate, flags) is not None