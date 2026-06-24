"""
Client subpackage.

`LLMClient` (the abstract base) is always importable — it has no optional
dependencies. The concrete implementations are imported lazily so that
`import agent_inspect.clients` succeeds even when the optional extras are
not installed:

  * `AzureOpenAIClient` requires the `azure-openai` extra (openai, backoff).
  * `LiteLLMClient`     requires litellm (and openai).

If you reference one of those names without the corresponding dependencies
installed, you will get an ImportError pointing you at the right pip extra.
"""

from typing import TYPE_CHECKING

from .llm_client import LLMClient as LLMClient

_LAZY_ATTRS = {
    "AzureOpenAIClient": (
        ".azure_openai_client",
        "AzureOpenAIClient",
        'pip install "agent_inspect[azure-openai]"',
    ),
    "LiteLLMClient": (
        ".litellm_client",
        "LiteLLMClient",
        "pip install litellm",
    ),
}

__all__ = ["LLMClient", *_LAZY_ATTRS.keys()]


def __getattr__(name: str):
    """PEP 562 hook — resolves lazy attributes on first access."""
    try:
        module_path, attr, hint = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    try:
        module = import_module(module_path, package=__name__)
    except ImportError as exc:
        raise ImportError(
            f"{name} requires optional dependencies that are not installed. "
            f"Install them with: {hint}"
        ) from exc

    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)

if TYPE_CHECKING:
    from .azure_openai_client import AzureOpenAIClient as AzureOpenAIClient
