"""Legal research agent benchmark codebase."""

from .compaction_agent import LegalAgent
from .compaction_config import HistoryCompaction
from .exceptions import RetryExhaustedError, retry_http_errors
from .get_agent import Parameters, build_input, create_llm, get_agent
from .history_compaction import llm_summary_compactor
from .prompt import INSTRUCTIONS_PROMPT, LEGAL_COMPACTION_PROMPT
from .tools import (
    VALID_TOOLS,
    CourtListenerSearch,
    ParseHtmlPage,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
)

__all__ = [
    "CourtListenerSearch",
    "HistoryCompaction",
    "INSTRUCTIONS_PROMPT",
    "LEGAL_COMPACTION_PROMPT",
    "LegalAgent",
    "Parameters",
    "ParseHtmlPage",
    "RetrieveInformation",
    "RetryExhaustedError",
    "SubmitFinalResult",
    "TavilyWebSearch",
    "VALID_TOOLS",
    "build_input",
    "create_llm",
    "get_agent",
    "llm_summary_compactor",
    "retry_http_errors",
]
