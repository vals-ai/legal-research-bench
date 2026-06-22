"""Build the legal-research agent.

History compaction is vendored under ``legal_agent/`` (``compaction_agent.py``,
``history_compaction.py``, ``compaction_config.py``, ``compaction_metadata.py``)
because the published ``model-library`` (0.1.25) does not yet expose
``HistoryCompaction`` / ``llm_summary_compactor``. ``LegalAgent`` is a thin
``Agent`` subclass that runs that compaction loop.
"""

from pathlib import Path

from model_library.agent import (
    AgentConfig,
    AgentHooks,
    TimeLimit,
    ToolCallRecord,
    TurnLimit,
    TurnResult,
    default_before_query,
)
from model_library.base import LLM, LLMConfig, RawResponse, TextInput
from model_library.base.input import InputItem
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel

from .compaction_agent import LegalAgent
from .compaction_config import DEFAULT_SUMMARY_PREFIX, HistoryCompaction
from .exceptions import RetryExhaustedError
from .history_compaction import llm_summary_compactor
from .prompt import INSTRUCTIONS_PROMPT, LEGAL_COMPACTION_PROMPT
from .tools import (
    VALID_TOOLS,
    CourtListenerSearch,
    ParseHtmlPage,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
    Tool,
)

DEFAULT_TIME_LIMIT_SECONDS = 10800  # 3 hours


class Parameters(BaseModel):
    model_name: str
    tools: list[str] = VALID_TOOLS
    llm_config: LLMConfig
    max_time_seconds: int = DEFAULT_TIME_LIMIT_SECONDS
    max_turns: int | None = None


def build_input(question: str) -> list[InputItem]:
    """Format a single question into the agent's initial input."""
    return [TextInput(text=INSTRUCTIONS_PROMPT.format(question=question))]


def create_llm(parameters: Parameters) -> LLM:
    """Create an LLM instance from parameters using the model registry."""
    return get_registry_model(parameters.model_name, parameters.llm_config)


def get_agent(
    parameters: Parameters,
    llm: LLM | None = None,
    log_dir: Path | None = None,
) -> LegalAgent:
    """Build the legal-research agent.

    Time-bounded (no turn limit unless ``parameters.max_turns`` is set) and
    configured to keep running through text-only model turns, so the agent only
    terminates when ``submit_final_result`` is called or the time/turn budget
    runs out.
    """
    if llm is None:
        llm = create_llm(parameters)

    available_tools: dict[str, type[Tool]] = {
        "web_search": TavilyWebSearch,
        "retrieve_information": RetrieveInformation,
        "parse_html_page": ParseHtmlPage,
        "courtlistener_search": CourtListenerSearch,
    }

    selected_tools: list[Tool] = []
    for tool_name in parameters.tools:
        if tool_name not in available_tools:
            raise ValueError(
                f"Tool {tool_name} not found. Available: {list(available_tools.keys())}"
            )
        tool_cls = available_tools[tool_name]
        if tool_name == "retrieve_information":
            selected_tools.append(tool_cls(llm=llm))  # type: ignore[call-arg]
        else:
            selected_tools.append(tool_cls())  # type: ignore[call-arg]

    selected_tools.append(SubmitFinalResult())

    def _before_query(
        history: list[InputItem], last_error: Exception | None
    ) -> list[InputItem]:
        """Nudge the model to call a tool when its previous turn produced no tool call."""
        if history and isinstance(history[-1], RawResponse):
            history.append(
                TextInput(
                    text=(
                        "Your last response produced no tool call. "
                        "Call `submit_final_result` if you have a final result, "
                        "otherwise continue with the next tool call."
                    )
                )
            )
        return default_before_query(history, last_error)

    def _should_stop(turn_result: TurnResult) -> bool:
        # Never stop on text-only turns; only ``submit_final_result`` (done_tool)
        # or the configured time/turn limit should end the run.
        return False

    def _on_tool_result(record: ToolCallRecord, state: dict) -> None:
        # Rate-limit exhaustion is infra failure, not model failure — surface it as an error.
        if record.error and record.error.type == "RetryExhaustedError":
            raise RetryExhaustedError(record.error.message)

    history_compaction = HistoryCompaction(
        threshold_percentage=0.80,
        compact_on_max_context=True,
    )
    compaction_hook = llm_summary_compactor(
        llm,
        history_compaction,
        prompt=LEGAL_COMPACTION_PROMPT,
        summary_prefix=DEFAULT_SUMMARY_PREFIX,
    )

    return LegalAgent(
        llm=llm,
        tools=selected_tools,
        name="legal",
        log_dir=log_dir or Path("logs"),
        config=AgentConfig(
            turn_limit=TurnLimit(max_turns=parameters.max_turns)
            if parameters.max_turns
            else None,
            time_limit=TimeLimit(max_seconds=parameters.max_time_seconds),
        ),
        hooks=AgentHooks(
            before_query=_before_query,
            should_stop=_should_stop,
            on_tool_result=_on_tool_result,
        ),
        compaction_hook=compaction_hook,
    )
