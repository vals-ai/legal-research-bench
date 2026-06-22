"""LLM-summary history compaction (vendored).

Vendored here because the published ``model-library`` (0.1.25) does not yet
expose ``llm_summary_compactor``. See ``get_agent.py``.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from model_library.agent.metadata import SerializableException
from model_library.base.base import LLM
from model_library.base.input import InputItem, SystemInput, TextInput, ToolDefinition
from model_library.base.output import QueryResultMetadata
from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_model_input_context_window

from .compaction_config import (
    DEFAULT_COMPACTION_PROMPT,
    DEFAULT_SUMMARY_PREFIX,
    HistoryCompaction,
    truncate_oldest,
)
from .compaction_metadata import CompactionSummary

if TYPE_CHECKING:
    from model_library.agent.metadata import AgentTurn

MIN_COMPACTION_THRESHOLD_HEADROOM_TOKENS = 1024


async def estimate_next_input_tokens(
    llm: LLM,
    previous_turn: "AgentTurn",
) -> int:
    metadata = previous_turn.query_result.metadata
    estimate = metadata.total_input_tokens + metadata.total_output_tokens
    if previous_turn.tool_call_records:
        encoding = await llm.get_encoding()
        for record in previous_turn.tool_call_records:
            estimate += len(
                encoding.encode(str(record.tool_output.output), disallowed_special=())
            )
    return estimate


class EmptyCompactionSummaryError(RuntimeError):
    """The compaction LLM returned no usable text after stripping whitespace."""


class CompactionHook(Protocol):
    async def __call__(
        self,
        history: list[InputItem],
        *,
        state: dict[str, Any],
        trigger: Literal["each_turn", "max_context"],
        turn_number: int,
        compaction_number: int,
        tools: list[ToolDefinition],
        previous_turn: "AgentTurn | None",
        output_dir: Path,
        question_id: str,
        run_id: str | None,
        logger: logging.Logger,
    ) -> tuple[list[InputItem], CompactionSummary | None]: ...


async def apply_compaction(
    hook: CompactionHook | None,
    history: list[InputItem],
    *,
    trigger: Literal["each_turn", "max_context"],
    state: dict[str, Any],
    compactions: list[CompactionSummary],
    turn_number: int,
    tools: list[ToolDefinition],
    previous_turn: "AgentTurn | None",
    output_dir: Path,
    question_id: str,
    run_id: str | None,
    logger: logging.Logger,
) -> tuple[list[InputItem], CompactionSummary | None]:
    if hook is None:
        return history, None
    new_history, summary = await hook(
        history,
        state=state,
        trigger=trigger,
        turn_number=turn_number,
        compaction_number=len(compactions) + 1,
        tools=tools,
        previous_turn=previous_turn,
        output_dir=output_dir,
        question_id=question_id,
        run_id=run_id,
        logger=logger,
    )
    if summary is not None:
        compactions.append(summary)
    return new_history, summary


def resolve_threshold_tokens(llm: LLM, cfg: HistoryCompaction) -> int | None:
    registry_key = llm._registry_key  # pyright: ignore[reportPrivateUsage]
    # ``input_context_window`` is not an attribute on published model-library (0.1.25);
    # fall back to the registry lookup below. ``getattr`` keeps this forward-compatible
    # with versions that do expose an instance-level override.
    input_context_window = getattr(llm, "input_context_window", None)
    ctx = input_context_window if isinstance(input_context_window, int) else None
    if ctx is None and registry_key:
        ctx = get_model_input_context_window(registry_key)
    if ctx is None:
        return None
    if cfg.threshold_tokens is not None:
        threshold = cfg.threshold_tokens
    else:
        assert cfg.threshold_percentage is not None
        threshold = int(ctx * cfg.threshold_percentage)
    return min(threshold, ctx - MIN_COMPACTION_THRESHOLD_HEADROOM_TOKENS)


def llm_summary_compactor(
    llm: LLM,
    cfg: HistoryCompaction,
    *,
    prompt: str = DEFAULT_COMPACTION_PROMPT,
    summary_prefix: str = DEFAULT_SUMMARY_PREFIX,
) -> CompactionHook | None:
    threshold = resolve_threshold_tokens(llm, cfg)
    if threshold is None:
        return None

    async def _do_compact(
        history: list[InputItem],
        *,
        turn_number: int,
        compaction_number: int,
        input_token_estimate: int | None,
        context_retry_limit: int,
        tools: list[ToolDefinition],
        output_dir: Path,
        question_id: str,
        run_id: str | None,
        logger: logging.Logger,
    ) -> tuple[list[InputItem], CompactionSummary]:
        del tools
        attempt = history
        summary = ""
        metadata: QueryResultMetadata | None = None
        context_retries = 0
        try:
            while not summary:
                try:
                    response = await llm.query(
                        input=[*attempt, TextInput(text=prompt)],
                        tools=[],
                        question_id=question_id,
                        run_id=run_id,
                        logger=logger,
                        in_agent=True,
                    )
                    summary = (response.output_text or "").strip()
                    metadata = response.metadata
                    if not summary:
                        raise EmptyCompactionSummaryError(
                            "Compaction LLM returned empty output"
                        )
                except MaxContextWindowExceededError:
                    context_retries += 1
                    if context_retries > context_retry_limit:
                        raise
                    attempt = truncate_oldest(attempt)
        except Exception as e:
            logger.warning(
                "History compaction failed; continuing with original history",
                exc_info=True,
            )
            return history, CompactionSummary(
                turn_number=turn_number,
                input_token_estimate=input_token_estimate,
                threshold_tokens=threshold,
                metadata=metadata,
                error=SerializableException.from_exception(e),
            )

        new_history: list[InputItem] = [
            *(item for item in history if isinstance(item, SystemInput)),
            TextInput(text=summary_prefix + summary),
        ]
        artifacts = output_dir / "compactions" / f"compaction_{compaction_number:03d}"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "previous_history.bin").write_bytes(LLM.serialize_input(history))
        rel = str(artifacts.relative_to(output_dir))
        logger.info(
            "History compacted: %s -> %s items | summary=%r",
            len(history),
            len(new_history),
            summary[:500],
        )
        return new_history, CompactionSummary(
            turn_number=turn_number,
            input_token_estimate=input_token_estimate,
            threshold_tokens=threshold,
            summary=summary,
            artifacts_subdir=rel,
            metadata=metadata,
        )

    async def hook(
        history: list[InputItem],
        *,
        state: dict[str, Any],
        trigger: Literal["each_turn", "max_context"],
        turn_number: int,
        compaction_number: int,
        tools: list[ToolDefinition],
        previous_turn: "AgentTurn | None",
        output_dir: Path,
        question_id: str,
        run_id: str | None,
        logger: logging.Logger,
    ) -> tuple[list[InputItem], CompactionSummary | None]:
        if previous_turn is not None:
            try:
                state["next_input_estimate"] = await estimate_next_input_tokens(
                    llm, previous_turn
                )
            except Exception:
                logger.warning(
                    "Failed to estimate next input tokens; using previous estimate",
                    exc_info=True,
                )
        next_input_estimate: int = state.get("next_input_estimate", 0)
        consecutive_failures: int = state.get("consecutive_failures", 0)

        if trigger == "each_turn":
            if next_input_estimate < threshold:
                return history, None
            if consecutive_failures >= cfg.max_failures:
                return history, None
            input_token_estimate: int | None = next_input_estimate
            context_retry_limit = cfg.max_compaction_context_retries
            logger.info(
                f"Compacting history: {next_input_estimate:,} >= {threshold:,} tokens"
            )
        else:
            if not cfg.compact_on_max_context:
                return history, None
            input_token_estimate = None
            context_retry_limit = cfg.max_compaction_context_retries
            logger.warning("Query exceeded context window; compacting")
            try:
                history = truncate_oldest(history)
            except ValueError:
                pass

        new_history, summary = await _do_compact(
            history,
            turn_number=turn_number,
            compaction_number=compaction_number,
            input_token_estimate=input_token_estimate,
            context_retry_limit=context_retry_limit,
            tools=tools,
            output_dir=output_dir,
            question_id=question_id,
            run_id=run_id,
            logger=logger,
        )
        if summary.success:
            state["next_input_estimate"] = 0
            state["consecutive_failures"] = 0
        elif trigger == "each_turn":
            state["consecutive_failures"] = consecutive_failures + 1
        return new_history, summary

    return hook
