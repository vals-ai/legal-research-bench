"""``Agent`` subclass that runs vendored history compaction on model-library 0.1.x.

Adds proactive (per-turn) and overflow-triggered history compaction to the
standard agent loop. See ``get_agent.py`` for how it is wired up.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Any

from model_library.agent import Agent, AgentResult, TurnResult
from model_library.agent.metadata import (
    AgentTurn,
    ErrorTurn,
    SerializableException,
    TurnSummary,
)
from model_library.agent.tool import Tool
from model_library.base.base import LLM
from model_library.base.input import InputItem
from model_library.exceptions import MaxContextWindowExceededError

from .history_compaction import CompactionHook, apply_compaction


class LegalAgent(Agent):
    """``Agent`` with proactive and overflow-triggered history compaction."""

    def __init__(
        self,
        llm: LLM,
        tools: Sequence[Tool],
        *,
        name: str,
        log_dir: Path,
        config,
        hooks,
        compaction_hook: CompactionHook | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            name=name,
            log_dir=log_dir,
            config=config,
            hooks=hooks,
        )
        self._compaction_hook = compaction_hook

    async def _run(
        self,
        input: Sequence[InputItem],
        *,
        question_id: str,
        run_id: str | None = None,
        state: dict[str, Any] | None = None,
        output_dir: Path,
        logger: Logger,
        docent_ingest: bool = False,
        atif_export: bool = False,
    ) -> AgentResult:
        if state is None:
            state = {}

        history: list[InputItem] = list(input)
        self._write_init_dir(output_dir, input, state, logger)

        turns: list[TurnSummary | ErrorTurn] = []
        raw_turns: list[AgentTurn | ErrorTurn] = []
        compactions = []
        compaction_state: dict[str, Any] = {}

        start_time = time.monotonic()
        final_error: SerializableException | None = None
        last_query_error: Exception | None = None

        turn_limit = self._config.turn_limit
        time_limit = self._config.time_limit
        turn_number = 0
        retry_overhead = 0.0

        try:
            while turn_limit is None or turn_number < turn_limit.max_turns:
                turn_start = time.monotonic()
                turn_number += 1

                logger.info(
                    f"Turn {turn_number}/{turn_limit.max_turns if turn_limit else '?'} starting"
                )

                if turn_number > 1:
                    if last_query_error is not None:
                        logger.debug(
                            f"before_query: handling error {type(last_query_error).__name__}: {last_query_error}"
                        )
                    new_history = self._hooks.before_query(history, last_query_error)
                    last_query_error = None
                    if new_history != history:
                        logger.debug(
                            f"before_query modified history: {len(history)} → {len(new_history)} items"
                        )
                    history = new_history

                prior = raw_turns[-1] if raw_turns else None
                history, _ = await apply_compaction(
                    self._compaction_hook,
                    history,
                    trigger="each_turn",
                    state=compaction_state,
                    compactions=compactions,
                    turn_number=turn_number,
                    tools=self._tool_defs,
                    previous_turn=prior if isinstance(prior, AgentTurn) else None,
                    output_dir=output_dir,
                    question_id=question_id,
                    run_id=run_id,
                    logger=logger,
                )

                elapsed = time.monotonic() - start_time
                effective_elapsed = (
                    elapsed
                    if (time_limit is not None and time_limit.include_retries)
                    else elapsed - retry_overhead
                )
                if (
                    time_limit is not None
                    and effective_elapsed >= time_limit.max_seconds
                ):
                    final_error = SerializableException(
                        type="MaxTimeExceeded",
                        message="Max time reached",
                        context={
                            "elapsed_seconds": elapsed,
                            "effective_elapsed_seconds": effective_elapsed,
                            "retry_overhead_seconds": retry_overhead,
                            "max_seconds": time_limit.max_seconds,
                        },
                    )
                    logger.warning(str(final_error))
                    break

                injected_turn_msgs: list[InputItem] = []
                if turn_limit is not None and turn_limit.turn_message is not None:
                    msg = turn_limit.turn_message(turn_number, turn_limit.max_turns)
                    if msg is not None:
                        logger.debug(
                            f"Hook turn_message({turn_number}/{turn_limit.max_turns}): {msg}"
                        )
                        injected_turn_msgs.append(msg)
                if time_limit is not None and time_limit.time_message is not None:
                    msg = time_limit.time_message(
                        effective_elapsed, time_limit.max_seconds
                    )
                    if msg is not None:
                        logger.debug(
                            f"Hook time_message({effective_elapsed:.3f}s/{time_limit.max_seconds}s): {msg}"
                        )
                        injected_turn_msgs.append(msg)

                if turn_limit is not None and turn_limit.tool_filter is not None:
                    tools_for_turn = turn_limit.tool_filter(
                        turn_number, turn_limit.max_turns, self._tool_defs
                    )
                else:
                    tools_for_turn = self._tool_defs

                try:
                    query_start = time.monotonic()
                    try:
                        response = await self._llm.query(
                            input=history + injected_turn_msgs,
                            tools=tools_for_turn,
                            question_id=question_id,
                            run_id=run_id,
                            logger=logger,
                            in_agent=True,
                        )
                    except MaxContextWindowExceededError:
                        history, summary = await apply_compaction(
                            self._compaction_hook,
                            history,
                            trigger="max_context",
                            state=compaction_state,
                            compactions=compactions,
                            turn_number=turn_number,
                            tools=self._tool_defs,
                            previous_turn=None,
                            output_dir=output_dir,
                            question_id=question_id,
                            run_id=run_id,
                            logger=logger,
                        )
                        if summary is None or not summary.success:
                            raise
                        query_start = time.monotonic()
                        response = await self._llm.query(
                            input=history + injected_turn_msgs,
                            tools=tools_for_turn,
                            question_id=question_id,
                            run_id=run_id,
                            logger=logger,
                            in_agent=True,
                        )
                    query_wall = time.monotonic() - query_start
                    turn_retry_overhead = max(
                        0.0, query_wall - response.metadata.default_duration_seconds
                    )
                    retry_overhead += turn_retry_overhead
                except Exception as query_error:
                    logger.warning(f"Query failed: {query_error}", exc_info=True)
                    turn_duration = time.monotonic() - turn_start
                    error_turn = ErrorTurn(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        error=SerializableException.from_exception(query_error),
                        duration_seconds=turn_duration,
                    )
                    turns.append(error_turn)
                    raw_turns.append(error_turn)
                    self._write_error_turn_dir(
                        output_dir, turn_number, error_turn, logger
                    )
                    last_query_error = query_error
                    continue

                history = list(response.history)
                tool_call_records = await self._execute_tool_calls(
                    response.tool_calls, state, history, logger
                )
                turn_duration = time.monotonic() - turn_start

                logger.info(
                    f"Turn {turn_number}/{turn_limit.max_turns if turn_limit else '?'} | {len(tool_call_records)} tool calls"
                    + f" | in: {response.metadata.total_input_tokens}, out: {response.metadata.total_output_tokens}"
                    + f", cost:{response.metadata.cost.total if response.metadata.cost else '?'}"
                )
                for record in tool_call_records:
                    icon = "✓" if record.tool_output.error is None else "✗"
                    error_str = (
                        f" {record.tool_output.error}"
                        if record.tool_output.error
                        else ""
                    )
                    logger.info(
                        f"  {icon} {record.tool_call.name} ({record.duration_seconds}s){error_str}"
                    )

                turn = AgentTurn(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    query_result=response,
                    tool_call_records=tool_call_records,
                    duration_seconds=turn_duration,
                    retry_overhead_seconds=turn_retry_overhead,
                )
                self._write_turn_dir(
                    output_dir, turn_number, turn, state, history, logger
                )
                turns.append(turn.to_summary())
                raw_turns.append(turn)

                if any(record.tool_output.done for record in tool_call_records):
                    logger.info("Stop: tool signaled done")
                    break

                elapsed = time.monotonic() - start_time
                turn_result = TurnResult(
                    turn_number=turn_number,
                    turn=turn,
                    state=state,
                    elapsed_seconds=elapsed,
                )
                if self._hooks.should_stop(turn_result):
                    logger.info("Stop: should_stop hook returned True")
                    break
            else:
                max_turns = turn_limit.max_turns if turn_limit else None
                final_error = SerializableException(
                    type="MaxTurnsExceeded",
                    message=f"Max turns ({max_turns}) reached",
                    context={"max_turns": max_turns},
                )
                logger.warning(str(final_error))
        except Exception as exc:
            final_error = SerializableException.from_exception(exc)
            logger.error(f"Agent loop failed: {final_error}", exc_info=True)

        elapsed = time.monotonic() - start_time
        answer = self._hooks.determine_answer(state, raw_turns, final_error)

        if docent_ingest:
            if not run_id:
                logger.warning("Docent ingestion skipped: run_id not provided")
            else:
                try:
                    from model_library.docent import (
                        agent_turns_to_docent_agent_run,
                        ingest,
                    )

                    ingest(
                        run_id,
                        agent_turns_to_docent_agent_run(
                            raw_turns,
                            question_id,
                            answer,
                        ),
                    )
                except Exception:
                    logger.warning("Docent ingestion failed", exc_info=True)

        if atif_export:
            try:
                import json

                from model_library.atif import ATIFTrajectory

                tool_defs: list[dict[str, Any]] | None = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.definition.body.name,
                            "description": tool.definition.body.description,
                            "parameters": {
                                "type": "object",
                                "properties": tool.definition.body.properties,
                                "required": tool.definition.body.required,
                            },
                        },
                    }
                    for tool in self._tools.values()
                ] or None

                reasoning_effort: str | float | None = self._llm.reasoning_effort
                if isinstance(reasoning_effort, bool):
                    reasoning_effort = 1.0 if reasoning_effort else 0.0

                agent_extra: dict[str, Any] | None = {
                    k: v
                    for k, v in {
                        "max_tokens": self._llm.max_tokens,
                        "temperature": self._llm.temperature,
                        "top_p": self._llm.top_p,
                        "top_k": self._llm.top_k,
                        "reasoning": self._llm.reasoning or None,
                        "compute_effort": self._llm.compute_effort,
                        "reasoning_effort": reasoning_effort,
                    }.items()
                    if v is not None
                } or None

                trajectory = ATIFTrajectory.from_agent_result(
                    turns=raw_turns,
                    agent_name=self._name,
                    model_name=self._llm.model_name,
                    tool_definitions=tool_defs,
                    reasoning_effort=reasoning_effort,
                    agent_extra=agent_extra,
                )
                trajectory_path = output_dir / "trajectory_atif.json"
                trajectory_path.write_text(
                    json.dumps(trajectory.to_json_dict(), indent=2, default=str)
                )
            except Exception:
                logger.warning("ATIF export failed", exc_info=True)

        result = AgentResult(
            final_answer=answer,
            final_error=final_error,
            final_history=history,
            turns=turns,
            final_duration_seconds=elapsed,
            output_dir=output_dir,
        )
        logger.debug(f"Run complete: {result!r}")

        try:
            result_path = output_dir / "result.json"
            result_path.write_text(result.model_dump_json(indent=2))
        except Exception:
            logger.exception("Failed to serialize result")

        return result
