"""Run the legal research agent benchmark over one or more questions.

Usage:
  legal-research --questions "Does the FLSA cover ..." --model google/gemini-3.1-pro-preview
  legal-research --question-file data/public.json
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from model_library.agent import AgentResult, TurnSummary
from model_library.base import LLMConfig, QueryResultMetadata
from tqdm.asyncio import tqdm

from .get_agent import DEFAULT_TIME_LIMIT_SECONDS, Parameters, build_input, get_agent
from .tools import VALID_TOOLS

_QUESTION_KEYS = ("question", "Question", "prompt", "text")


def load_questions(path: Path) -> list[str]:
    """Load questions from a file.

    Supports ``.json`` (a list of strings, a list of objects with a question
    field, or an object with a top-level ``tests``/``questions`` list — the
    benchmark's ``data/public.json`` uses ``tests``) and ``.txt`` / other (one
    question per non-empty line).
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("tests") or data.get("questions") or []
        questions: list[str] = []
        for item in data:
            if isinstance(item, str):
                questions.append(item.strip())
                continue
            if isinstance(item, dict):
                for key in _QUESTION_KEYS:
                    if isinstance(item.get(key), str):
                        questions.append(item[key].strip())
                        break
                else:
                    raise ValueError(
                        f"No question field found in JSON item; expected one of {_QUESTION_KEYS}: {item}"
                    )
        return [q for q in questions if q]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _aggregate_metadata(result: AgentResult) -> QueryResultMetadata:
    """Sum metadata across all turns, including tool sub-LLM calls (e.g. retrieve_information)."""
    total = QueryResultMetadata()
    for turn in result.turns:
        if isinstance(turn, TurnSummary):
            total = total + turn.metadata
            for tc in turn.tool_calls:
                if tc.metadata is not None:
                    total = total + tc.metadata
    return total


async def run_questions(
    questions: list[str],
    max_concurrent: int,
    parameters: Parameters,
    log_dir: Path,
) -> list[dict[str, Any]]:
    """Run multiple questions in parallel and write a results.json summary."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(question: str, index: int) -> AgentResult | Exception:
        async with semaphore:
            try:
                agent = get_agent(parameters, log_dir=log_dir)
                return await agent.run(build_input(question), question_id=f"q{index:03d}")
            except Exception as exc:  # keep one bad question from sinking the batch
                return exc

    tasks = [process_question(q, i + 1) for i, q in enumerate(questions)]
    results: list[AgentResult | Exception] = await tqdm.gather(
        *tasks, desc="Processing questions"
    )

    formatted: list[dict[str, Any]] = []
    for question, result in zip(questions, results):
        if isinstance(result, Exception):
            formatted.append(
                {"question": question, "success": False, "error": str(result)}
            )
            print(f"\nFAIL Question failed: {question}\n   Error: {result}\n")
            continue

        meta = _aggregate_metadata(result)
        formatted.append(
            {
                "question": question,
                "success": result.success,
                "final_answer": result.final_answer,
                "total_turns": result.total_turns,
                "tool_usage": result.tool_usage,
                "final_error": result.final_error.model_dump(mode="json")
                if result.final_error
                else None,
                "token_usage": meta.model_dump(mode="json"),
            }
        )
        if not result.success and result.final_error:
            print(
                f"\nFAIL Question failed: {question}\n   Turns: {result.total_turns}\n"
                f"   Error: [{result.final_error.type}] {result.final_error.message}\n"
            )
        else:
            print(
                f"\nOK Question succeeded: {question}\n   Turns: {result.total_turns}\n"
                f"   Result: {result.final_answer}\n"
            )

    log_dir.mkdir(parents=True, exist_ok=True)
    results_file = log_dir / "results.json"
    results_file.write_text(json.dumps(formatted, indent=2))
    print(f"\nResults saved to: {results_file}")

    return formatted


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the harness for the legal research agent benchmark"
    )
    parser.add_argument(
        "--questions", type=str, nargs="+", help="One or more questions to process"
    )
    parser.add_argument(
        "--question-file",
        type=Path,
        help="Path to a question file (.json list/objects, or .txt with one question per line)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-3.1-pro-preview",
        help="Model to use to generate completions",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=VALID_TOOLS,
        choices=VALID_TOOLS,
        help="List of tools to make available to the agent",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="Max tokens for completion generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=DEFAULT_TIME_LIMIT_SECONDS,
        help="Maximum wall-clock time in seconds before the agent stops (default: 3 hours)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional turn cap for local testing (default: no turn limit, time-bounded only)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of questions to process concurrently",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory where per-question agent logs and results.json are written",
    )
    args = parser.parse_args()

    load_dotenv(override=True, dotenv_path=Path(".env"))

    if args.question_file:
        questions = load_questions(args.question_file)
    elif args.questions:
        questions = args.questions
    else:
        parser.error(
            "No questions provided. Use one of --question-file or --questions."
        )

    parameters = Parameters(
        model_name=args.model,
        tools=args.tools,
        max_time_seconds=args.max_time,
        max_turns=args.max_turns,
        llm_config=LLMConfig(max_tokens=args.max_tokens, temperature=args.temperature),
    )

    await run_questions(
        questions=questions,
        max_concurrent=args.parallelism,
        parameters=parameters,
        log_dir=args.log_dir,
    )


def main_sync() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
