"""History-compaction configuration (vendored).

Vendored here because the published ``model-library`` (0.1.25) does not yet
expose ``HistoryCompaction`` / ``truncate_oldest``. See ``get_agent.py``.
"""

from pydantic import BaseModel, field_validator, model_validator

from model_library.base.input import InputItem, RawResponse

DEFAULT_COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.
"""

DEFAULT_SUMMARY_PREFIX = (
    "Another LLM agent started to solve this task and produced this summary of its "
    "progress. Use the information here to continue the work without duplicating effort.\n\n"
)


def truncate_oldest(history: list[InputItem]) -> list[InputItem]:
    """Remove the oldest exchange (response + following inputs) from history."""
    preamble: list[InputItem] = []
    i = 0
    while i < len(history) and not isinstance(history[i], RawResponse):
        preamble.append(history[i])
        i += 1

    rest = history[i:]
    if not rest:
        raise ValueError("No prior exchange found to truncate.")

    i = 0
    while i < len(rest) and isinstance(rest[i], RawResponse):
        i += 1

    while i < len(rest) and not isinstance(rest[i], RawResponse):
        i += 1

    return preamble + list(rest[i:])


class HistoryCompaction(BaseModel):
    threshold_tokens: int | None = None
    threshold_percentage: float | None = None
    max_failures: int = 2
    compact_on_max_context: bool = False
    max_compaction_context_retries: int = 2

    @field_validator("threshold_tokens", mode="before")
    @classmethod
    def _validate_threshold_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"threshold_tokens must be >= 1, got {v}")
        return v

    @field_validator("threshold_percentage", mode="before")
    @classmethod
    def _validate_threshold_percentage(cls, v: float | None) -> float | None:
        if v is not None and not 0 < v <= 0.90:
            raise ValueError(f"threshold_percentage must be in (0, 0.90], got {v}")
        return v

    @field_validator("max_failures", mode="before")
    @classmethod
    def _validate_max_failures(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_failures must be >= 1, got {v}")
        return v

    @field_validator("max_compaction_context_retries", mode="before")
    @classmethod
    def _validate_max_compaction_context_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_compaction_context_retries must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_one_threshold(self) -> "HistoryCompaction":
        if self.threshold_tokens is not None and self.threshold_percentage is not None:
            raise ValueError(
                "Specify exactly one of threshold_tokens or threshold_percentage."
            )
        if self.threshold_tokens is None and self.threshold_percentage is None:
            self.threshold_percentage = 0.85
        return self
