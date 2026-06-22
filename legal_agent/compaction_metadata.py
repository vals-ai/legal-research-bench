"""Compaction summary metadata (vendored).

Vendored here because the published ``model-library`` (0.1.25) does not yet
expose ``CompactionSummary``. See ``get_agent.py``.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field, computed_field

from model_library.agent.metadata import SerializableException
from model_library.base.output import QueryResultMetadata


class CompactionSummary(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    turn_number: int | None = None
    input_token_estimate: int | None = None
    threshold_tokens: int | None = None
    summary: str | None = None
    artifacts_subdir: str | None = None
    metadata: QueryResultMetadata | None = None
    error: SerializableException | None = None

    @computed_field
    @property
    def success(self) -> bool:
        return self.error is None
