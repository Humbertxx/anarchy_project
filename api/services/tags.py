"""Tag reads backing the /tags endpoint."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models import Tag


def list_tags(session: Session) -> list[Tag]:
    """Return all tags ordered by name for filter dropdowns."""
    statement = select(Tag).order_by(Tag.name)
    return list(session.execute(statement).scalars().all())
