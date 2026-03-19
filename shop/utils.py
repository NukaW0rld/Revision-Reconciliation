"""Shared utilities for the shop package."""
from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current UTC time as a naïve datetime (no tzinfo).

    Drop-in replacement for the deprecated ``datetime.utcnow()``.  Returns a
    naïve datetime so existing SQLAlchemy ``DateTime`` columns (which store
    naïve UTC strings in SQLite) continue to work without a schema migration.

    If the project ever migrates columns to ``DateTime(timezone=True)``, this
    helper is the single place to update — change the return to
    ``datetime.now(timezone.utc)`` and remove the ``replace``.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
