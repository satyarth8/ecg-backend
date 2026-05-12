"""
database.py
─────────────────────────────────────────────────────────────────────────────
MongoDB Atlas connection singleton.

Usage (anywhere in the codebase):
    from database import get_db, collections

    db = get_db()
    collections.users.insert_one({...})
    collections.ecg_summaries.find({...})

Design:
  - MongoClient is created ONCE (module-level singleton) on first import.
  - All collection accessors are exposed via `collections` namespace object.
  - Works for BOTH the RPi edge server and the Render cloud API.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("ECGDatabase")

# ── Singleton client & db ─────────────────────────────────────────────────
_client: MongoClient | None = None
_db = None

DB_NAME = "ecg_db"


def get_client() -> MongoClient:
    """Return the singleton MongoClient, creating it on first call."""
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise ConfigurationError(
                "MONGO_URI environment variable is not set. "
                "Copy .env.example → .env and fill in your Atlas connection string."
            )
        log.info("Connecting to MongoDB Atlas …")
        _client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
        # Trigger a real connection to catch bad URI early
        _client.admin.command("ping")
        log.info("MongoDB Atlas connected ✓")
    return _client


def get_db():
    """Return the ecg_db database object."""
    global _db
    if _db is None:
        _db = get_client()[DB_NAME]
    return _db


# ── Collection accessor namespace ─────────────────────────────────────────
class _Collections:
    """
    Lazy accessor for all ECG project collections.
    Usage:  from database import collections
            collections.alerts.find({...})
    """

    @property
    def users(self):
        return get_db()["users"]

    @property
    def patients(self):
        return get_db()["patients"]

    @property
    def devices(self):
        return get_db()["devices"]

    @property
    def ecg_summaries(self):
        return get_db()["ecg_summaries"]

    @property
    def alerts(self):
        return get_db()["alerts"]


collections = _Collections()
