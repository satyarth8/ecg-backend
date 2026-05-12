"""
init_db.py
─────────────────────────────────────────────────────────────────────────────
One-time MongoDB Atlas setup script.

Run ONCE to:
  1. Create all collections with JSON Schema validators.
  2. Create indexes for performance (patient_id, timestamp, device_id).

Safe to re-run — uses create_collection with try/except, so it won't
overwrite existing data.

Usage:
    # From inside ecg-backend directory, with venv active:
    python init_db.py

Requires MONGO_URI to be set in a .env file (copy from .env.example).
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import logging
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid

from database import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("InitDB")


# ══════════════════════════════════════════════════════════════════════════
# Collection Schemas (MongoDB JSON Schema validators)
# ══════════════════════════════════════════════════════════════════════════

USERS_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["username", "email", "password_hash", "role", "created_at"],
        "properties": {
            "username":      {"bsonType": "string", "description": "must be a string"},
            "email":         {"bsonType": "string", "description": "must be a string"},
            "password_hash": {"bsonType": "string", "description": "bcrypt hash"},
            "role": {
                "bsonType": "string",
                "enum": ["admin", "doctor", "nurse", "patient"],
                "description": "must be one of: admin, doctor, nurse, patient",
            },
            "created_at": {"bsonType": "date", "description": "ISODate"},
        },
    }
}

PATIENTS_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name"],
        "properties": {
            "user_id":           {"bsonType": "objectId"},
            "name":              {"bsonType": "string"},
            "dob":               {"bsonType": ["date", "null"]},
            "assigned_room":     {"bsonType": ["string", "null"]},
            "assigned_doctors":  {"bsonType": "array", "items": {"bsonType": "objectId"}},
            "assigned_nurses":   {"bsonType": "array", "items": {"bsonType": "objectId"}},
        },
    }
}

DEVICES_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["device_id", "room_number", "status"],
        "properties": {
            "device_id":      {"bsonType": "string", "description": "e.g. rpi-room-101"},
            "room_number":    {"bsonType": "string"},
            "status":         {"bsonType": "string", "enum": ["active", "inactive", "maintenance"]},
            "registered_at":  {"bsonType": ["date", "null"]},
        },
    }
}

ECG_SUMMARIES_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["patient_id", "device_id", "start_time", "end_time", "prediction"],
        "properties": {
            "patient_id":        {"bsonType": "objectId"},
            "device_id":         {"bsonType": "string"},
            "start_time":        {"bsonType": "date"},
            "end_time":          {"bsonType": "date"},
            "heart_rate":        {"bsonType": ["double", "int", "null"]},
            "rr_mean":           {"bsonType": ["double", "int", "null"]},
            "rr_std":            {"bsonType": ["double", "int", "null"]},
            "sdnn":              {"bsonType": ["double", "int", "null"]},
            "rmssd":             {"bsonType": ["double", "int", "null"]},
            "beat_variance":     {"bsonType": ["double", "int", "null"]},
            "r_peak_count":      {"bsonType": ["int", "null"]},
            "sqi":               {"bsonType": ["double", "int", "null"]},
            "prediction":        {"bsonType": "string", "enum": ["Normal", "ABNORMAL", "Poor Signal"]},
            "probability":       {"bsonType": ["double", "int", "null"]},
            "consecutive_count": {"bsonType": ["int", "null"]},
        },
    }
}

ALERTS_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["patient_id", "device_id", "severity", "timestamp"],
        "properties": {
            "patient_id":        {"bsonType": "objectId"},
            "device_id":         {"bsonType": "string"},
            "severity":          {"bsonType": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
            "timestamp":         {"bsonType": "date"},
            "consecutive_count": {"bsonType": ["int", "null"]},
            "probability":       {"bsonType": ["double", "int", "null"]},
            "acknowledged":      {"bsonType": "bool"},
            "acknowledged_by":   {"bsonType": ["objectId", "null"]},
        },
    }
}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def create_collection_safe(db, name: str, validator: dict):
    """Create a collection with a JSON schema validator. Skip if it exists."""
    try:
        db.create_collection(name, validator={"$jsonSchema": validator["$jsonSchema"]})
        log.info(f"  ✓ Created collection: {name}")
    except CollectionInvalid:
        log.info(f"  – Collection already exists, skipping: {name}")


def create_indexes(db):
    """Create indexes for query performance."""
    log.info("Creating indexes …")

    # ecg_summaries — most queried: by patient + time desc (for history charts)
    db["ecg_summaries"].create_index(
        [("patient_id", ASCENDING), ("start_time", DESCENDING)],
        name="patient_time_desc",
    )
    db["ecg_summaries"].create_index(
        [("device_id", ASCENDING), ("start_time", DESCENDING)],
        name="device_time_desc",
    )
    log.info("  ✓ ecg_summaries: patient_time_desc, device_time_desc")

    # alerts — fetch unacknowledged alerts quickly
    db["alerts"].create_index(
        [("patient_id", ASCENDING), ("acknowledged", ASCENDING), ("timestamp", DESCENDING)],
        name="patient_unacked_alerts",
    )
    # Debounce check: find recent alert for patient in last N minutes
    db["alerts"].create_index(
        [("patient_id", ASCENDING), ("timestamp", DESCENDING)],
        name="patient_alert_recent",
    )
    log.info("  ✓ alerts: patient_unacked_alerts, patient_alert_recent")

    # devices — look up by device_id string (unique)
    db["devices"].create_index(
        [("device_id", ASCENDING)],
        unique=True,
        name="device_id_unique",
    )
    log.info("  ✓ devices: device_id_unique (unique)")

    # users — email must be unique for login
    db["users"].create_index(
        [("email", ASCENDING)],
        unique=True,
        name="email_unique",
    )
    log.info("  ✓ users: email_unique (unique)")

    log.info("All indexes created ✓")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("  ECG Project — MongoDB Atlas Initialisation")
    log.info("=" * 55)

    try:
        db = get_db()
        log.info(f"Connected to database: {db.name}")
    except Exception as e:
        log.error(f"Failed to connect to MongoDB: {e}")
        log.error("Check that MONGO_URI is correct in your .env file.")
        sys.exit(1)

    log.info("Creating collections …")
    create_collection_safe(db, "users",        USERS_SCHEMA)
    create_collection_safe(db, "patients",     PATIENTS_SCHEMA)
    create_collection_safe(db, "devices",      DEVICES_SCHEMA)
    create_collection_safe(db, "ecg_summaries", ECG_SUMMARIES_SCHEMA)
    create_collection_safe(db, "alerts",       ALERTS_SCHEMA)

    create_indexes(db)

    log.info("=" * 55)
    log.info("  MongoDB Atlas setup complete ✓")
    log.info(f"  Database : {db.name}")
    log.info(f"  Collections: {db.list_collection_names()}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
