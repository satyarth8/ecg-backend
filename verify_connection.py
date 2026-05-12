"""
verify_connection.py
─────────────────────────────────────────────────────────────────────────────
Quick MongoDB Atlas connection test.

Run this BEFORE running init_db.py to confirm your MONGO_URI is correct
and the Atlas cluster is reachable.

Usage:
    python verify_connection.py

Expected output (success):
    ✓ Connected to MongoDB Atlas
    ✓ Database  : ecg_db
    ✓ Ping      : OK
    ✓ Collections: ['users', 'patients', ...]
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 55)
    print("  ECG Project — MongoDB Atlas Connection Test")
    print("=" * 55)

    uri = os.getenv("MONGO_URI")
    if not uri:
        print("✗ MONGO_URI is not set.")
        print("  → Copy .env.example to .env and fill in your Atlas URI.")
        sys.exit(1)

    # Mask password for safe display
    masked = uri
    if "@" in uri and ":" in uri:
        try:
            proto, rest = uri.split("://", 1)
            creds, host = rest.split("@", 1)
            user, _ = creds.split(":", 1)
            masked = f"{proto}://{user}:****@{host}"
        except Exception:
            masked = "mongodb+srv://****"

    print(f"  URI    : {masked}")
    print()

    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, OperationFailure

        client = MongoClient(uri, serverSelectionTimeoutMS=15_000)

        # Ping
        client.admin.command("ping")
        print("✓ Connected to MongoDB Atlas")

        db = client["ecg_db"]
        print(f"✓ Database  : {db.name}")
        print("✓ Ping      : OK")

        collections = db.list_collection_names()
        if collections:
            print(f"✓ Collections: {collections}")
        else:
            print("✓ Collections: (none yet — run init_db.py to create them)")

        # Write test — insert and immediately delete a doc
        test_col = db["_connection_test"]
        result = test_col.insert_one({"test": True})
        test_col.delete_one({"_id": result.inserted_id})
        print("✓ Write test : OK (inserted + deleted test doc)")

        print()
        print("=" * 55)
        print("  All checks passed — MongoDB Atlas is ready ✓")
        print("  Next step: run  python init_db.py")
        print("=" * 55)

    except ConnectionFailure as e:
        print(f"✗ Connection failed: {e}")
        print()
        print("  Possible causes:")
        print("  1. Wrong MONGO_URI (typo in username/password/cluster name)")
        print("  2. Your IP is not whitelisted in Atlas → Network Access → 0.0.0.0/0")
        print("  3. Atlas cluster is paused (free M0 auto-pauses after 60 days of inactivity)")
        sys.exit(1)

    except OperationFailure as e:
        print(f"✗ Auth error: {e}")
        print()
        print("  → Check username and password in your MONGO_URI.")
        print("  → Make sure the user has readWrite role on the ecg_db database.")
        sys.exit(1)

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
