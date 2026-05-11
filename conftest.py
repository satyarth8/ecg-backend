# pytest configuration
import sys
from pathlib import Path

# After Phase 1 restructure, source code lives in backend/src/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
