from pathlib import Path

# Base project directory 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MMS region files (input)
MMS_REGIONFILES_DIR = PROJECT_ROOT / "mms-regionfiles"

# Compiled solar wind time intervals (output)
COMPILED_TINTS_DIR = PROJECT_ROOT / "sw_tints_2026"







if not MMS_REGIONFILES_DIR.exists():
    raise RuntimeError(f"Region files directory not found: {MMS_REGIONFILES_DIR}")
