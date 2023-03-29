from pathlib import Path

import pseudopeople

BASE_DIR = Path(pseudopeople.__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"

QWERTY_ERRORS = DATA_ROOT / "qwerty_errors.yaml"
