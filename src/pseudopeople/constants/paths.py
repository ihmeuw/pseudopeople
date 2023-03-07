from pathlib import Path

import pseudopeople


REPO_DIR = Path(pseudopeople.__file__).resolve().parent.parent.parent
BASE_DIR = Path(pseudopeople.__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"

NICKNAMES_DATA = DATA_ROOT / "nicknames.csv"
