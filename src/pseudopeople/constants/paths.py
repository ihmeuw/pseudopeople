from pathlib import Path

import pseudopeople

BASE_DIR = Path(pseudopeople.__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"

INCORRECT_SELECT_NOISE_OPTIONS_DATA = DATA_ROOT / "incorrect_select_options.csv"
QWERTY_ERRORS = DATA_ROOT / "qwerty_errors.yaml"

SAMPLE_DATA_ROOT = DATA_ROOT / "sample_datasets"
