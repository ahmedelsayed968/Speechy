import os
from pathlib import Path

# Get the current working directory (project root)
PROJECT_ROOT = Path(os.getcwd()).resolve()
CONFIG_DIR = PROJECT_ROOT/".conf"
GENDER_MODEL_PATH = PROJECT_ROOT/"model"/"gender-detector.joblib"
GENDER_MODEL_SCALER_PATH = PROJECT_ROOT / "model" / "gender-detector-scaler.joblib"