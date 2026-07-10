"""Central configuration for the MachineSense ML pipeline (Phase 0).

All audio/feature parameters follow the DCASE 2020 Task 2 baseline so results are
comparable to published numbers on the MIMII dataset.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # ml/
ARTIFACTS = ROOT / "artifacts"
FEATURE_CACHE = ARTIFACTS / "feature_cache"

# --- Dataset -----------------------------------------------------------------
# Unzip one MIMII machine type here, e.g. for "fan":
#   ml/data/fan/id_00/normal/*.wav
#   ml/data/fan/id_00/abnormal/*.wav
MACHINE = "fan"                                  # fan | pump | valve | slider
DATA_DIR = ROOT / "data" / MACHINE
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
TEST_NORMAL_FRACTION = 0.2                       # held-out normal clips used for AUC

# --- Audio / log-mel features (DCASE 2020 Task 2 baseline) -------------------
SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 128
POWER = 2.0
FRAMES = 5                                       # context frames concatenated
FEATURE_DIM = N_MELS * FRAMES                    # = 640

# --- Autoencoder -------------------------------------------------------------
HIDDEN = 128
BOTTLENECK = 8
DEPTH = 4                                         # dense layers per half
EPOCHS = 100
BATCH = 512
LR = 1e-3
VAL_SPLIT = 0.1
SEED = 42

# --- Export ------------------------------------------------------------------
MODEL_KERAS = ARTIFACTS / "model.keras"
REP_VECTORS = ARTIFACTS / "rep_vectors.npy"      # representative set for int8 calibration
TFLITE_INT8 = ARTIFACTS / "model_int8.tflite"
C_HEADER = ARTIFACTS / "model_data.cc"           # copy into firmware/main/ for Phase 1
C_VAR_NAME = "g_model_data"
