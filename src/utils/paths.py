import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_RAW = os.path.join(ROOT, "data/raw/Train_Test_IoT_Modbus.csv")
DATA_PROCESSED = os.path.join(ROOT, "data/processed")

TRAIN_NORMAL_PATH = os.path.join(DATA_PROCESSED, "train_normal.npy")
TRAIN_ATTACK_PATH = os.path.join(DATA_PROCESSED, "train_attack.npy")

TEST_PATH = os.path.join(DATA_PROCESSED, "test.npy")
TEST_LABEL_PATH = os.path.join(DATA_PROCESSED, "test_labels.npy")

Z_NORMAL = os.path.join(DATA_PROCESSED, "z_normal.npy")
Z_ATTACK = os.path.join(DATA_PROCESSED, "z_attack.npy")
Z_TEST = os.path.join(DATA_PROCESSED, "z_test.npy")

MODEL_DIR = os.path.join(ROOT, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
AE_MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder.pt")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.npy")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost.json")
