import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.hybrid.hybrid_detector import HybridDetector
from src.utils.paths import TEST_PATH, TEST_LABEL_PATH

def evaluate_hybrid():

    # Load test data
    X = np.load(TEST_PATH)
    y_true = np.load(TEST_LABEL_PATH)

    # Không dùng input_dim nữa
    model = HybridDetector(latent_dim=128)

    # Dự đoán
    y_pred = model.predict(X)

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_true, y_pred))

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    evaluate_hybrid()
