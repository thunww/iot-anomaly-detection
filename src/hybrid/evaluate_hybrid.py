import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.hybrid.hybrid_detector import HybridDetector
from src.utils.paths import TEST_PATH, TEST_LABEL_PATH

def evaluate_hybrid():

    X = np.load(TEST_PATH)
    y_true = np.load(TEST_LABEL_PATH)

    input_dim = X.shape[1]
    model = HybridDetector(input_dim=input_dim, latent_dim=128)

    y_pred = model.predict(X)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate_hybrid()
