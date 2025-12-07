import numpy as np
from src.hybrid.hybrid_detector import HybridDetector


def main():
    # Load train_normal to get feature count
    X_train = np.load("data/processed/train_normal.npy")
    n_features = X_train.shape[1]

    # Random abnormal sample
    sample = np.random.randint(10000, 60000, size=(1, n_features))

    detector = HybridDetector()
    pred = detector.predict(sample)[0]

    print("\n===== TEST RANDOM SAMPLE =====")
    print("Prediction:", pred)
    print("=>", "ATTACK" if pred == 1 else "NORMAL")


if __name__ == "__main__":
    main()
