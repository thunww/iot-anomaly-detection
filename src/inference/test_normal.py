import numpy as np
from src.hybrid.hybrid_detector import HybridDetector

DATA_NORMAL = "data/processed/train_normal.npy"


def main():
    # Load 1 sample normal
    X = np.load(DATA_NORMAL)
    sample = X[0].reshape(1, -1)

    # Load model
    detector = HybridDetector()

    pred = detector.predict(sample)[0]

    print("\n===== TEST NORMAL SAMPLE =====")
    print("Prediction:", pred)
    print("=>", "ATTACK" if pred == 1 else "NORMAL")


if __name__ == "__main__":
    main()
