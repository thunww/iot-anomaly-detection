import numpy as np
from src.hybrid.hybrid_detector import HybridDetector

DATA_ATTACK = "data/processed/train_attack.npy"


def main():
    # Load 1 sample attack
    X = np.load(DATA_ATTACK)
    sample = X[0].reshape(1, -1)

    # Load model
    detector = HybridDetector()

    pred = detector.predict(sample)[0]

    print("\n===== TEST ATTACK SAMPLE =====")
    print("Prediction:", pred)
    print("=>", "ATTACK" if pred == 1 else "NORMAL")


if __name__ == "__main__":
    main()
