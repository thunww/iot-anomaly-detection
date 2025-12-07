from src.hybrid.hybrid_detector import HybridDetector
import numpy as np


def main():
    # LẤY ĐÚNG SỐ FEATURES TỪ FILE PROCESSED
    import numpy as np
    X_train = np.load("data/processed/train_normal.npy")
    n_features = X_train.shape[1]

    # tạo sample giả đúng số cột
    sample = np.random.randint(10000, 60000, size=(1, n_features))

    detector = HybridDetector()
    pred = detector.predict(sample)[0]

    print("Prediction:", pred)
    print("=>", "ATTACK" if pred == 1 else "NORMAL")

if __name__ == "__main__":
    main()
