import numpy as np
from inference.predict import Predictor

TEST_PATH = "data/processed/test.npy"
LABEL_PATH = "data/processed/test_labels.npy"

def run_inference_pipeline():
    print("[INFO] Running inference pipeline...")

    X = np.load(TEST_PATH)
    y_true = np.load(LABEL_PATH)

    predictor = Predictor()
    result = predictor.predict(X)

    y_pred = result["hybrid"]

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    print("\n==== RESULTS ====")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")

    acc = (tp + tn) / len(y_true)
    print(f"\nAccuracy = {acc*100:.2f}%")

    return y_pred

if __name__ == "__main__":
    run_inference_pipeline()
