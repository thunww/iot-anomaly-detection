import numpy as np
from hybrid.hybrid_detector import HybridDetector

class Predictor:
    def __init__(self):
        self.detector = HybridDetector()

    def predict(self, X):
        """
        X: numpy array (1 sample or many)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        hybrid_pred, mse, latent = self.detector.predict(X)
        return {
            "hybrid": hybrid_pred,
            "recon_error": mse,
            "latent": latent
        }
