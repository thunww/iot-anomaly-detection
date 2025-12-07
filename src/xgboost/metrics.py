from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def compute_metrics(y, pred, prob=None):
    metrics = {
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred)
    }
    if prob is not None:
        metrics["auc"] = roc_auc_score(y, prob)
    return metrics
