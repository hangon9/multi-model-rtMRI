from sklearn.metrics import classification_report, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, label_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        average=None,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    per_class = {}

    for idx, name in enumerate(label_names):
        per_class[name] = {
            "precision": precision[idx],
            "recall": recall[idx],
            "f1": f1[idx],
            "support": support[idx],
        }

    return {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }