"""Baselines scored with leave-one-event-out cross-validation.

    1. MNDWI threshold: no learning, just "water if MNDWI > threshold".
    2. Logistic regression on all 22 features.

Reads data/processed/samples.npz (made by features.py) and saves
outputs/baseline_results.csv.

Usage (from the project folder):
    python src/baselines.py
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SAMPLES_PATH = Path("data/processed/samples.npz")
OUTPUT_DIR = Path("outputs")

MNDWI_THRESHOLD = 0.0


def metrics(y_true, y_pred):
    """IoU, F1, precision and recall for the water class (label 1)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    # max(..., 1) avoids dividing by zero when a denominator is 0.
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


def mndwi_baseline(X, feature_names):
    """Predict water (1) wherever MNDWI is above the threshold, else land (0)."""
    col = list(feature_names).index("mndwi")
    return (X[:, col] > MNDWI_THRESHOLD).astype(np.int8)


def logistic_regression_loeo(X, y, event):
    """Train on 10 events, predict the 11th, for every event. Returns predictions for all rows."""
    y_pred = np.zeros_like(y)
    for held_out in sorted(set(event)):
        test = event == held_out
        train = ~test

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        model.fit(X[train], y[train])
        y_pred[test] = model.predict(X[test])
        print(f"  finished {held_out}")
    return y_pred


def results_by_event(y, y_pred, event, model_name):
    """One row of metrics per event, plus an ALL row over every event."""
    rows = []
    for name in sorted(set(event)) + ["ALL"]:
        rows_in_event = np.ones(len(y), dtype=bool) if name == "ALL" else event == name
        m = metrics(y[rows_in_event], y_pred[rows_in_event])
        rows.append([model_name, name, round(m["iou"], 3), round(m["f1"], 3),
                     round(m["precision"], 3), round(m["recall"], 3)])
    return rows


def main():
    data = np.load(SAMPLES_PATH)
    X, y, event = data["X"], data["y"], data["event"]
    feature_names = data["feature_names"]
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features, {len(set(event))} events")

    all_rows = []

    print("\nMNDWI threshold")
    all_rows += results_by_event(y, mndwi_baseline(X, feature_names), event, "mndwi_threshold")

    print("\nLogistic regression (leave-one-event-out)")
    y_pred = logistic_regression_loeo(X, y, event)
    all_rows += results_by_event(y, y_pred, event, "logistic_regression")

    header = ["model", "event", "iou", "f1", "precision", "recall"]
    print()
    print(f"{header[0]:<20}{header[1]:<12}" + "".join(f"{h:>11}" for h in header[2:]))
    for row in all_rows:
        print(f"{row[0]:<20}{row[1]:<12}" + "".join(f"{v:>11}" for v in row[2:]))

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "baseline_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    print(f"\nSaved table to {out_path}")


if __name__ == "__main__":
    main()
