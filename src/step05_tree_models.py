"""Random Forest and LightGBM, tested one flood event at a time.

Same idea as step04_baselines.py: hold out one event, train on the other 10,
predict the held-out one. Tree models do not need feature scaling.

Saves outputs/tree_model_results.csv (numbered, so old results are never overwritten).

Run from the project folder:
    python src/step05_tree_models.py
"""
import csv

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from step04_baselines import OUTPUT_DIR, SAMPLES_PATH, new_output_path, results_by_event


def load_data():
    """Load the features, labels and event names made by step03_features.py."""
    data = np.load(SAMPLES_PATH)
    X = data["X"]
    y = data["y"]
    event = data["event"]
    return X, y, event


def make_random_forest():
    """A fresh Random Forest. min_samples_leaf keeps the trees from growing huge."""
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=0,
    )
    return model


def make_lightgbm():
    """A fresh LightGBM model."""
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=0,
        verbose=-1,
    )
    return model


def predict_leave_one_event_out(make_model, X, y, event):
    """For each event: train on the other events, predict this event."""
    prediction = np.zeros(len(y), dtype=int)
    all_events = sorted(set(event))

    for held_out in all_events:
        test_rows = event == held_out
        train_rows = event != held_out

        X_train = X[train_rows]
        y_train = y[train_rows]
        X_test = X[test_rows]

        model = make_model()
        model.fit(X_train, y_train)
        prediction[test_rows] = model.predict(X_test)
        print("  finished", held_out)

    return prediction


def print_table(rows):
    """Print the results table: one line per model and event."""
    print()
    print("model            event          IoU      F1  precision    recall")
    for row in rows:
        model_name, event_name, iou, f1, precision, recall = row
        print(f"{model_name:<16} {event_name:<10} {iou:7.3f} {f1:7.3f} {precision:10.3f} {recall:9.3f}")


def save_table(rows, path):
    """Save the results table as a CSV file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    header = ["model", "event", "iou", "f1", "precision", "recall"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    X, y, event = load_data()
    print("Rows:", X.shape[0], " Features:", X.shape[1])

    all_rows = []

    print("\nRandom Forest")
    forest_prediction = predict_leave_one_event_out(make_random_forest, X, y, event)
    all_rows += results_by_event(y, forest_prediction, event, "random_forest")

    print("\nLightGBM")
    lightgbm_prediction = predict_leave_one_event_out(make_lightgbm, X, y, event)
    all_rows += results_by_event(y, lightgbm_prediction, event, "lightgbm")

    print_table(all_rows)

    out_path = new_output_path("tree_model_results")
    save_table(all_rows, out_path)
    print("\nSaved table to", out_path)


if __name__ == "__main__":
    main()
