"""Noisy-label experiment: does weakly labelled training data help?

Three ways to train, but ALWAYS tested on hand labels:
    hand_only : train on hand-labelled rows from the other 10 events
    weak_only : train on weakly labelled rows from the other 10 events
    both      : train on both of the above together

For each event: train on the other 10 events, test on that event's hand rows.
The held-out event's weak rows are never used for training.

The number of weak chips is varied (WEAK_CHIP_COUNTS) to see if more weak
data keeps helping. Each count uses the same number of chips per event.

Needs both files made by step03_features.py:
    python src/step03_features.py          -> data/processed/samples.npz
    python src/step03_features.py weak     -> data/processed/samples_weak.npz

Run from the project folder (model is lightgbm unless you name another):
    python src/step08_noisy_labels.py
    python src/step08_noisy_labels.py logistic_regression
    python src/step08_noisy_labels.py random_forest
    python src/step08_noisy_labels.py mlp
"""
import random
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from step04_baselines import PROCESSED_DIR, new_output_path, results_by_event
from step06_mlp import make_mlp
from step05_tree_models import make_lightgbm, make_random_forest, save_table

# How many weak chips to train on (in total, spread equally over the events)
WEAK_CHIP_COUNTS = [2000]

# Which setups to run with the weak chips: "weak_only", "both" or both of them
WEAK_SETUPS = ["weak_only"]

# Also run the hand-only reference (no weak data)?
RUN_HAND_ONLY = False

SEED = 0


def make_logistic_regression():
    """A fresh scaled logistic regression."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    return model


MODELS = {
    "logistic_regression": make_logistic_regression,
    "random_forest": make_random_forest,
    "lightgbm": make_lightgbm,
    "mlp": make_mlp,
}


def choose_model_name():
    """lightgbm unless the command line names another model."""
    model_name = "lightgbm"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    if model_name not in MODELS:
        raise SystemExit(f"Unknown model: {model_name}. Use one of {list(MODELS)}")
    return model_name


def load_samples(file_name):
    """Load X, y, event and chip from a samples file made by step03_features.py."""
    data = np.load(PROCESSED_DIR / file_name)
    return data["X"], data["y"], data["event"], data["chip"]


def choose_weak_chips(weak_event, weak_chip, how_many):
    """Pick how_many weak chips, the same number from every event.

    Chips are handed out one per event in turn (random inside each event),
    like in step07_download_weak.py. A bigger how_many always contains the smaller
    ones, so the training sets grow step by step.
    """
    # Find the chips of each event (a chip has many rows, so look at each chip once)
    by_event = {}
    for chip_id in sorted(set(weak_chip)):
        event_name = chip_id.split("_")[0]
        if event_name not in by_event:
            by_event[event_name] = []
        by_event[event_name].append(chip_id)

    for event_name in sorted(by_event):
        random.Random(SEED).shuffle(by_event[event_name])

    chosen = []
    position = 0
    while len(chosen) < how_many:
        added_any = False
        for event_name in sorted(by_event):
            chips_of_event = by_event[event_name]
            if position < len(chips_of_event) and len(chosen) < how_many:
                chosen.append(chips_of_event[position])
                added_any = True
        if not added_any:
            break
        position += 1

    return chosen


def predict_hand_rows(make_model, setup, hand, weak, weak_rows_allowed):
    """For each event: train on the other events, predict that event's hand rows.

    setup is "hand_only", "weak_only" or "both".
    weak_rows_allowed marks which weak rows may be used (the chosen chips).
    """
    X_hand, y_hand, event_hand = hand
    X_weak, y_weak, event_weak = weak

    prediction = np.zeros(len(y_hand), dtype=int)

    for held_out in sorted(set(event_hand)):
        # Hand rows: test = this event, train = the other events
        hand_test = event_hand == held_out
        hand_train = event_hand != held_out

        # Weak rows: never from the held-out event, and only the chosen chips
        weak_train = (event_weak != held_out) & weak_rows_allowed

        if setup == "hand_only":
            X_train = X_hand[hand_train]
            y_train = y_hand[hand_train]
        elif setup == "weak_only":
            X_train = X_weak[weak_train]
            y_train = y_weak[weak_train]
        else:
            X_train = np.concatenate([X_hand[hand_train], X_weak[weak_train]])
            y_train = np.concatenate([y_hand[hand_train], y_weak[weak_train]])

        model = make_model()
        model.fit(X_train, y_train)
        prediction[hand_test] = model.predict(X_hand[hand_test])
        print(f"    finished {held_out} (trained on {len(y_train)} rows)")

    return prediction


def main():
    model_name = choose_model_name()
    make_model = MODELS[model_name]
    print("Model:", model_name)

    X_hand, y_hand, event_hand, _ = load_samples("samples.npz")
    X_weak, y_weak, event_weak, chip_weak = load_samples("samples_weak.npz")
    print("Hand rows:", len(y_hand), " Weak rows:", len(y_weak))

    hand = (X_hand, y_hand, event_hand)
    weak = (X_weak, y_weak, event_weak)

    all_rows = []

    # 1. Hand only (reference). No weak data, so it runs once.
    if RUN_HAND_ONLY:
        print("\nhand_only")
        no_weak_rows = np.zeros(len(y_weak), dtype=bool)
        prediction = predict_hand_rows(make_model, "hand_only", hand, weak, no_weak_rows)
        all_rows += results_by_event(y_hand, prediction, event_hand, f"{model_name}_hand_only")

    # 2. The weak setups, for each number of weak chips
    for how_many in WEAK_CHIP_COUNTS:
        chosen_chips = choose_weak_chips(event_weak, chip_weak, how_many)
        weak_rows_allowed = np.isin(chip_weak, chosen_chips)

        for setup in WEAK_SETUPS:
            print(f"\n{setup}, {len(chosen_chips)} weak chips")
            prediction = predict_hand_rows(make_model, setup, hand, weak, weak_rows_allowed)
            name = f"{model_name}_{setup}_{len(chosen_chips)}"
            all_rows += results_by_event(y_hand, prediction, event_hand, name)

    # Print only the ALL rows (one line per setup); the CSV has every event
    print()
    print("setup                               IoU      F1  precision    recall")
    for row in all_rows:
        row_name, event_name, iou, f1, precision, recall = row
        if event_name == "ALL":
            print(f"{row_name:<30} {iou:7.3f} {f1:7.3f} {precision:10.3f} {recall:9.3f}")

    out_path = new_output_path(f"noisy_label_results_{model_name}")
    save_table(all_rows, out_path)
    print("\nSaved table to", out_path)


if __name__ == "__main__":
    main()
