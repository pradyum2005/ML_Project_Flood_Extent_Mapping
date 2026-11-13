"""Full-chip scoring: score every valid pixel of each held-out chip.

Steps 4-6 scored only the balanced sample (about 45% water). Real chips are
only about 9% water, so those scores look better than real life. Here we:
    1. hold out one event and train each model on the balanced sample of the
       other 10 events (exactly like steps 4-6),
    2. predict EVERY valid pixel of every chip of the held-out event,
    3. count true positives, false positives and false negatives,
    4. turn the counts into IoU, F1, precision and recall.

Chips are handled one at a time, so memory use stays small.

Also saves the predicted map of every chip (one .npz file per chip) so step 10
(flood maps) and the error analysis can reuse them without training again.
In each map: 1 = water, 0 = land, -1 = not a valid pixel.

Saves outputs/full_chip_results.csv (numbered, so old results are never
overwritten) and the maps to data/processed/full_chip_predictions/ (also
numbered: full_chip_predictions2, 3, ... if the folder already exists).

Run from the project folder:
    python src/step09_full_chip_scoring.py
"""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from step03_features import FEATURE_NAMES, compute_features, list_chips, load_chip, valid_mask
from step04_baselines import MNDWI_THRESHOLD, PROCESSED_DIR, new_output_path
from step05_tree_models import load_data, make_lightgbm, make_random_forest, save_table
from step06_mlp import make_mlp

MNDWI_COLUMN = FEATURE_NAMES.index("mndwi")


def make_logistic_regression():
    """A fresh scaled logistic regression (same settings as step 4)."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    return model


# The models that need training. The MNDWI threshold needs no training.
MODELS = {
    "logistic_regression": make_logistic_regression,
    "random_forest": make_random_forest,
    "lightgbm": make_lightgbm,
    "mlp": make_mlp,
}

MODEL_NAMES = ["mndwi_threshold"] + list(MODELS)


def new_folder_path(base_name):
    """A folder path that does not exist yet, so old maps are never overwritten."""
    path = PROCESSED_DIR / base_name
    number = 2
    while path.exists():
        path = PROCESSED_DIR / f"{base_name}{number}"
        number += 1
    return path


def train_models(X_train, y_train):
    """Train every model in MODELS on the training rows. Returns {name: model}."""
    trained = {}
    for name in MODELS:
        start = time.time()
        model = MODELS[name]()
        model.fit(X_train, y_train)
        trained[name] = model
        seconds = time.time() - start
        print(f"    trained {name} in {seconds:.0f} s")
    return trained


def predict_pixels(trained, X_pixels):
    """Predict water (1) or land (0) for each pixel row, with every model."""
    predictions = {}

    mndwi = X_pixels[:, MNDWI_COLUMN]
    predictions["mndwi_threshold"] = (mndwi > MNDWI_THRESHOLD).astype(np.int8)

    for name in trained:
        model = trained[name]
        predictions[name] = model.predict(X_pixels).astype(np.int8)

    return predictions


def count_errors(y_true, y_pred):
    """Count true positives, false positives and false negatives for water."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, fn


def scores_from_counts(tp, fp, fn):
    """IoU, F1, precision and recall from the counts (same formulas as step 4)."""
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return round(iou, 3), round(f1, 3), round(precision, 3), round(recall, 3)


def save_chip_maps(folder, chip_id, label, valid, predictions):
    """Save the label, the valid mask and each model's full 512x512 map."""
    maps = {}
    for name in predictions:
        full_map = np.full(valid.shape, -1, dtype=np.int8)
        full_map[valid] = predictions[name]
        maps[name] = full_map

    np.savez_compressed(folder / f"{chip_id}.npz", label=label, valid=valid, **maps)


def main():
    X, y, event = load_data()
    events = sorted(set(event))
    chip_ids = list_chips("hand")
    print("Training rows:", len(y), " Chips:", len(chip_ids), " Events:", len(events))

    map_folder = new_folder_path("full_chip_predictions")
    map_folder.mkdir(parents=True)
    print("Saving maps to", map_folder)

    # counts[model][event] = [tp, fp, fn]
    counts = {}
    for name in MODEL_NAMES:
        counts[name] = {}
        for event_name in events:
            counts[name][event_name] = [0, 0, 0]

    # Number of valid pixels and water pixels per event (to show the real water share)
    valid_pixels = {}
    water_pixels = {}

    run_start = time.time()

    for held_out in events:
        event_start = time.time()
        print(f"\n{held_out}")

        # 1. Train on the balanced sample of the other 10 events
        train_rows = event != held_out
        trained = train_models(X[train_rows], y[train_rows])

        # 2. Predict every valid pixel of every chip of the held-out event
        event_chips = []
        for chip_id in chip_ids:
            if chip_id.split("_")[0] == held_out:
                event_chips.append(chip_id)

        valid_pixels[held_out] = 0
        water_pixels[held_out] = 0

        for chip_id in event_chips:
            s2, label, _ = load_chip(chip_id, "hand")
            valid = valid_mask(s2, label)
            if valid.sum() == 0:
                print(f"    {chip_id}: no valid pixels, skipped")
                continue

            features = compute_features(s2)
            X_pixels = features[valid]
            y_pixels = label[valid]

            predictions = predict_pixels(trained, X_pixels)

            # 3. Add this chip's counts to the event's counts
            for name in MODEL_NAMES:
                tp, fp, fn = count_errors(y_pixels, predictions[name])
                counts[name][held_out][0] += tp
                counts[name][held_out][1] += fp
                counts[name][held_out][2] += fn

            valid_pixels[held_out] += len(y_pixels)
            water_pixels[held_out] += int((y_pixels == 1).sum())

            save_chip_maps(map_folder, chip_id, label, valid, predictions)

        minutes = (time.time() - event_start) / 60
        print(f"    scored {len(event_chips)} chips in {minutes:.1f} min total for this event")

    # 4. Turn the counts into scores: one row per model and event, plus ALL
    all_rows = []
    for name in MODEL_NAMES:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for event_name in events:
            tp, fp, fn = counts[name][event_name]
            total_tp += tp
            total_fp += fp
            total_fn += fn
            iou, f1, precision, recall = scores_from_counts(tp, fp, fn)
            all_rows.append([name, event_name, iou, f1, precision, recall])

        iou, f1, precision, recall = scores_from_counts(total_tp, total_fp, total_fn)
        all_rows.append([name, "ALL", iou, f1, precision, recall])

    # Real water share of each event (all valid pixels, not the balanced sample)
    print("\nevent        valid pixels   water share")
    for event_name in events:
        share = 100 * water_pixels[event_name] / max(valid_pixels[event_name], 1)
        print(f"{event_name:<12} {valid_pixels[event_name]:>12}   {share:9.1f}%")

    # Print only the ALL rows; the CSV has every event
    print("\nmodel                     IoU      F1  precision    recall")
    for row in all_rows:
        name, event_name, iou, f1, precision, recall = row
        if event_name == "ALL":
            print(f"{name:<22} {iou:7.3f} {f1:7.3f} {precision:10.3f} {recall:9.3f}")

    out_path = new_output_path("full_chip_results")
    save_table(all_rows, out_path)
    print("\nSaved table to", out_path)

    total_minutes = (time.time() - run_start) / 60
    print(f"Total time: {total_minutes:.1f} min")


if __name__ == "__main__":
    main()
