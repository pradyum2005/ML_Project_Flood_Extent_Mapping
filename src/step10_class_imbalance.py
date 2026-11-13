"""Class-imbalance comparison, scored on full chips (like step 9).

Step 9 showed that our models call too much water on real chips: they were
trained on a sample that is about 45% water, but real chips are about 9% water.
Here we compare ways of handling the imbalance:

    Sampling (what the model trains on):
        balanced : the usual sample, up to 1000 water + 1000 land pixels per chip
                   (this is "undersampling" of land)
        natural  : 2000 random valid pixels per chip, so the water share is real

    Class weight:
        weighted   : class_weight="balanced" (water and land count equally)
        unweighted : no class weight

    Decision threshold:
        Call a pixel water if the model's water probability is above the
        threshold. The usual threshold is 0.5. We try several.

That is 2 x 2 = 4 trained models per held-out event, each scored at every
threshold, on every valid pixel of the held-out event's chips.

Note: the threshold rows are a "what if" check. Choosing the best threshold by
looking at these test scores would be cheating a little, so in the report use
0.5 as the main result and show the other thresholds as a sensitivity check.

Saves outputs/class_imbalance_results_<model>.csv (numbered, never overwritten).
The natural sample is saved once to data/processed/samples_natural.npz and
reused on later runs.

Run from the project folder:
    python src/step10_class_imbalance.py                        # LightGBM (~15 min)
    python src/step10_class_imbalance.py logistic_regression    # Logistic regression (~15 min)
    python src/step10_class_imbalance.py random_forest          # Random Forest (~1.5 hours)
"""
import csv
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from step03_features import compute_features, list_chips, load_chip, valid_mask
from step04_baselines import OUTPUT_DIR, PROCESSED_DIR, new_output_path
from step05_tree_models import load_data, make_lightgbm, make_random_forest

NATURAL_SAMPLES_PATH = PROCESSED_DIR / "samples_natural.npz"
PIXELS_PER_CHIP = 2000  # same number of rows per chip as the balanced sample
SEED = 0

SAMPLINGS = ["balanced", "natural"]
CLASS_WEIGHTS = ["weighted", "unweighted"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

MODEL_NAMES = ["lightgbm", "logistic_regression", "random_forest"]


def choose_model_name():
    """lightgbm unless the command line names another model."""
    model_name = "lightgbm"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    if model_name not in MODEL_NAMES:
        raise SystemExit(f"Unknown model: {model_name}. Use one of {MODEL_NAMES}")
    return model_name


def make_model(model_name, class_weight):
    """A fresh model with the same settings as steps 4-5, with or without class weight."""
    if model_name == "logistic_regression":
        weight = "balanced"
        if class_weight == "unweighted":
            weight = None
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight=weight, max_iter=1000),
        )
        return model

    if model_name == "lightgbm":
        model = make_lightgbm()
    else:
        model = make_random_forest()

    if class_weight == "unweighted":
        model.set_params(class_weight=None)

    return model


def build_natural_sample(chip_ids):
    """Pick up to PIXELS_PER_CHIP random valid pixels from every chip.

    Unlike the balanced sample, water and land are not picked separately, so
    the sample keeps the real water share of each chip.
    Saved to NATURAL_SAMPLES_PATH so it is only built once.
    """
    if NATURAL_SAMPLES_PATH.exists():
        print("Loading natural sample from", NATURAL_SAMPLES_PATH)
        data = np.load(NATURAL_SAMPLES_PATH)
        return data["X"], data["y"], data["event"]

    print("Building natural sample (only needed once)...")
    rng = np.random.default_rng(SEED)
    X_parts = []
    y_parts = []
    event_parts = []

    for chip_id in chip_ids:
        s2, label, _ = load_chip(chip_id, "hand")
        valid = valid_mask(s2, label)
        ys, xs = np.nonzero(valid)
        if len(ys) == 0:
            continue

        how_many = min(PIXELS_PER_CHIP, len(ys))
        pick = rng.choice(len(ys), size=how_many, replace=False)

        features = compute_features(s2)
        X_parts.append(features[ys[pick], xs[pick]])
        y_parts.append(label[ys[pick], xs[pick]].astype(np.int8))
        event_name = chip_id.split("_")[0]
        event_parts.append(np.full(how_many, event_name))

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    event = np.concatenate(event_parts)

    np.savez_compressed(NATURAL_SAMPLES_PATH, X=X, y=y, event=event)
    print("Saved natural sample to", NATURAL_SAMPLES_PATH)
    return X, y, event


def setup_name(sampling, class_weight):
    """For example 'balanced_weighted'."""
    return f"{sampling}_{class_weight}"


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


def save_results(rows, path):
    """Save the results table as a CSV file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    header = ["model", "sampling", "class_weight", "threshold", "event",
              "iou", "f1", "precision", "recall"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    model_name = choose_model_name()
    print("Model:", model_name)

    chip_ids = list_chips("hand")

    # The two training samples
    samples = {}
    X_balanced, y_balanced, event_balanced = load_data()
    samples["balanced"] = (X_balanced, y_balanced, event_balanced)
    X_natural, y_natural, event_natural = build_natural_sample(chip_ids)
    samples["natural"] = (X_natural, y_natural, event_natural)

    for sampling in SAMPLINGS:
        X_s, y_s, event_s = samples[sampling]
        print(f"{sampling} sample: {len(y_s)} rows, {100 * y_s.mean():.1f}% water")

    events = sorted(set(event_balanced))

    # counts[(setup, threshold, event)] = [tp, fp, fn]
    counts = {}
    for sampling in SAMPLINGS:
        for class_weight in CLASS_WEIGHTS:
            name = setup_name(sampling, class_weight)
            for threshold in THRESHOLDS:
                for event_name in events:
                    counts[(name, threshold, event_name)] = [0, 0, 0]

    run_start = time.time()

    for held_out in events:
        event_start = time.time()
        print(f"\n{held_out}")

        # 1. Train the 4 models on the other 10 events
        trained = {}
        for sampling in SAMPLINGS:
            X_s, y_s, event_s = samples[sampling]
            train_rows = event_s != held_out
            for class_weight in CLASS_WEIGHTS:
                name = setup_name(sampling, class_weight)
                model = make_model(model_name, class_weight)
                model.fit(X_s[train_rows], y_s[train_rows])
                trained[name] = model
        print(f"    trained 4 models in {time.time() - event_start:.0f} s")

        # 2. Score every valid pixel of the held-out event's chips
        for chip_id in chip_ids:
            if chip_id.split("_")[0] != held_out:
                continue

            s2, label, _ = load_chip(chip_id, "hand")
            valid = valid_mask(s2, label)
            if valid.sum() == 0:
                continue

            features = compute_features(s2)
            X_pixels = features[valid]
            y_pixels = label[valid]

            for name in trained:
                # Column 1 of predict_proba is the probability of water
                water_probability = trained[name].predict_proba(X_pixels)[:, 1]

                for threshold in THRESHOLDS:
                    prediction = (water_probability > threshold).astype(np.int8)
                    tp, fp, fn = count_errors(y_pixels, prediction)
                    key = (name, threshold, held_out)
                    counts[key][0] += tp
                    counts[key][1] += fp
                    counts[key][2] += fn

        minutes = (time.time() - event_start) / 60
        print(f"    finished {held_out} in {minutes:.1f} min")

    # 3. Turn the counts into scores: one row per setup, threshold and event, plus ALL
    all_rows = []
    for sampling in SAMPLINGS:
        for class_weight in CLASS_WEIGHTS:
            name = setup_name(sampling, class_weight)
            for threshold in THRESHOLDS:
                total_tp = 0
                total_fp = 0
                total_fn = 0
                for event_name in events:
                    tp, fp, fn = counts[(name, threshold, event_name)]
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    iou, f1, precision, recall = scores_from_counts(tp, fp, fn)
                    all_rows.append([model_name, sampling, class_weight, threshold,
                                     event_name, iou, f1, precision, recall])

                iou, f1, precision, recall = scores_from_counts(total_tp, total_fp, total_fn)
                all_rows.append([model_name, sampling, class_weight, threshold,
                                 "ALL", iou, f1, precision, recall])

    # Print only the ALL rows; the CSV has every event
    print("\nsampling  class_weight  threshold     IoU      F1  precision    recall")
    for row in all_rows:
        _, sampling, class_weight, threshold, event_name, iou, f1, precision, recall = row
        if event_name == "ALL":
            print(f"{sampling:<9} {class_weight:<12} {threshold:9.1f} "
                  f"{iou:7.3f} {f1:7.3f} {precision:10.3f} {recall:9.3f}")

    out_path = new_output_path(f"class_imbalance_results_{model_name}")
    save_results(all_rows, out_path)
    print("\nSaved table to", out_path)

    total_minutes = (time.time() - run_start) / 60
    print(f"Total time: {total_minutes:.1f} min")


if __name__ == "__main__":
    main()
