"""Flood maps with the best setup from step 10, for three models.

Step 10 showed the best setup is: train on the natural sample (real water
share, ~10% water) with NO class weight. Here we use that setup for:
    LightGBM, Random Forest and MLP
(and the MNDWI threshold, which needs no training, as the baseline).

For each held-out event:
    1. train each model on the natural sample of the other 10 events,
    2. predict every valid pixel of every chip of the held-out event,
    3. score two ways:
         all_water  : every valid pixel (same as steps 9 and 10)
         flood_only : leave out JRC permanent water (rivers, lakes), so we
                      score only the water that the flood added
    4. save a flood map figure for the chip with the most flood pixels.

Flood map colours:
    grey  = no data,  white = land,
    blue  = flood water (water that is not permanent),
    navy  = permanent water (JRC says it is always water)

Saves:
    outputs/flood_map_results.csv          scores (numbered, never overwritten)
    outputs/flood_maps/<chip>.png          one figure per event (numbered folder)
    data/processed/flood_map_predictions/  every chip's predicted maps, for step 12
                                           (numbered folder)

Needs data/processed/samples_natural.npz from step 10. If it is missing,
it is built from the chips first (about 3 minutes).

Run from the project folder:
    python src/step11_flood_maps.py
Takes about 1 hour on an 8-core laptop (Random Forest and MLP are the slow
ones). Keep the laptop plugged in and stop it from sleeping.
"""
import csv
import time

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from step03_features import compute_features, list_chips, load_chip, valid_mask
from step04_baselines import OUTPUT_DIR, new_output_path
from step05_tree_models import make_lightgbm, make_random_forest
from step06_mlp import make_mlp
from step09_full_chip_scoring import (count_errors, new_folder_path, predict_pixels,
                                      save_chip_maps, scores_from_counts)
from step10_class_imbalance import build_natural_sample


def make_unweighted_lightgbm():
    """LightGBM with the step 5 settings, but no class weight."""
    model = make_lightgbm()
    model.set_params(class_weight=None)
    return model


def make_unweighted_random_forest():
    """Random Forest with the step 5 settings, but no class weight."""
    model = make_random_forest()
    model.set_params(class_weight=None)
    return model


# The MLP has no class weight option, so it is already "unweighted".
MODELS = {
    "lightgbm": make_unweighted_lightgbm,
    "random_forest": make_unweighted_random_forest,
    "mlp": make_mlp,
}

MAP_NAMES = ["mndwi_threshold"] + list(MODELS)
SCORED_ON = ["all_water", "flood_only"]

# Colours for the flood map values -1, 0, 1, 2
FLOOD_MAP_COLOURS = ListedColormap(["lightgray", "white", "deepskyblue", "navy"])


def new_figure_folder(base_name):
    """A folder in outputs/ that does not exist yet, so old figures are never overwritten."""
    path = OUTPUT_DIR / base_name
    number = 2
    while path.exists():
        path = OUTPUT_DIR / f"{base_name}{number}"
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


def add_counts(counts, key, y_true, y_pred):
    """Add this chip's true positives, false positives and false negatives to counts[key]."""
    tp, fp, fn = count_errors(y_true, y_pred)
    counts[key][0] += tp
    counts[key][1] += fp
    counts[key][2] += fn


def make_flood_map(water, valid, jrc):
    """Turn a full-size water mask into a flood map.

    -1 = no data, 0 = land, 1 = flood water, 2 = permanent water (JRC).
    """
    flood_map = np.full(valid.shape, -1, dtype=np.int8)
    flood_map[valid] = 0
    flood_map[valid & water] = 1
    flood_map[valid & water & (jrc == 1)] = 2
    return flood_map


def make_rgb(s2):
    """True-colour image (red = B4, green = B3, blue = B2), stretched to 0-1 for display."""
    rgb = np.stack([s2[3], s2[2], s2[1]], axis=-1).astype(float)
    bright = max(np.percentile(rgb, 99), 1)
    rgb = np.clip(rgb / bright, 0, 1)
    return rgb


def save_flood_figure(folder, chip_id, s2, label, jrc, valid, predictions):
    """Save one figure: the image, the true flood map, and each model's flood map."""
    true_water = label == 1
    true_flood_map = make_flood_map(true_water, valid, jrc)

    # Pixels used for the flood-only score: valid and not permanent water
    not_permanent = jrc[valid] != 1
    y_pixels = label[valid]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    axes = axes.flatten()

    axes[0].imshow(make_rgb(s2))
    axes[0].set_title("Sentinel-2 true colour")

    axes[1].imshow(true_flood_map, cmap=FLOOD_MAP_COLOURS, vmin=-1, vmax=2)
    axes[1].set_title("Hand label (truth)")

    for i, name in enumerate(MAP_NAMES):
        predicted_water = np.zeros(valid.shape, dtype=bool)
        predicted_water[valid] = predictions[name] == 1
        flood_map = make_flood_map(predicted_water, valid, jrc)

        tp, fp, fn = count_errors(y_pixels[not_permanent], predictions[name][not_permanent])
        iou, f1, precision, recall = scores_from_counts(tp, fp, fn)

        ax = axes[i + 2]
        ax.imshow(flood_map, cmap=FLOOD_MAP_COLOURS, vmin=-1, vmax=2)
        ax.set_title(f"{name}  (flood IoU {iou:.2f})")

    for ax in axes:
        ax.axis("off")

    legend = [
        Patch(facecolor="lightgray", edgecolor="black", label="no data"),
        Patch(facecolor="white", edgecolor="black", label="land"),
        Patch(facecolor="deepskyblue", edgecolor="black", label="flood water"),
        Patch(facecolor="navy", edgecolor="black", label="permanent water (JRC)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4)
    fig.suptitle(chip_id, fontsize=16)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    fig.savefig(folder / f"{chip_id}.png", dpi=100)
    plt.close(fig)


def save_results(rows, path):
    """Save the results table as a CSV file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    header = ["model", "scored_on", "event", "iou", "f1", "precision", "recall"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    chip_ids = list_chips("hand")
    X_natural, y_natural, event_natural = build_natural_sample(chip_ids)
    events = sorted(set(event_natural))
    print(f"Natural sample: {len(y_natural)} rows, {100 * y_natural.mean():.1f}% water")

    figure_folder = new_figure_folder("flood_maps")
    figure_folder.mkdir(parents=True)
    map_folder = new_folder_path("flood_map_predictions")
    map_folder.mkdir(parents=True)
    print("Saving figures to", figure_folder)
    print("Saving chip maps to", map_folder)

    # counts[(model, scored_on, event)] = [tp, fp, fn]
    counts = {}
    for name in MAP_NAMES:
        for scored_on in SCORED_ON:
            for event_name in events:
                counts[(name, scored_on, event_name)] = [0, 0, 0]

    run_start = time.time()

    for held_out in events:
        event_start = time.time()
        print(f"\n{held_out}")

        # 1. Train on the natural sample of the other 10 events
        train_rows = event_natural != held_out
        trained = train_models(X_natural[train_rows], y_natural[train_rows])

        # The chip with the most flood pixels gets a figure
        best_chip = None
        best_flood_pixels = -1

        # 2. Predict every valid pixel of every chip of the held-out event
        for chip_id in chip_ids:
            if chip_id.split("_")[0] != held_out:
                continue

            s2, label, jrc = load_chip(chip_id, "hand")
            valid = valid_mask(s2, label)
            if valid.sum() == 0:
                continue

            features = compute_features(s2)
            X_pixels = features[valid]
            y_pixels = label[valid]
            not_permanent = jrc[valid] != 1

            predictions = predict_pixels(trained, X_pixels)

            # 3. Score: all water, and flood water only (permanent water left out)
            for name in MAP_NAMES:
                add_counts(counts, (name, "all_water", held_out),
                           y_pixels, predictions[name])
                add_counts(counts, (name, "flood_only", held_out),
                           y_pixels[not_permanent], predictions[name][not_permanent])

            save_chip_maps(map_folder, chip_id, label, valid, predictions)

            flood_pixels = int(((y_pixels == 1) & not_permanent).sum())
            if flood_pixels > best_flood_pixels:
                best_flood_pixels = flood_pixels
                best_chip = (chip_id, s2, label, jrc, valid, predictions)

        # 4. Figure for the chip with the most flood pixels
        if best_chip is not None:
            chip_id, s2, label, jrc, valid, predictions = best_chip
            save_flood_figure(figure_folder, chip_id, s2, label, jrc, valid, predictions)
            print(f"    saved figure for {chip_id} ({best_flood_pixels} flood pixels)")

        minutes = (time.time() - event_start) / 60
        print(f"    finished {held_out} in {minutes:.1f} min")

    # 5. Turn the counts into scores: one row per model, scoring and event, plus ALL
    all_rows = []
    for name in MAP_NAMES:
        for scored_on in SCORED_ON:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            for event_name in events:
                tp, fp, fn = counts[(name, scored_on, event_name)]
                total_tp += tp
                total_fp += fp
                total_fn += fn
                iou, f1, precision, recall = scores_from_counts(tp, fp, fn)
                all_rows.append([name, scored_on, event_name, iou, f1, precision, recall])

            iou, f1, precision, recall = scores_from_counts(total_tp, total_fp, total_fn)
            all_rows.append([name, scored_on, "ALL", iou, f1, precision, recall])

    # Print only the ALL rows; the CSV has every event
    print("\nmodel             scored_on       IoU      F1  precision    recall")
    for row in all_rows:
        name, scored_on, event_name, iou, f1, precision, recall = row
        if event_name == "ALL":
            print(f"{name:<17} {scored_on:<11} {iou:7.3f} {f1:7.3f} {precision:10.3f} {recall:9.3f}")

    out_path = new_output_path("flood_map_results")
    save_results(all_rows, out_path)
    print("\nSaved table to", out_path)

    total_minutes = (time.time() - run_start) / 60
    print(f"Total time: {total_minutes:.1f} min")


if __name__ == "__main__":
    main()
