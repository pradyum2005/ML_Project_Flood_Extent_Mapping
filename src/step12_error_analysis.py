"""Error analysis and feature importance for the best setup (step 11).

Part 1: Permutation importance (LightGBM, natural sample, no class weight)
    For each held-out event: train on the other 10 events, then shuffle one
    feature at a time in the held-out event's rows and measure how much the
    IoU drops. A big drop means the model relies on that feature.
    Shuffling breaks the link between that feature and the label, while
    keeping its values realistic, so we never need to retrain.

Part 2: Error analysis, from the maps step 11 saved (no training needed)
    a. Per-chip scores for every model, so you can find the worst chips.
    b. Per-event error breakdown for every model:
         false_alarm_rate    : share of land pixels called water
         miss_rate           : share of water pixels called land
         edge_pixel_share    : share of valid pixels that are near a water edge
         edge_error_share    : share of all errors that are near a water edge
                               (if this is much bigger than edge_pixel_share,
                               errors pile up at edges, where labels are unsure)
         permanent_miss_rate : share of permanent (JRC) water that was missed
         flood_miss_rate     : share of flood water that was missed
    c. What the pixels look like: mean MNDWI, NDWI, NDVI, NIR and SWIR1 of
       true water, true land, false alarms and misses (best model only).
       If false alarms have water-like values, the features cannot tell them
       apart (for example wet soil or shadows).
    d. Error maps of the worst chips (best model only).

Needs:
    data/processed/samples_natural.npz       (step 10, or built if missing)
    data/processed/flood_map_predictions/    (step 11)

Saves (numbered, never overwritten):
    outputs/permutation_importance.csv  and  .png
    outputs/per_chip_errors.csv
    outputs/error_breakdown.csv
    outputs/error_feature_means.csv
    outputs/worst_chips/<chip>.png

Run from the project folder:
    python src/step12_error_analysis.py              # both parts (~10 min)
    python src/step12_error_analysis.py importance   # part 1 only (~5 min)
    python src/step12_error_analysis.py errors       # part 2 only (~5 min)
"""
import csv
import sys
import time

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation
from sklearn.inspection import permutation_importance

from step03_features import FEATURE_NAMES, compute_features, list_chips, load_chip, valid_mask
from step04_baselines import OUTPUT_DIR, PROCESSED_DIR, new_output_path
from step09_full_chip_scoring import scores_from_counts
from step10_class_imbalance import build_natural_sample
from step11_flood_maps import MAP_NAMES, make_rgb, make_unweighted_lightgbm, new_figure_folder

# The folder step 11 saved its maps to. If step 11 was run more than once,
# change this to the folder you want (for example "flood_map_predictions2").
MAP_FOLDER = PROCESSED_DIR / "flood_map_predictions"

BEST_MODEL = "lightgbm"

# Permutation importance: how many times each feature is shuffled
N_REPEATS = 5
SEED = 0

# A pixel is "near an edge" if both water and land are within this many pixels
EDGE_WIDTH = 2

# Worst-chip figures: only chips with at least this many water pixels count,
# because IoU on a chip with almost no water is not meaningful
MIN_WATER_PIXELS = 1000
WORST_CHIPS_TO_PLOT = 3

# Features described in part 2c
FEATURES_TO_DESCRIBE = ["mndwi", "ndwi", "ndvi", "nir", "swir1"]
OUTCOMES = ["true_water", "true_land", "false_alarm", "miss"]

# Colours for the error map values -1, 0, 1, 2, 3
ERROR_MAP_COLOURS = ListedColormap(["lightgray", "white", "deepskyblue", "red", "orange"])


def new_png_path(base_name):
    """A .png path in outputs/ that does not exist yet (like new_output_path for CSVs)."""
    path = OUTPUT_DIR / f"{base_name}.png"
    number = 2
    while path.exists():
        path = OUTPUT_DIR / f"{base_name}{number}.png"
        number += 1
    return path


def save_csv(header, rows, path):
    """Save a table as a CSV file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Part 1: permutation importance
# ---------------------------------------------------------------------------

def run_permutation_importance(chip_ids):
    """Permutation importance of every feature, one held-out event at a time."""
    print("\nPart 1: permutation importance")
    X, y, event = build_natural_sample(chip_ids)
    events = sorted(set(event))

    all_rows = []
    importance_per_event = []

    for held_out in events:
        start = time.time()
        train_rows = event != held_out
        test_rows = event == held_out

        model = make_unweighted_lightgbm()
        model.fit(X[train_rows], y[train_rows])

        # scoring="jaccard" is the IoU of the water class
        # n_jobs=1 because LightGBM already uses all CPU cores itself
        result = permutation_importance(
            model,
            X[test_rows],
            y[test_rows],
            scoring="jaccard",
            n_repeats=N_REPEATS,
            random_state=SEED,
            n_jobs=1,
        )

        for i, feature in enumerate(FEATURE_NAMES):
            mean_drop = round(float(result.importances_mean[i]), 4)
            std_drop = round(float(result.importances_std[i]), 4)
            all_rows.append([feature, held_out, mean_drop, std_drop])

        importance_per_event.append(result.importances_mean)
        print(f"    finished {held_out} in {time.time() - start:.0f} s")

    # Average over the 11 events
    mean_importance = np.mean(importance_per_event, axis=0)
    spread_between_events = np.std(importance_per_event, axis=0)
    for i, feature in enumerate(FEATURE_NAMES):
        all_rows.append([feature, "MEAN", round(float(mean_importance[i]), 4),
                         round(float(spread_between_events[i]), 4)])

    out_path = new_output_path("permutation_importance")
    save_csv(["feature", "event", "iou_drop_mean", "iou_drop_std"], all_rows, out_path)
    print("Saved table to", out_path)

    # Print the features from most to least important
    order = np.argsort(mean_importance)[::-1]
    print("\nfeature          IoU drop (mean over events)")
    for i in order:
        print(f"{FEATURE_NAMES[i]:<16} {mean_importance[i]:8.4f}")

    save_importance_chart(mean_importance, spread_between_events)


def save_importance_chart(mean_importance, spread_between_events):
    """Horizontal bar chart, most important feature at the top."""
    order = np.argsort(mean_importance)  # smallest first, so the biggest ends up on top
    names = []
    values = []
    spreads = []
    for i in order:
        names.append(FEATURE_NAMES[i])
        values.append(mean_importance[i])
        spreads.append(spread_between_events[i])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(names, values, xerr=spreads, color="steelblue", ecolor="gray")
    ax.set_xlabel("Drop in IoU when the feature is shuffled (mean over 11 events)")
    ax.set_title("Permutation importance, LightGBM (natural sample, no class weight)")
    plt.tight_layout()

    out_path = new_png_path("permutation_importance")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print("Saved chart to", out_path)


# ---------------------------------------------------------------------------
# Part 2: error analysis
# ---------------------------------------------------------------------------

def empty_counts():
    """All the counters used for one (model, event) pair."""
    counts = {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "valid_pixels": 0, "edge_pixels": 0, "errors": 0, "edge_errors": 0,
        "permanent_water": 0, "permanent_missed": 0,
        "flood_water": 0, "flood_missed": 0,
    }
    return counts


def near_water_edge(label, width=EDGE_WIDTH):
    """True where both water and land are within `width` pixels."""
    water = label == 1
    land = label == 0
    near_water = binary_dilation(water, iterations=width)
    near_land = binary_dilation(land, iterations=width)
    return near_water & near_land


def update_counts(counts, y_true, y_pred, near_edge, permanent):
    """Add one chip's pixels to the counters. All inputs are 1-D (valid pixels only)."""
    counts["tp"] += int(((y_pred == 1) & (y_true == 1)).sum())
    counts["fp"] += int(((y_pred == 1) & (y_true == 0)).sum())
    counts["fn"] += int(((y_pred == 0) & (y_true == 1)).sum())
    counts["tn"] += int(((y_pred == 0) & (y_true == 0)).sum())

    errors = y_pred != y_true
    counts["valid_pixels"] += len(y_true)
    counts["edge_pixels"] += int(near_edge.sum())
    counts["errors"] += int(errors.sum())
    counts["edge_errors"] += int((errors & near_edge).sum())

    permanent_water = (y_true == 1) & permanent
    flood_water = (y_true == 1) & ~permanent
    counts["permanent_water"] += int(permanent_water.sum())
    counts["permanent_missed"] += int((permanent_water & (y_pred == 0)).sum())
    counts["flood_water"] += int(flood_water.sum())
    counts["flood_missed"] += int((flood_water & (y_pred == 0)).sum())


def safe_share(part, whole):
    """part / whole, rounded, or 0 when whole is 0."""
    return round(part / max(whole, 1), 3)


def breakdown_row(model_name, event_name, counts):
    """One row of the error breakdown table."""
    c = counts
    row = [
        event_name,
        model_name,
        safe_share(c["fp"], c["fp"] + c["tn"]),            # false_alarm_rate
        safe_share(c["fn"], c["tp"] + c["fn"]),            # miss_rate
        safe_share(c["edge_pixels"], c["valid_pixels"]),   # edge_pixel_share
        safe_share(c["edge_errors"], c["errors"]),         # edge_error_share
        safe_share(c["permanent_missed"], c["permanent_water"]),  # permanent_miss_rate
        safe_share(c["flood_missed"], c["flood_water"]),   # flood_miss_rate
    ]
    return row


def outcome_masks(y_true, y_pred):
    """Which valid pixels are true water, true land, false alarms and misses."""
    masks = {
        "true_water": (y_pred == 1) & (y_true == 1),
        "true_land": (y_pred == 0) & (y_true == 0),
        "false_alarm": (y_pred == 1) & (y_true == 0),
        "miss": (y_pred == 0) & (y_true == 1),
    }
    return masks


def make_error_map(label, valid, predicted):
    """-1 = no data, 0 = land correct, 1 = water correct, 2 = false alarm, 3 = miss."""
    error_map = np.full(valid.shape, -1, dtype=np.int8)
    error_map[valid & (predicted == 0) & (label == 0)] = 0
    error_map[valid & (predicted == 1) & (label == 1)] = 1
    error_map[valid & (predicted == 1) & (label == 0)] = 2
    error_map[valid & (predicted == 0) & (label == 1)] = 3
    return error_map


def save_error_figure(folder, chip_id, iou):
    """Image, truth and the best model's error map for one chip."""
    s2, label, jrc = load_chip(chip_id, "hand")
    maps = np.load(MAP_FOLDER / f"{chip_id}.npz")
    valid = maps["valid"]
    predicted = maps[BEST_MODEL]

    truth = np.full(valid.shape, -1, dtype=np.int8)
    truth[valid] = label[valid]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(make_rgb(s2))
    axes[0].set_title("Sentinel-2 true colour")
    axes[1].imshow(truth, cmap=ListedColormap(["lightgray", "white", "deepskyblue"]),
                   vmin=-1, vmax=1)
    axes[1].set_title("Hand label (truth)")
    axes[2].imshow(make_error_map(label, valid, predicted), cmap=ERROR_MAP_COLOURS,
                   vmin=-1, vmax=3)
    axes[2].set_title(f"{BEST_MODEL} errors (IoU {iou:.2f})")
    for ax in axes:
        ax.axis("off")

    legend = [
        Patch(facecolor="lightgray", edgecolor="black", label="no data"),
        Patch(facecolor="white", edgecolor="black", label="land (correct)"),
        Patch(facecolor="deepskyblue", edgecolor="black", label="water (correct)"),
        Patch(facecolor="red", edgecolor="black", label="false alarm"),
        Patch(facecolor="orange", edgecolor="black", label="miss"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5)
    fig.suptitle(chip_id, fontsize=16)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    fig.savefig(folder / f"{chip_id}.png", dpi=100)
    plt.close(fig)


def run_error_analysis(chip_ids):
    """Part 2: per-chip scores, error breakdown, feature means, worst-chip figures."""
    print("\nPart 2: error analysis, reading maps from", MAP_FOLDER)
    if not MAP_FOLDER.exists():
        raise SystemExit(f"{MAP_FOLDER} not found. Run step 11 first.")

    describe_columns = []
    for name in FEATURES_TO_DESCRIBE:
        describe_columns.append(FEATURE_NAMES.index(name))

    # counts[(model, event)] -> counters; event "ALL" collects every chip
    counts = {}
    # feature_sums[(event, outcome)] -> sum of each described feature; plus pixel counts
    feature_sums = {}
    feature_pixels = {}

    per_chip_rows = []
    best_model_chips = []  # (iou, chip_id) for chips with enough water

    for chip_id in chip_ids:
        map_path = MAP_FOLDER / f"{chip_id}.npz"
        if not map_path.exists():
            continue  # chips with no valid pixels were not saved

        event_name = chip_id.split("_")[0]
        maps = np.load(map_path)
        label = maps["label"]
        valid = maps["valid"]

        s2, _, jrc = load_chip(chip_id, "hand")
        y_true = label[valid]
        near_edge = near_water_edge(label)[valid]
        permanent = jrc[valid] == 1

        # a. and b. Per-chip scores and the error counters, for every model
        water_pixels = int((y_true == 1).sum())
        chip_row = [
            chip_id,
            event_name,
            len(y_true),
            round(100 * water_pixels / len(y_true), 2),
            round(100 * int(permanent.sum()) / len(y_true), 2),
        ]

        for model_name in MAP_NAMES:
            y_pred = maps[model_name][valid]

            for key in [(model_name, event_name), (model_name, "ALL")]:
                if key not in counts:
                    counts[key] = empty_counts()
                update_counts(counts[key], y_true, y_pred, near_edge, permanent)

            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            iou, f1, precision, recall = scores_from_counts(tp, fp, fn)
            chip_row.append(iou)

            if model_name == BEST_MODEL:
                chip_row.append(fp)
                chip_row.append(fn)
                if water_pixels >= MIN_WATER_PIXELS:
                    best_model_chips.append((iou, chip_id))

        per_chip_rows.append(chip_row)

        # c. Feature means per outcome, best model only
        features = compute_features(s2)
        X_described = features[valid][:, describe_columns]
        masks = outcome_masks(y_true, maps[BEST_MODEL][valid])

        for outcome in OUTCOMES:
            for key in [(event_name, outcome), ("ALL", outcome)]:
                if key not in feature_sums:
                    feature_sums[key] = np.zeros(len(describe_columns))
                    feature_pixels[key] = 0
                feature_sums[key] += X_described[masks[outcome]].sum(axis=0)
                feature_pixels[key] += int(masks[outcome].sum())

    # a. Per-chip table
    header = ["chip", "event", "valid_pixels", "water_pct", "permanent_water_pct"]
    for model_name in MAP_NAMES:
        header.append(f"iou_{model_name}")
        if model_name == BEST_MODEL:
            header.append(f"false_alarms_{model_name}")
            header.append(f"misses_{model_name}")
    out_path = new_output_path("per_chip_errors")
    save_csv(header, per_chip_rows, out_path)
    print("Saved per-chip table to", out_path)

    # b. Error breakdown table
    events = []
    for row in per_chip_rows:
        if row[1] not in events:
            events.append(row[1])
    events.sort()
    breakdown_rows = []
    for event_name in events + ["ALL"]:
        for model_name in MAP_NAMES:
            breakdown_rows.append(breakdown_row(model_name, event_name,
                                                counts[(model_name, event_name)]))
    header = ["event", "model", "false_alarm_rate", "miss_rate", "edge_pixel_share",
              "edge_error_share", "permanent_miss_rate", "flood_miss_rate"]
    out_path = new_output_path("error_breakdown")
    save_csv(header, breakdown_rows, out_path)
    print("Saved error breakdown to", out_path)

    print(f"\nError breakdown for {BEST_MODEL}")
    print("event       false_alarm   miss   edge_pixels  edge_errors  perm_miss  flood_miss")
    for row in breakdown_rows:
        if row[1] == BEST_MODEL:
            print(f"{row[0]:<11} {row[2]:11.3f} {row[3]:6.3f} {row[4]:12.3f} "
                  f"{row[5]:12.3f} {row[6]:10.3f} {row[7]:11.3f}")

    # c. Feature means table
    feature_rows = []
    for event_name in events + ["ALL"]:
        for outcome in OUTCOMES:
            key = (event_name, outcome)
            pixels = feature_pixels[key]
            row = [event_name, outcome, pixels]
            for value in feature_sums[key] / max(pixels, 1):
                row.append(round(float(value), 4))
            feature_rows.append(row)
    header = ["event", "outcome", "pixels"] + FEATURES_TO_DESCRIBE
    out_path = new_output_path("error_feature_means")
    save_csv(header, feature_rows, out_path)
    print("\nSaved feature means to", out_path)

    print(f"\nMean feature values, all events ({BEST_MODEL})")
    print("outcome          pixels  " + "".join(f"{name:>9}" for name in FEATURES_TO_DESCRIBE))
    for row in feature_rows:
        if row[0] == "ALL":
            values = "".join(f"{value:9.3f}" for value in row[3:])
            print(f"{row[1]:<13} {row[2]:>10}  {values}")

    # d. Error maps of the worst chips (lowest IoU first)
    best_model_chips.sort()
    figure_folder = new_figure_folder("worst_chips")
    figure_folder.mkdir(parents=True)
    print(f"\nWorst chips for {BEST_MODEL} (at least {MIN_WATER_PIXELS} water pixels):")
    for iou, chip_id in best_model_chips[:WORST_CHIPS_TO_PLOT]:
        save_error_figure(figure_folder, chip_id, iou)
        print(f"    {chip_id}: IoU {iou:.3f}")
    print("Saved figures to", figure_folder)


def main():
    part = "both"
    if len(sys.argv) > 1:
        part = sys.argv[1]
    if part not in ["both", "importance", "errors"]:
        raise SystemExit("Use: importance, errors, or nothing for both")

    chip_ids = list_chips("hand")
    start = time.time()

    if part in ["both", "importance"]:
        run_permutation_importance(chip_ids)
    if part in ["both", "errors"]:
        run_error_analysis(chip_ids)

    print(f"\nTotal time: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
