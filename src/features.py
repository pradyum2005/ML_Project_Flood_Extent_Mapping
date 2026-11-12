"""Turn Sen1Floods11 chips into per-pixel features and sample pixels for training.

Each pixel gets 22 features:
    6 bands (blue, green, red, nir, swir1, swir2)
    4 indices (NDWI, MNDWI, NDVI, NDBI)
    5x5 window mean and variance of each band (12 features)

Usage (from the project folder):
    python src/features.py          # hand labels -> data/processed/samples.npz
    python src/features.py weak     # weak labels -> data/processed/samples_weak.npz
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

PROCESSED_DIR = Path("data/processed")

# Where each label source keeps its files, and where its samples are saved
SOURCES = {
    "hand": {
        "image_dir": Path("data/HandLabeled/S2Hand"),
        "label_dir": Path("data/HandLabeled/LabelHand"),
        "jrc_dir": Path("data/HandLabeled/JRCWaterHand"),
        "image_end": "_S2Hand.tif",
        "label_end": "_LabelHand.tif",
        "jrc_end": "_JRCWaterHand.tif",
        "output_name": "samples.npz",
    },
    "weak": {
        "image_dir": Path("data/WeaklyLabeled/S2Weak"),
        "label_dir": Path("data/WeaklyLabeled/S2IndexLabelWeak"),
        "jrc_dir": None,
        "image_end": "_S2Weak.tif",
        "label_end": "_S2IndexLabelWeak.tif",
        "jrc_end": None,
        "output_name": "samples_weak.npz",
    },
}

# Position of each band in the 13-band file (B1 is index 0, B2 is index 1, ...).
BAND_INDEX = {"blue": 1, "green": 2, "red": 3, "nir": 7, "swir1": 11, "swir2": 12}
BAND_ORDER = list(BAND_INDEX)

WINDOW = 5
REFLECTANCE_SCALE = 10000.0  # Sentinel-2 stores reflectance multiplied by 10000

FEATURE_NAMES = (
    BAND_ORDER
    + ["ndwi", "mndwi", "ndvi", "ndbi"]
    + [f"{b}_mean{WINDOW}" for b in BAND_ORDER]
    + [f"{b}_var{WINDOW}" for b in BAND_ORDER]
)


def list_chips(source="hand"):
    """Return chip ids, e.g. 'Bolivia_290290', in sorted order.

    A chip is listed only if both its image and its label file exist.
    For the weak source this means only the chips we downloaded.
    """
    paths = SOURCES[source]
    image_files = sorted(paths["image_dir"].glob("*" + paths["image_end"]))

    chip_ids = []
    for image_file in image_files:
        chip_id = image_file.name.replace(paths["image_end"], "")
        label_file = paths["label_dir"] / (chip_id + paths["label_end"])
        if label_file.exists():
            chip_ids.append(chip_id)
    return chip_ids


def load_chip(chip_id, source="hand"):
    """Read one chip. Returns (s2, label, jrc): (13, H, W), (H, W), (H, W).

    The weak source has no JRC file, so jrc is None there.
    """
    paths = SOURCES[source]

    with rasterio.open(paths["image_dir"] / (chip_id + paths["image_end"])) as f:
        s2 = f.read()
    with rasterio.open(paths["label_dir"] / (chip_id + paths["label_end"])) as f:
        label = f.read(1)

    jrc = None
    if paths["jrc_dir"] is not None:
        with rasterio.open(paths["jrc_dir"] / (chip_id + paths["jrc_end"])) as f:
            jrc = f.read(1)

    return s2, label, jrc


def normalized_difference(a, b):
    """(a - b) / (a + b), with 0 wherever a + b is 0 so we never divide by zero."""
    total = a + b
    out = np.zeros_like(a)
    np.divide(a - b, total, out=out, where=total != 0)
    return out


def window_mean_var(band, size=WINDOW):
    """Mean and variance of each pixel's size x size neighbourhood."""
    mean = uniform_filter(band, size=size, mode="reflect")
    mean_of_squares = uniform_filter(band * band, size=size, mode="reflect")
    var = np.maximum(mean_of_squares - mean * mean, 0)  # tiny negatives from rounding
    return mean, var


def compute_features(s2):
    """Compute the 22 features for every pixel. Returns an array of shape (H, W, 22)."""
    bands = {name: s2[i].astype(np.float32) / REFLECTANCE_SCALE
             for name, i in BAND_INDEX.items()}

    indices = [
        normalized_difference(bands["green"], bands["nir"]),    # NDWI
        normalized_difference(bands["green"], bands["swir1"]),  # MNDWI
        normalized_difference(bands["nir"], bands["red"]),      # NDVI
        normalized_difference(bands["swir1"], bands["nir"]),    # NDBI
    ]

    means, variances = [], []
    for name in BAND_ORDER:
        mean, var = window_mean_var(bands[name])
        means.append(mean)
        variances.append(var)

    layers = [bands[name] for name in BAND_ORDER] + indices + means + variances
    return np.stack(layers, axis=-1)


def valid_mask(s2, label):
    """True where a pixel has a real label (0 or 1) and is not empty in the image."""
    has_label = label != -1
    image_empty = (s2 == 0).all(axis=0)
    return has_label & ~image_empty


def sample_pixels(features, label, mask, n_per_class, rng):
    """Pick up to n_per_class water and n_per_class land pixels from valid pixels.

    Returns X (n, 22) and y (n,). Chips with few water pixels give fewer water rows.
    """
    rows, labels = [], []
    for value in (0, 1):
        ys, xs = np.nonzero(mask & (label == value))
        if len(ys) == 0:
            continue
        pick = rng.choice(len(ys), size=min(n_per_class, len(ys)), replace=False)
        rows.append(features[ys[pick], xs[pick]])
        labels.append(np.full(len(pick), value, dtype=np.int8))
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty(0, dtype=np.int8)
    return np.concatenate(rows), np.concatenate(labels)


def build_dataset(chip_ids, source="hand", n_per_class=1000, seed=0):
    """Sample pixels from every chip. Returns X, y, event (per row) and chip (per row)."""
    rng = np.random.default_rng(seed)
    X_parts, y_parts, event_parts, chip_parts = [], [], [], []

    for i, chip_id in enumerate(chip_ids, start=1):
        s2, label, _ = load_chip(chip_id, source)
        features = compute_features(s2)
        mask = valid_mask(s2, label)
        X, y = sample_pixels(features, label, mask, n_per_class, rng)

        X_parts.append(X)
        y_parts.append(y)
        event_parts.append(np.full(len(y), chip_id.split("_")[0]))
        chip_parts.append(np.full(len(y), chip_id))
        if i % 50 == 0:
            print(f"  processed {i}/{len(chip_ids)}")

    return (np.concatenate(X_parts), np.concatenate(y_parts),
            np.concatenate(event_parts), np.concatenate(chip_parts))


def main():
    # "hand" unless the command line says otherwise: python src/features.py weak
    source = "hand"
    if len(sys.argv) > 1:
        source = sys.argv[1]
    if source not in SOURCES:
        print("Unknown source:", source, "- use one of", list(SOURCES))
        return

    chip_ids = list_chips(source)
    print(f"Source: {source}")
    print(f"Building samples from {len(chip_ids)} chips, {len(FEATURE_NAMES)} features each")
    X, y, event, chip = build_dataset(chip_ids, source, n_per_class=1000)

    print(f"\nX shape: {X.shape}   water share: {100 * y.mean():.1f}%")
    print(f"NaNs in X: {int(np.isnan(X).sum())}")
    for name in sorted(set(event)):
        in_event = event == name
        print(f"  {name:<10} rows {in_event.sum():>7}   water {int((y[in_event] == 1).sum()):>7}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / SOURCES[source]["output_name"]
    np.savez_compressed(out_path, X=X, y=y, event=event, chip=chip,
                        feature_names=np.array(FEATURE_NAMES))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
