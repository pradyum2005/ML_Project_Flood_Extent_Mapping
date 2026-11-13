"""Inspect one Sen1Floods11 hand-labeled chip: print its stats and plot it.

Usage (from the project folder):
    python src/step01_inspect_chip.py                      # inspects the first chip found
    python src/step01_inspect_chip.py Ghana_103272         # inspects a specific chip
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window, so Tk is not needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import rasterio

DATA_DIR = Path("data/HandLabeled")
OUTPUT_DIR = Path("outputs")

# Sentinel-2 band names in the order they are stored in the 13-band file.
BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
              "B8", "B8A", "B9", "B10", "B11", "B12"]


def find_chip_id():
    """Return the chip id from the command line, or the first chip on disk."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    first_file = sorted((DATA_DIR / "S2Hand").glob("*_S2Hand.tif"))[0]
    return first_file.name.replace("_S2Hand.tif", "")


def main():
    chip_id = find_chip_id()
    print(f"Chip: {chip_id}")

    with rasterio.open(DATA_DIR / "S2Hand" / f"{chip_id}_S2Hand.tif") as f:
        s2 = f.read()  # shape: (bands, height, width)
    with rasterio.open(DATA_DIR / "LabelHand" / f"{chip_id}_LabelHand.tif") as f:
        label = f.read(1)  # shape: (height, width)
    with rasterio.open(DATA_DIR / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif") as f:
        jrc = f.read(1)

    print(f"S2 shape: {s2.shape}  dtype: {s2.dtype}")
    print(f"Label shape: {label.shape}  JRC shape: {jrc.shape}")

    print("\nPer-band min / max / mean:")
    for name, band in zip(BAND_NAMES, s2):
        print(f"  {name:>4}: {band.min():9.1f} {band.max():9.1f} {band.mean():9.1f}")

    total = label.size
    print("\nLabel values (1 = water, 0 = land, -1 = no data):")
    for value in (1, 0, -1):
        count = int((label == value).sum())
        print(f"  {value:>2}: {count:7d} pixels ({100 * count / total:.1f}%)")
    print(f"JRC permanent water pixels: {int((jrc == 1).sum())}")

    # True-colour image: red = B4, green = B3, blue = B2 (indices 3, 2, 1).
    rgb = np.stack([s2[3], s2[2], s2[1]], axis=-1).astype(float)
    rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)  # stretch to 0-1 for display

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 true colour")
    # One colour per label value, in order: -1 (no data), 0 (land), 1 (water).
    label_cmap = ListedColormap(["red", "white", "blue"])
    axes[1].imshow(label, cmap=label_cmap, vmin=-1, vmax=1)
    axes[1].set_title("Hand label (red=no data, white=land, blue=water)")
    axes[2].imshow(jrc, cmap="Blues")
    axes[2].set_title("JRC permanent water")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(chip_id)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{chip_id}.png"
    fig.savefig(out_path, dpi=100)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
