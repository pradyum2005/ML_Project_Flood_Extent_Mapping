"""Summarize the whole hand-labeled set, one row per flood event.

Usage (from the project folder):
    python src/summarize_data.py

Prints a table and saves it to outputs/data_summary.csv.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

DATA_DIR = Path("data/HandLabeled")
OUTPUT_DIR = Path("outputs")

# A chip with less than this share of water (among valid pixels) counts as "almost no water".
LOW_WATER_SHARE = 0.01


def read_band(path, band=1):
    """Read one band of a .tif file as a 2D array."""
    with rasterio.open(path) as f:
        return f.read(band)


def main():
    label_files = sorted((DATA_DIR / "LabelHand").glob("*_LabelHand.tif"))
    print(f"Found {len(label_files)} chips\n")

    # One dictionary of running totals per event, created automatically on first use.
    events = defaultdict(lambda: defaultdict(int))

    for i, label_path in enumerate(label_files, start=1):
        chip_id = label_path.name.replace("_LabelHand.tif", "")
        event = chip_id.split("_")[0]

        label = read_band(label_path)
        jrc = read_band(DATA_DIR / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif")
        with rasterio.open(DATA_DIR / "S2Hand" / f"{chip_id}_S2Hand.tif") as f:
            s2 = f.read()

        water = int((label == 1).sum())
        land = int((label == 0).sum())
        nodata = int((label == -1).sum())
        valid = water + land

        # A pixel is "empty" in the image if all 13 bands are exactly 0.
        s2_empty = (s2 == 0).all(axis=0)

        totals = events[event]
        totals["chips"] += 1
        totals["water"] += water
        totals["land"] += land
        totals["nodata"] += nodata
        totals["low_water_chips"] += int(valid > 0 and water / valid < LOW_WATER_SHARE)
        totals["no_valid_chips"] += int(valid == 0)
        totals["jrc_chips"] += int((jrc == 1).any())
        totals["s2_empty"] += int(s2_empty.sum())
        totals["s2_empty_but_labeled"] += int((s2_empty & (label != -1)).sum())

        if i % 50 == 0:
            print(f"  processed {i}/{len(label_files)}")

    header = ["event", "chips", "water_%", "land_%", "nodata_%",
              "low_water_chips", "all_nodata_chips", "chips_with_jrc",
              "s2_empty_px", "s2_empty_but_labeled_px"]
    rows = []
    for event in sorted(events):
        t = events[event]
        total_px = t["water"] + t["land"] + t["nodata"]
        rows.append([
            event,
            t["chips"],
            round(100 * t["water"] / total_px, 1),
            round(100 * t["land"] / total_px, 1),
            round(100 * t["nodata"] / total_px, 1),
            t["low_water_chips"],
            t["no_valid_chips"],
            t["jrc_chips"],
            t["s2_empty"],
            t["s2_empty_but_labeled"],
        ])

    # Totals row across all events.
    all_water = sum(events[e]["water"] for e in events)
    all_land = sum(events[e]["land"] for e in events)
    all_nodata = sum(events[e]["nodata"] for e in events)
    all_px = all_water + all_land + all_nodata
    rows.append([
        "ALL",
        sum(events[e]["chips"] for e in events),
        round(100 * all_water / all_px, 1),
        round(100 * all_land / all_px, 1),
        round(100 * all_nodata / all_px, 1),
        sum(events[e]["low_water_chips"] for e in events),
        sum(events[e]["no_valid_chips"] for e in events),
        sum(events[e]["jrc_chips"] for e in events),
        sum(events[e]["s2_empty"] for e in events),
        sum(events[e]["s2_empty_but_labeled"] for e in events),
    ])

    print()
    print(f"{header[0]:<10}" + "".join(f"{h:>{max(len(h), 8) + 2}}" for h in header[1:]))
    for row in rows:
        line = f"{row[0]:<10}"
        for h, value in zip(header[1:], row[1:]):
            line += f"{value:>{max(len(h), 8) + 2}}"
        print(line)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "data_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nSaved table to {out_path}")


if __name__ == "__main__":
    main()
