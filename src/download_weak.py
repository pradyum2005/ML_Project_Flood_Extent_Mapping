"""Download a random subset of the weakly labeled Sentinel-2 images.

The weak labels (S2IndexLabelWeak) are already in data/WeaklyLabeled/.
This script picks NUM_CHIPS of those chips, the same number from every
event (random inside each event), and downloads the matching images
(S2Weak) from the public sen1floods11 bucket.

Run from the project folder:
    python src/download_weak.py

It is safe to stop and run again: files already downloaded are skipped.
To get more chips later, raise NUM_CHIPS. The chips already chosen stay
the same because the shuffle uses a fixed seed.
"""
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NUM_CHIPS = 2000
SEED = 0
NUM_WORKERS = 8
NUM_TRIES = 3

# Events that have weak labels but no hand labels, so they cannot be tested on
SKIP_EVENTS = ["Colombia"]

BUCKET_URL = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2Weak"
LABEL_DIR = Path("data/WeaklyLabeled/S2IndexLabelWeak")
IMAGE_DIR = Path("data/WeaklyLabeled/S2Weak")


def list_chip_ids():
    """Chip ids (like Ghana_103272) for every weak label file we already have.

    Chips from the events in SKIP_EVENTS are left out.
    """
    label_files = sorted(LABEL_DIR.glob("*_S2IndexLabelWeak.tif"))
    chip_ids = []
    for label_file in label_files:
        chip_id = label_file.name.replace("_S2IndexLabelWeak.tif", "")
        event = chip_id.split("_")[0]
        if event in SKIP_EVENTS:
            continue
        chip_ids.append(chip_id)
    return chip_ids


def choose_chips(chip_ids, how_many):
    """Pick how_many chips: random inside each event, the same number from every event.

    Chips are handed out one per event in turn. An event that runs out of chips
    is skipped, so the chips it cannot supply go to the other events.
    """
    # Group the chips by event, and shuffle each group with a fixed seed
    by_event = {}
    for chip_id in chip_ids:
        event = chip_id.split("_")[0]
        if event not in by_event:
            by_event[event] = []
        by_event[event].append(chip_id)

    for event in sorted(by_event):
        random.Random(SEED).shuffle(by_event[event])

    # Take the 1st chip of every event, then the 2nd of every event, and so on
    chosen = []
    position = 0
    while len(chosen) < how_many:
        added_any = False
        for event in sorted(by_event):
            chips_of_event = by_event[event]
            if position < len(chips_of_event) and len(chosen) < how_many:
                chosen.append(chips_of_event[position])
                added_any = True
        if not added_any:
            break
        position += 1

    return chosen


def download_one(chip_id):
    """Download one image. Returns 'skipped', 'downloaded' or 'failed'."""
    file_name = f"{chip_id}_S2Weak.tif"
    final_path = IMAGE_DIR / file_name
    partial_path = IMAGE_DIR / (file_name + ".part")

    if final_path.exists():
        return "skipped"

    url = f"{BUCKET_URL}/{file_name}"
    for attempt in range(NUM_TRIES):
        try:
            urllib.request.urlretrieve(url, partial_path)
            # Rename only when the download finished, so a half file is never kept
            partial_path.rename(final_path)
            return "downloaded"
        except Exception as error:
            print(f"  {file_name} attempt {attempt + 1} failed: {error}")
            time.sleep(2)

    return "failed"


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    all_chips = list_chip_ids()
    print("Weak label files found:", len(all_chips))
    if len(all_chips) == 0:
        print("No labels found in", LABEL_DIR)
        return

    chosen = choose_chips(all_chips, NUM_CHIPS)
    print("Chips chosen:", len(chosen))

    # Show how many chips each event has in the chosen set
    counts = {}
    for chip_id in chosen:
        event = chip_id.split("_")[0]
        counts[event] = counts.get(event, 0) + 1
    for event in sorted(counts):
        print(f"  {event:<10} {counts[event]}")

    results = {"skipped": 0, "downloaded": 0, "failed": 0}
    failed_chips = []
    done = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for chip_id, result in zip(chosen, pool.map(download_one, chosen)):
            results[result] += 1
            if result == "failed":
                failed_chips.append(chip_id)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(chosen)} done")

    print()
    print("Downloaded:", results["downloaded"])
    print("Already there:", results["skipped"])
    print("Failed:", results["failed"])
    if failed_chips:
        print("Run the script again to retry these:", failed_chips[:10])
    print("Images are in", IMAGE_DIR)


if __name__ == "__main__":
    main()
