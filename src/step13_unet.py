"""U-Net trained from scratch, scored the same way as steps 9-11.

Our other models look at one pixel at a time (plus a 5x5 window). A U-Net
looks at the whole image: its convolution layers learn their own spatial
features, and each pixel's prediction can use context from far around it.

It uses the same 6 bands as our features (blue, green, red, nir, swir1, swir2),
the same leave-one-event-out idea, and the same full-chip scoring:
    all_water  : every valid pixel of the held-out event's chips
    flood_only : the same, but JRC permanent water left out
so its numbers compare directly with steps 10 and 11 (LightGBM: 0.844 all_water).

Training details:
    - inputs: the 6 bands / 10000, then standardised with the mean and std of
      the training chips (like StandardScaler); invalid pixels are set to 0
    - loss: binary cross-entropy on valid pixels only, NO class weight
      (step 10 showed that class weighting causes false alarms)
    - each epoch: one random 256x256 crop of every training chip, with random flips
    - 10% of the training chips are kept aside for validation; the epoch with the
      best validation IoU is kept (saved to models/unet_<event>.pt)
    - prediction: on the full 512x512 chip, water if the score is above 0

Because a Colab session can stop at any time, ONE held-out event is run at a
time. Each finished event saves its counts to data/processed/unet_counts/, and
"summary" combines all finished events into one table.

Run from the project folder (on Colab: put "!" in front):
    python src/step13_unet.py Bolivia     # one held-out event (start with this)
    python src/step13_unet.py all         # every event not finished yet
    python src/step13_unet.py summary     # table of all finished events

Saves:
    data/processed/unet_chips.npz        the 6 bands, labels and masks of all chips
                                         (made once, so later runs start faster)
    models/unet_<event>.pt               the best model for each held-out event
    data/processed/unet_counts/<event>.csv   counts for each finished event
    data/processed/unet_predictions/<chip>.npz   predicted map of every test chip
    outputs/unet_results.csv             from "summary" (numbered, never overwritten)

Needs a GPU: on Google Colab, Runtime -> Change runtime type -> T4 GPU.
Roughly 15-30 minutes per event on a T4. It also runs on a CPU, but very slowly.
"""
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from step03_features import BAND_INDEX, BAND_ORDER, REFLECTANCE_SCALE, list_chips, load_chip, valid_mask
from step04_baselines import PROCESSED_DIR, new_output_path
from step09_full_chip_scoring import count_errors, scores_from_counts

CACHE_PATH = PROCESSED_DIR / "unet_chips.npz"
COUNTS_DIR = PROCESSED_DIR / "unet_counts"
MAP_DIR = PROCESSED_DIR / "unet_predictions"
MODEL_DIR = Path("models")

BASE_CHANNELS = 32       # channels in the first layer; doubles at every level down
CROP_SIZE = 256          # training crop size (512 must be divisible by 16 for prediction)
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SHARE = 0.1   # share of the training chips used to pick the best epoch
SEED = 0

SCORED_ON = ["all_water", "flood_only"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_all_chips():
    """All hand-labelled chips as arrays. Built from the .tif files once, then cached.

    Returns a dict with:
        chip_ids : (N,) chip names
        events   : (N,) event names
        images   : (N, 6, 512, 512) int16, the 6 bands in BAND_ORDER
        labels   : (N, 512, 512) int8, 1 water, 0 land, -1 no data
        jrc      : (N, 512, 512) uint8, 1 = permanent water
        valid    : (N, 512, 512) bool, the same valid pixels as our other models
    """
    if CACHE_PATH.exists():
        print("Loading chips from", CACHE_PATH)
        cached = np.load(CACHE_PATH)
        data = {}
        for key in cached.files:
            data[key] = cached[key]
        return data

    print("Reading chips from the .tif files (only needed once)...")
    band_positions = []
    for name in BAND_ORDER:
        band_positions.append(BAND_INDEX[name])

    chip_ids = list_chips("hand")
    events = []
    images = []
    labels = []
    jrcs = []
    valids = []

    for i, chip_id in enumerate(chip_ids, start=1):
        s2, label, jrc = load_chip(chip_id, "hand")
        events.append(chip_id.split("_")[0])
        images.append(s2[band_positions].astype(np.int16))
        labels.append(label.astype(np.int8))
        jrcs.append(jrc.astype(np.uint8))
        valids.append(valid_mask(s2, label))
        if i % 50 == 0:
            print(f"  read {i}/{len(chip_ids)}")

    data = {
        "chip_ids": np.array(chip_ids),
        "events": np.array(events),
        "images": np.stack(images),
        "labels": np.stack(labels),
        "jrc": np.stack(jrcs),
        "valid": np.stack(valids),
    }
    np.savez(CACHE_PATH, **data)
    print("Saved chips to", CACHE_PATH)
    return data


def band_mean_std(data, rows):
    """Mean and std of each band over the valid pixels of the given chips."""
    n_bands = data["images"].shape[1]
    total = np.zeros(n_bands)
    total_squares = np.zeros(n_bands)
    count = 0

    for row in rows:
        valid = data["valid"][row]
        bands = data["images"][row][:, valid].astype(np.float64) / REFLECTANCE_SCALE
        total += bands.sum(axis=1)
        total_squares += (bands * bands).sum(axis=1)
        count += bands.shape[1]

    mean = total / count
    std = np.sqrt(total_squares / count - mean * mean)
    return mean.astype(np.float32), std.astype(np.float32)


def normalized_image(data, row, mean, std):
    """One chip's 6 bands, scaled and standardised, with invalid pixels set to 0."""
    image = data["images"][row].astype(np.float32) / REFLECTANCE_SCALE
    image = (image - mean[:, None, None]) / std[:, None, None]
    image[:, ~data["valid"][row]] = 0
    return image


def make_batch(data, rows, mean, std, rng, device):
    """Random crops (with random flips) of the given chips, as tensors on the device."""
    images = []
    targets = []
    masks = []

    for row in rows:
        image = normalized_image(data, row, mean, std)
        target = (data["labels"][row] == 1).astype(np.float32)
        mask = data["valid"][row].astype(np.float32)

        # Random crop
        size = image.shape[1]
        top = rng.integers(0, size - CROP_SIZE + 1)
        left = rng.integers(0, size - CROP_SIZE + 1)
        image = image[:, top:top + CROP_SIZE, left:left + CROP_SIZE]
        target = target[top:top + CROP_SIZE, left:left + CROP_SIZE]
        mask = mask[top:top + CROP_SIZE, left:left + CROP_SIZE]

        # Random flips (the same flip for image, target and mask)
        if rng.random() < 0.5:
            image = image[:, :, ::-1]
            target = target[:, ::-1]
            mask = mask[:, ::-1]
        if rng.random() < 0.5:
            image = image[:, ::-1, :]
            target = target[::-1, :]
            mask = mask[::-1, :]

        # Flipping makes the arrays "reversed views"; torch needs normal arrays
        images.append(np.ascontiguousarray(image))
        targets.append(np.ascontiguousarray(target))
        masks.append(np.ascontiguousarray(mask))

    images = torch.from_numpy(np.stack(images)).to(device)
    targets = torch.from_numpy(np.stack(targets)).unsqueeze(1).to(device)
    masks = torch.from_numpy(np.stack(masks)).unsqueeze(1).to(device)
    return images, targets, masks


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """A standard U-Net with 4 levels down and 4 levels up.

    Going down, the image gets smaller (512 -> 256 -> 128 -> 64 -> 32) and the
    number of channels grows, so the network sees more and more context.
    Going up, it grows back to full size. The "skip connections" join each
    level up with the same level down, so fine details are not lost.
    """

    def __init__(self, in_channels=6, base=BASE_CHANNELS):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.bottom = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, kernel_size=1)  # one water score per pixel

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        bottom = self.bottom(self.pool(d4))

        u4 = self.conv4(torch.cat([self.up4(bottom), d4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out(u1)


def masked_loss(scores, targets, masks):
    """Binary cross-entropy averaged over the valid pixels only (no class weight)."""
    loss_per_pixel = nn.functional.binary_cross_entropy_with_logits(
        scores, targets, reduction="none")
    total = (loss_per_pixel * masks).sum()
    return total / masks.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Training and testing
# ---------------------------------------------------------------------------

def predict_chip(model, data, row, mean, std, device):
    """Predict a full chip. Returns (512, 512) array: 1 water, 0 land."""
    image = normalized_image(data, row, mean, std)
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(image)
    water = (scores[0, 0] > 0).cpu().numpy().astype(np.int8)
    return water


def validation_iou(model, data, rows, mean, std, device):
    """IoU over the valid pixels of the validation chips."""
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for row in rows:
        water = predict_chip(model, data, row, mean, std, device)
        valid = data["valid"][row]
        y_true = data["labels"][row][valid]
        tp, fp, fn = count_errors(y_true, water[valid])
        total_tp += tp
        total_fp += fp
        total_fn += fn
    iou, f1, precision, recall = scores_from_counts(total_tp, total_fp, total_fn)
    return iou


def train_one_event(held_out, data, device):
    """Train a U-Net on every event except held_out. Returns model, mean and std."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # Training chips: other events, with at least one valid pixel
    chip_rows = []
    for row in range(len(data["chip_ids"])):
        if data["events"][row] != held_out and data["valid"][row].sum() > 0:
            chip_rows.append(row)
    chip_rows = rng.permutation(chip_rows)

    n_validation = int(len(chip_rows) * VALIDATION_SHARE)
    validation_rows = chip_rows[:n_validation]
    train_rows = chip_rows[n_validation:]
    print(f"    {len(train_rows)} training chips, {len(validation_rows)} validation chips")

    mean, std = band_mean_std(data, train_rows)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"unet_{held_out}.pt"
    best_iou = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        order = rng.permutation(train_rows)
        total_loss = 0
        n_batches = 0

        for first in range(0, len(order), BATCH_SIZE):
            batch_rows = order[first:first + BATCH_SIZE]
            images, targets, masks = make_batch(data, batch_rows, mean, std, rng, device)

            optimizer.zero_grad()
            scores = model(images)
            loss = masked_loss(scores, targets, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        iou = validation_iou(model, data, validation_rows, mean, std, device)
        note = ""
        if iou > best_iou:
            best_iou = iou
            best_state = copy.deepcopy(model.state_dict())
            # mean and std are saved as plain lists so torch.load can read them later
            torch.save({"model": best_state, "mean": mean.tolist(), "std": std.tolist()},
                       model_path)
            note = "  (best so far, saved)"

        seconds = time.time() - start
        print(f"    epoch {epoch:2d}  loss {total_loss / n_batches:.4f}  "
              f"validation IoU {iou:.3f}  {seconds:.0f} s{note}")

    model.load_state_dict(best_state)
    print(f"    best validation IoU {best_iou:.3f}, model saved to {model_path}")
    return model, mean, std


def test_one_event(held_out, model, mean, std, data, device):
    """Predict every chip of held_out and count errors. Returns counts[scored_on]."""
    model.eval()
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    counts = {}
    for scored_on in SCORED_ON:
        counts[scored_on] = [0, 0, 0]

    for row in range(len(data["chip_ids"])):
        if data["events"][row] != held_out:
            continue
        valid = data["valid"][row]
        if valid.sum() == 0:
            continue

        water = predict_chip(model, data, row, mean, std, device)
        y_true = data["labels"][row][valid]
        y_pred = water[valid]
        not_permanent = data["jrc"][row][valid] != 1

        pairs = {
            "all_water": (y_true, y_pred),
            "flood_only": (y_true[not_permanent], y_pred[not_permanent]),
        }
        for scored_on in SCORED_ON:
            y_true_part, y_pred_part = pairs[scored_on]
            tp, fp, fn = count_errors(y_true_part, y_pred_part)
            counts[scored_on][0] += tp
            counts[scored_on][1] += fp
            counts[scored_on][2] += fn

        # Save the full map: -1 = not valid, 0 = land, 1 = water
        full_map = np.full(valid.shape, -1, dtype=np.int8)
        full_map[valid] = y_pred
        chip_id = data["chip_ids"][row]
        np.savez_compressed(MAP_DIR / f"{chip_id}.npz",
                            label=data["labels"][row], valid=valid, unet=full_map)

    return counts


def save_event_counts(held_out, counts):
    """Save one event's counts so a later "summary" can combine all events."""
    COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(COUNTS_DIR / f"{held_out}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scored_on", "tp", "fp", "fn"])
        for scored_on in SCORED_ON:
            tp, fp, fn = counts[scored_on]
            writer.writerow([scored_on, tp, fp, fn])


def load_event_counts(held_out):
    """Read the counts saved by save_event_counts."""
    counts = {}
    with open(COUNTS_DIR / f"{held_out}.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip the header
        for scored_on, tp, fp, fn in reader:
            counts[scored_on] = [int(tp), int(fp), int(fn)]
    return counts


def run_event(held_out, data, device):
    """Train, test and save one held-out event."""
    print(f"\n{held_out}")
    start = time.time()
    model, mean, std = train_one_event(held_out, data, device)
    counts = test_one_event(held_out, model, mean, std, data, device)
    save_event_counts(held_out, counts)

    for scored_on in SCORED_ON:
        tp, fp, fn = counts[scored_on]
        iou, f1, precision, recall = scores_from_counts(tp, fp, fn)
        print(f"    {scored_on:<11} IoU {iou:.3f}  F1 {f1:.3f}  "
              f"precision {precision:.3f}  recall {recall:.3f}")
    print(f"    finished {held_out} in {(time.time() - start) / 60:.1f} min")


def summary(events):
    """Combine every finished event into one table (one row per event, plus ALL)."""
    finished = []
    for event_name in events:
        if (COUNTS_DIR / f"{event_name}.csv").exists():
            finished.append(event_name)
    if not finished:
        raise SystemExit("No finished events yet. Run an event first.")

    rows = []
    for scored_on in SCORED_ON:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for event_name in finished:
            tp, fp, fn = load_event_counts(event_name)[scored_on]
            total_tp += tp
            total_fp += fp
            total_fn += fn
            iou, f1, precision, recall = scores_from_counts(tp, fp, fn)
            rows.append(["unet", scored_on, event_name, iou, f1, precision, recall])

        iou, f1, precision, recall = scores_from_counts(total_tp, total_fp, total_fn)
        rows.append(["unet", scored_on, "ALL", iou, f1, precision, recall])

    print(f"Finished events ({len(finished)} of {len(events)}): {', '.join(finished)}")
    if len(finished) < len(events):
        print("Note: ALL only covers the finished events, so it is not yet comparable "
              "with the ALL rows of steps 9-11.")

    print("\nscored_on   event          IoU      F1  precision    recall")
    for row in rows:
        _, scored_on, event_name, iou, f1, precision, recall = row
        print(f"{scored_on:<11} {event_name:<10} {iou:7.3f} {f1:7.3f} {precision:10.3f} {recall:9.3f}")

    out_path = new_output_path("unet_results")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scored_on", "event", "iou", "f1", "precision", "recall"])
        writer.writerows(rows)
    print("\nSaved table to", out_path)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Use: an event name (e.g. Bolivia), all, or summary")
    choice = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cpu":
        print("Warning: no GPU found. Training will be very slow.")

    data = load_all_chips()
    events = sorted(set(data["events"]))

    if choice == "summary":
        summary(events)
        return

    if choice == "all":
        to_run = []
        for event_name in events:
            if (COUNTS_DIR / f"{event_name}.csv").exists():
                print(f"{event_name}: already finished, skipped")
            else:
                to_run.append(event_name)
    elif choice in events:
        to_run = [choice]
    else:
        raise SystemExit(f"Unknown event: {choice}. Use one of {events}, all, or summary")

    n_parameters = sum(p.numel() for p in UNet().parameters())
    print(f"U-Net with {n_parameters:,} weights")

    for event_name in to_run:
        run_event(event_name, data, device)

    print("\nRun 'python src/step13_unet.py summary' to combine the finished events.")


if __name__ == "__main__":
    main()
