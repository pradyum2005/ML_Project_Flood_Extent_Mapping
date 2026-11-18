# Flood Extent Mapping from Satellite Images Using Machine Learning

Pixel-level flood mapping on the **Sen1Floods11** dataset with classical machine learning.
Every pixel of a Sentinel-2 image is classified as **water** or **land** from 22 spectral and
neighbourhood features. Models are always tested on **flood events they have never seen**.

The project compares a non-ML index baseline (MNDWI) with logistic regression, Random Forest,
LightGBM and a shallow neural network (MLP), plus a **U-Net** trained from scratch that learns its
own spatial features from whole images. It also studies:

- training on **weak (automatically made) labels** instead of hand labels,
- the gap between scores on a **balanced sample** and on **every pixel of real chips**,
- **class imbalance** (sampling, class weights, decision threshold),
- **flood maps** with permanent water removed, **feature importance** and **error analysis**.

## Main findings

1. **ML beats the index baseline, but the ML models are close to each other.** On the balanced
   sample, every ML model scores IoU 0.814–0.822, against 0.760 for the MNDWI threshold.
2. **Balanced-sample scores are too optimistic.** The training/test sample is ~45% water, but real
   chips are ~9% water. Scored on every valid pixel, IoU drops to 0.60–0.71.
3. **Matching the real class balance is the biggest improvement in the project.** Training on a
   *natural* sample (real water share) with no class weight raises LightGBM's full-chip IoU from
   **0.679 to 0.844**, and logistic regression's from 0.625 to 0.798.
4. **Weak labels do not replace hand labels.** Training on weak labels only gives lower IoU, and
   adding weak labels to hand labels makes results worse.
5. **Flood events differ far more than models do.** Per-event IoU ranges from about 0.39
   (Pakistan) to 0.95 (Mekong).

## Data

[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11), Sentinel-2 (13 bands), 512 × 512
chips at 10 m resolution.

- **Hand-labelled set:** 446 chips over 11 flood events (Bolivia, Ghana, India, Mekong, Nigeria,
  Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA). Labels: 1 = water, 0 = land, -1 = no data.
  Overall 9.2% water, 77.2% land, 13.6% no data.
- **Weakly labelled set:** labels made by a Sentinel-2 index rule (`S2IndexLabelWeak`).
  2,000 chips are used (an equal number per event, chosen at random with seed 0). Colombia is left
  out because it has no hand labels and so cannot be tested on.
- **JRC permanent water** masks are used to separate flood water from rivers and lakes.

The data is not in this repository. Expected layout (run every script from the project folder):

```
data/
  HandLabeled/
    S2Hand/          <chip>_S2Hand.tif
    LabelHand/       <chip>_LabelHand.tif
    JRCWaterHand/    <chip>_JRCWaterHand.tif
  WeaklyLabeled/
    S2IndexLabelWeak/  <chip>_S2IndexLabelWeak.tif
    S2Weak/            <chip>_S2Weak.tif   (downloaded by step 7)
  processed/         made by the scripts
outputs/             made by the scripts
```

The data is in the public bucket `gs://sen1floods11` (also reachable at
`https://storage.googleapis.com/sen1floods11/`); see the Sen1Floods11 README for the folder paths.

## Method

**Features (22 per pixel)**
- 6 bands: blue (B2), green (B3), red (B4), NIR (B8), SWIR1 (B11), SWIR2 (B12)
- 4 indices: NDWI, MNDWI, NDVI, NDBI
- mean and variance of each band in a 5 × 5 window (12 features)

**Valid pixels:** the label is not -1 and the image is not empty (all bands 0) at that pixel.

**Training samples**
- *Balanced sample* (`samples.npz`): up to 1,000 water + 1,000 land valid pixels per chip
  (797,848 rows, 44.9% water).
- *Natural sample* (`samples_natural.npz`): up to 2,000 random valid pixels per chip, so the water
  share is realistic (879,892 rows, 10.6% water).
- *Weak sample* (`samples_weak.npz`): balanced sampling on the weak chips (3,965,713 rows).

**Evaluation: leave-one-event-out.** For each of the 11 events, train on the other 10 and test on
the held-out event. Metrics are for the water class: **IoU** (main metric), F1, precision, recall.
The "ALL" row adds up the counts of all 11 held-out events.

Scores are reported in two ways:
- **balanced sample** (steps 4–8): the held-out event's balanced sample. Good for comparing models,
  but not realistic.
- **full chips** (steps 9–11): **every valid pixel** of every held-out chip. These are the honest
  numbers.

## Pipeline

All scripts are in `src/` and are numbered in the order they were built. Results are never
overwritten: if a file already exists, a number is added (`..._results2.csv`, `3`, ...).

| Step | Script | What it does | Main output | Run time* |
|---|---|---|---|---|
| 1 | `step01_inspect_chip.py` | Band stats and a 3-panel picture of one chip | `outputs/<chip>.png` | seconds |
| 2 | `step02_summarize_data.py` | Per-event dataset summary | `outputs/data_summary.csv` | ~1 min |
| 3 | `step03_features.py` | Features + balanced sampling (`weak` option for the weak set) | `data/processed/samples.npz`, `samples_weak.npz` | minutes |
| 4 | `step04_baselines.py` | MNDWI > 0 threshold and logistic regression | `outputs/baseline_results.csv` | minutes |
| 5 | `step05_tree_models.py` | Random Forest and LightGBM | `outputs/tree_model_results.csv` | ~25 min |
| 6 | `step06_mlp.py` | Shallow MLP (64, 32 hidden units) | `outputs/mlp_results.csv` | ~20 min |
| 7 | `step07_download_weak.py` | Downloads 2,000 weak-label images | `data/WeaklyLabeled/S2Weak/` | depends on network |
| 8 | `step08_noisy_labels.py` | Train on weak / hand / both labels, test on hand labels | `outputs/noisy_label_results_<model>.csv` | minutes to hours |
| 9 | `step09_full_chip_scoring.py` | All 5 models scored on every valid pixel | `outputs/full_chip_results.csv` | ~1 hour |
| 10 | `step10_class_imbalance.py` | Balanced vs natural sample × class weight on/off × thresholds 0.3–0.9 | `outputs/class_imbalance_results_<model>.csv` | ~40 min (LightGBM), ~6 min (logistic regression) |
| 11 | `step11_flood_maps.py` | Best setup for LightGBM, Random Forest and MLP; flood maps with JRC permanent water removed | `outputs/flood_map_results.csv`, `outputs/flood_maps/` | ~1 hour |
| 12 | `step12_error_analysis.py` | Permutation importance + error analysis of step 11's maps | `outputs/permutation_importance.csv/.png`, `per_chip_errors.csv`, `error_breakdown.csv`, `error_feature_means.csv`, `outputs/worst_chips/` | ~10 min |
| 13 | `step13_unet.py` | U-Net trained from scratch on the same 6 bands, one held-out event per run, scored on full chips like step 11 | `outputs/unet_results.csv`, `models/unet_<event>.pt`, `data/processed/unet_predictions/` | ~15–30 min per event on a GPU (estimate); much slower on a CPU |

\*Measured or estimated on an 8-core laptop without a GPU, except step 13. Keep the computer from
sleeping during long runs: a sleeping laptop pauses the run.

Model settings: logistic regression with feature scaling; Random Forest with 100 trees and
`min_samples_leaf=10`; LightGBM with 200 trees and learning rate 0.1; MLP with hidden layers
(64, 32), batch size 1024, early stopping, at most 50 epochs. `random_state=0` everywhere. Steps
4–9 use `class_weight="balanced"` where the model supports it; steps 10–11 compare it with no weight.

### U-Net (step 13)

The per-pixel models only see a 5 × 5 window. The U-Net sees the whole chip: it shrinks the image
four times (512 → 32 pixels) to gather wide context, then grows it back to full size, with skip
connections that keep fine detail. It uses the **same 6 bands** as the other models, so the
comparison is *learned spatial context* vs *hand-made window features*.

- **Model:** standard U-Net, 4 levels, 32 channels in the first level (about 7.8 million weights),
  batch norm and ReLU.
- **Input:** the 6 bands / 10000, standardised with the training chips' mean and std; invalid
  pixels set to 0.
- **Loss:** binary cross-entropy on valid pixels only, **no class weight** (step 10 showed class
  weighting causes false alarms).
- **Training:** one random 256 × 256 crop of every training chip per epoch, random flips, Adam
  (learning rate 0.001), batch size 8, 50 epochs. 10% of the training chips are kept for
  validation, and the epoch with the best validation IoU is kept.
- **Testing:** the full 512 × 512 chips of the held-out event, scored on every valid pixel as
  `all_water` and `flood_only`, the same as step 11.

Each run handles one held-out event, because Colab sessions can stop at any time. Finished events
are saved and skipped by later runs, and `summary` combines them into one table.

## Results

### Balanced sample (steps 4–6), hand labels, all events

| Model | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| MNDWI threshold | 0.760 | 0.864 | 0.881 | 0.847 |
| Logistic regression | **0.822** | 0.902 | 0.897 | 0.907 |
| Random Forest | 0.814 | 0.898 | 0.930 | 0.868 |
| LightGBM | 0.818 | 0.900 | 0.918 | 0.883 |
| MLP | 0.814 | 0.898 | 0.923 | 0.873 |

### Noisy labels (step 8), balanced sample, always tested on hand labels

| Training data | Logistic regression IoU | LightGBM IoU | Random Forest IoU |
|---|---|---|---|
| hand labels only | 0.822 | 0.818 | 0.814 |
| weak only, 100 / 500 / 1,000 / 2,000 chips | 0.764 / 0.778 / 0.780 / 0.786 | – / – / – / 0.760 | – / – / – / 0.759 |
| hand + weak, 100 / 500 / 1,000 / 2,000 chips | 0.817 / 0.806 / 0.800 / 0.799 | not run | not run |

Models trained on weak labels have very high precision but low recall (LightGBM: 0.981 / 0.772),
because the weak rule misses water. The MLP was not run on weak data: it would take several hours
and the three other models already show the same pattern.

### Full chips (step 9), balanced sample + class weight, all events

| Model | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| MNDWI threshold | 0.596 | 0.747 | 0.609 | 0.964 |
| Logistic regression | 0.625 | 0.769 | 0.635 | 0.974 |
| Random Forest | **0.713** | 0.832 | 0.739 | 0.952 |
| LightGBM | 0.679 | 0.809 | 0.700 | 0.957 |
| MLP | 0.691 | 0.817 | 0.716 | 0.952 |

Recall stays high but precision falls: the models learned from ~45% water and call too much
water on real chips (4.6–24.1% water per event).

### Class imbalance (step 10), full chips, threshold 0.5

| Sampling | Class weight | LightGBM IoU | Logistic regression IoU |
|---|---|---|---|
| balanced | weighted (setup of steps 4–9) | 0.679 | 0.625 |
| balanced | none | 0.712 | 0.655 |
| natural | weighted | 0.776 | 0.712 |
| **natural** | **none** | **0.844** | **0.798** |

LightGBM, natural + no weight: precision 0.921, recall 0.910. For this setup the default
threshold of 0.5 is already about the best (logistic regression: 0.800 at 0.4), so no threshold
tuning on test data is needed. The old setups only catch up at thresholds of 0.8–0.9, because
their probabilities are biased towards water. The other thresholds should be read as a
sensitivity check, not as tuned results.

### Flood maps, feature importance and error analysis (steps 11–12)

*Results to be added after running steps 11 and 12.* Step 11 reports two scores per model:
`all_water` (every valid pixel, comparable with steps 9–10) and `flood_only` (JRC permanent water
left out, so only the water added by the flood is scored).

### U-Net (step 13)

*Results to be added after running step 13.* Its `all_water` and `flood_only` scores compare
directly with step 11 (same events, same pixels, same metrics). The `ALL` row is only comparable
once all 11 events have been run.

## Limitations

- **Spatial context is limited for the classical models.** They only see a 5 × 5 window; the
  U-Net (step 13) is the only model with wide spatial context, and it was not tuned.
- **Optical data only.** Sentinel-2 cannot see through clouds; Sentinel-1 radar, which is also in
  Sen1Floods11, was not used.
- **No hyperparameter tuning.** Default-style settings were used for every model.
- **Band order** (blue = file band index 1, ..., SWIR2 = index 12) is taken from the Sen1Floods11
  documentation and not checked against the files. The MNDWI baseline working well suggests it is
  right.
- **Weak and hand chips were not checked for overlap** (same place and date), which could leak test
  information into the noisy-label experiment.
- Six chips have no valid pixels and are skipped (5 in Ghana, 1 in Paraguay).

## Setup

```
python -m venv .myvenv
.myvenv\Scripts\activate          (Windows)   or   source .myvenv/bin/activate
pip install -r requirements.txt
```

Then run the steps in order from the project folder, for example:

```
python src/step03_features.py
python src/step10_class_imbalance.py lightgbm
python src/step11_flood_maps.py
python src/step12_error_analysis.py
```

Steps 4–10 need `data/processed/samples.npz` (step 3), and step 8 also needs `samples_weak.npz`.
Steps 10–12 build `data/processed/samples_natural.npz` themselves if it is missing. Step 12's
error analysis needs the maps saved by step 11. Step 13 only needs the hand-labelled data.

### Running the U-Net (step 13)

It runs on a CPU, but a GPU is strongly recommended. On **Google Colab**: put the project folder
(with `data/HandLabeled/`) in Google Drive, choose **Runtime → Change runtime type → T4 GPU**,
then run:

```python
from google.colab import drive
drive.mount("/content/drive")
```
```python
%cd /content/drive/MyDrive/ML_Project_Flood_Extent_Mapping
!pip install -q rasterio lightgbm
```
```python
!python src/step13_unet.py Bolivia     # one held-out event (start here)
!python src/step13_unet.py all         # every event not finished yet
!python src/step13_unet.py summary     # combine finished events into outputs/unet_results.csv
```

Working inside the Drive folder means the models, counts and maps survive the session ending. The
first run reads the `.tif` files once and caches them as `data/processed/unet_chips.npz`
(about 1.4 GB), so later runs start quickly.
