# Session notes: Flood Extent Mapping project (handoff for the next Claude session)

Last updated: 2026-11-12. Final submission deadline: **2026-11-19**. Keep the last ~3 days (Nov 17-19) for the report and slides.

## About the user and how to work with them

- Pradyumna More, modelling lead on a 3-person university team. Python beginner.
- **Code style:** modular functions, but plain code: one step per line, descriptive names, no dense one-liners or clever comprehensions. f-string formatting is fine.
- **Workflow:** write the code, explain it, and don't run it unless they ask. They review first and tell you what to do next. They often ask line-by-line questions about syntax and ML concepts: answer simply, with small examples.
- Uses **Windows cmd** (not PowerShell). Project venv: `.myvenv` (`.myvenv\Scripts\python.exe`). Laptop has 8 CPU cores and no GPU.
- Results must never overwrite old results: every results CSV goes through `new_output_path()` in `step04_baselines.py`, which adds a number (`..._results2.csv`, `3`, ...).
- They asked to keep things unconfusing: weak data is used **only** in `step08_noisy_labels.py` (train on weak, test on hand). Don't add "train on weak, test on weak" options to the other scripts.

## Project summary

"Flood Extent Mapping from Satellite Images Using Machine Learning". Data: **Sen1Floods11**, Sentinel-2 (13 bands), 512x512 chips at 10 m.

- Hand-labelled set: 446 chips over 11 events (Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA). Labels: 1 water, 0 land, -1 no data.
- Weak set: S2IndexLabelWeak labels (from a Sentinel-2 index rule) + S2Weak images. Colombia is excluded on purpose (no hand labels, so it cannot be tested on).
- Per-pixel features (22): 6 bands (blue, green, red, nir, swir1, swir2), 4 indices (NDWI, MNDWI, NDVI, NDBI), 5x5 window mean and variance of each band.
- Sampling: up to 1,000 water + 1,000 land valid pixels per chip, seed 0. A valid pixel has label != -1 and is not all-zero in the image.
- Evaluation: **leave-one-event-out (LOEO)**: hold out one event, train on the other 10, test on it. Main metric IoU for water, plus F1, precision, recall.
- Planned parts: 5 models, noisy-label experiment, class-imbalance comparison, full-chip flood maps with permanent-water (JRC) removal, error analysis, permutation importance.

## Files (all in `src/`, run from the project folder)

| File | What it does | Status |
|---|---|---|
| `step01_inspect_chip.py` | Prints band stats and saves a 3-panel PNG of one chip to `outputs/` | Done, committed |
| `step02_summarize_data.py` | Per-event dataset summary -> `outputs/data_summary.csv` | Done, committed |
| `step03_features.py` | Features + sampling. `python src/step03_features.py` -> `data/processed/samples.npz`; `python src/step03_features.py weak` -> `data/processed/samples_weak.npz` | Done, both files exist. Weak option uncommitted |
| `step04_baselines.py` | MNDWI > 0 threshold + logistic regression (scaled, class_weight balanced), LOEO, hand data -> `outputs/baseline_results.csv`. Also holds shared helpers `metrics`, `results_by_event`, `new_output_path`, `SAMPLES_PATH`, `PROCESSED_DIR`, `OUTPUT_DIR` | Done, latest edits uncommitted |
| `step05_tree_models.py` | Random Forest (100 trees, min_samples_leaf 10) + LightGBM (200 trees, lr 0.1), both class_weight balanced, random_state 0, LOEO, hand data -> `outputs/tree_model_results.csv`. Holds helpers `load_data`, `predict_leave_one_event_out`, `print_table`, `save_table` | Done, uncommitted |
| `step06_mlp.py` | Shallow MLP (StandardScaler + hidden layers 64, 32, batch 1024, early stopping, max_iter 50), LOEO, hand data -> `outputs/mlp_results.csv` | Written; first run was queued (see below) |
| `step07_download_weak.py` | Picked 2,000 weak chips (equal per event, random inside each event, seed 0, no Colombia) and downloaded their S2Weak images | Done (all downloaded), uncommitted |
| `step08_noisy_labels.py` | Noisy-label experiment. Always tests on hand labels. Setups: hand_only / weak_only / both, weak chip counts in `WEAK_CHIP_COUNTS`. Held-out event's weak rows are never used for training. Model chosen on the command line: `logistic_regression`, `random_forest`, `lightgbm` (default), `mlp`. -> `outputs/noisy_label_results_<model>.csv` | Written and used. Current settings: `WEAK_CHIP_COUNTS = [2000]`, `WEAK_SETUPS = ["weak_only"]`, `RUN_HAND_ONLY = False` |

Git: last commit is "implemented tree models". Uncommitted: changes to `step04_baselines.py`, `step03_features.py`, `step05_tree_models.py`, `step08_noisy_labels.py`, and new `step07_download_weak.py`, `step06_mlp.py`. Suggest committing before long runs.

`requirements.txt` still lacks `rasterio`, `matplotlib` and `lightgbm` (they are installed in `.myvenv`).

## Data facts

- Hand: water 9.2%, land 77.2%, no data 13.6% of all pixels. 5 Ghana chips have no valid pixels. 148 chips have <1% water. 60,454 pixels are labelled but empty in the image (50,026 in Paraguay).
- `samples.npz`: 797,848 rows, 44.9% water. `samples_weak.npz`: 3,965,713 rows from ~2,000 weak chips.
- Band order is assumed from the Sen1Floods11 docs (B1..B12, blue=1, green=2, red=3, nir=7, swir1=11, swir2=12), not verified from the files.

## Results so far (LOEO, balanced sample, ALL row)

Hand-only training (from `baseline_results.csv`, `tree_model_results.csv`):

| Model | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| MNDWI threshold | 0.760 | 0.864 | 0.881 | 0.847 |
| Logistic regression | 0.822 | 0.902 | 0.897 | 0.907 |
| Random Forest | 0.814 | 0.898 | 0.930 | 0.868 |
| LightGBM | 0.818 | 0.900 | 0.918 | 0.883 |

Event difficulty (IoU 0.68 Pakistan to 0.92 Spain) is much bigger than model differences. Trees: higher precision, lower recall than logistic regression.

Noisy-label experiment (test on hand labels):

| Setup | Logistic regression IoU | LightGBM IoU |
|---|---|---|
| hand_only | 0.822 | 0.818 (from tree_model_results) |
| weak_only 100 / 500 / 1000 / 2000 chips | 0.764 / 0.778 / 0.780 / 0.786 | - / - / - / 0.760 |
| both 100 / 500 / 1000 / 2000 chips | 0.817 / 0.806 / 0.800 / 0.799 | not run |

Findings: weak-only is worse than hand-only and improves only slowly with more chips. Adding weak data to hand data *hurts* and gets worse as weak rows outnumber hand rows. Weak-trained models have high precision and low recall (weak labels miss water); LightGBM weak-only has precision 0.981, recall 0.772, and follows the weak rule's mistakes more than logistic regression does. Pakistan is hit hardest (LightGBM weak-only 0.435).

**Runs in progress when this note was written** (a queue on the user's laptop): Random Forest weak_only 2000 (was at 5/11 events, ~10 min per event), then `step06_mlp.py` on hand data, then `step08_noisy_labels.py mlp`. Check `outputs/` for `noisy_label_results_random_forest.csv`, `mlp_results.csv` and `noisy_label_results_mlp.csv`. If missing, they did not finish.

## Still to do (in priority order)

1. **Collect the queued results** (RF weak-only, MLP hand, MLP weak-only) and summarise them in one table with the results above.
2. **Class-imbalance comparison**: class weighting vs undersampling (part of the plan, not started).
3. **Full-chip scoring**: score on every valid pixel of each test chip, not only the balanced sample. Current scores overstate real performance (real water share is ~9%, not ~45%). Highest priority for honest numbers.
4. **Flood maps with permanent-water removal**: predict full chips, remove JRC permanent water to get flood extent, save maps as PNGs for the report.
5. **Error analysis** (Pakistan and Ghana are worst) and **permutation importance**.
6. **Report and slides** (Nov 17-19).

Optional / if time allows: per-chip row cap for weak data (e.g. 250 rows per chip) to make RF on weak data ~8x faster; check whether weak chips overlap the hand chips (same place and date), which could leak test information; add the `both` setup for LightGBM; add missing packages to `requirements.txt`; CNN/U-Net only as a stretch goal (no GPU; Colab would be needed).

## Caveats to keep repeating

- All scores so far come from the **balanced sample**, so they are fine for comparing models but are not real-world accuracy.
- Weak and hand chip overlap has **not** been checked.
- The other project scripts (`step04_baselines.py`, `step05_tree_models.py`, `step06_mlp.py`) are hand-data only by design.
