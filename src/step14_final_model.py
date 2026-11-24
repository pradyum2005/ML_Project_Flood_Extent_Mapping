"""Train the final model on ALL 11 events and save it for submission.

Steps 10-11 picked the best setup: LightGBM trained on the natural sample
(real water share) with NO class weight. Leave-one-event-out scored it at
IoU 0.844 on every valid pixel (step 11). Those runs train one model per
held-out event and do not keep them. This step trains one model on every
event, which is the model you would use on a new image.

Saves:
    models/lightgbm_final.joblib   the trained model and the feature names

Use it:
    import joblib
    saved = joblib.load("models/lightgbm_final.joblib")
    model = saved["model"]
    # X: (n_pixels, 22) features made by step03_features.compute_features,
    #    columns in the order saved["feature_names"]
    water = model.predict(X)   # 1 = water, 0 = land (threshold 0.5)

Run from the project folder:
    python src/step14_final_model.py
Needs data/processed/samples_natural.npz (built from the chips if missing).
"""
import time
from pathlib import Path

import joblib
import numpy as np

from step03_features import FEATURE_NAMES, list_chips
from step10_class_imbalance import build_natural_sample
from step11_flood_maps import make_unweighted_lightgbm

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lightgbm_final.joblib"


def main():
    X, y, event = build_natural_sample(list_chips("hand"))
    events = sorted(set(event))
    print(f"Natural sample: {len(y)} rows, {100 * y.mean():.1f}% water, {len(events)} events")

    start = time.time()
    model = make_unweighted_lightgbm()
    model.fit(X, y)
    print(f"Trained LightGBM on all events in {time.time() - start:.0f} s")

    train_water = model.predict(X)
    tp = int(np.sum((train_water == 1) & (y == 1)))
    fp = int(np.sum((train_water == 1) & (y == 0)))
    fn = int(np.sum((train_water == 0) & (y == 1)))
    print(f"Training-sample IoU {tp / (tp + fp + fn):.3f} "
          "(optimistic: these pixels were trained on; the honest score is step 11's 0.844)")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "setup": "LightGBM, 200 trees, learning rate 0.1, natural sample, no class weight, "
                 "threshold 0.5",
        "trained_on_events": events,
        "leave_one_event_out_iou": 0.844,
    }, MODEL_PATH)
    print("Saved final model to", MODEL_PATH)


if __name__ == "__main__":
    main()
