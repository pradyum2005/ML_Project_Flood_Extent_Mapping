"""Shallow neural network (MLP), tested one flood event at a time.

Same idea as baselines.py and tree_models.py: hold out one event, train on
the other 10, predict the held-out one. A neural network needs scaled
features, so a StandardScaler comes first (fitted on training data only).

Saves outputs/mlp_results.csv (numbered, so old results are never overwritten).

Run from the project folder:
    python src/mlp.py
"""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import new_output_path, results_by_event
from tree_models import load_data, predict_leave_one_event_out, print_table, save_table


def make_mlp():
    """A fresh scaled MLP with two small hidden layers.

    hidden_layer_sizes=(64, 32): 64 neurons, then 32 neurons.
    batch_size=1024: learn from 1024 rows at a time (faster than the default 200).
    early_stopping=True: keep 10% of the training rows aside and stop when the
        score on them stops improving, so it does not train longer than needed.
    max_iter=50: at most 50 passes over the training data.

    MLPClassifier has no class_weight option. Our samples are already about
    45% water, so that is fine here.
    """
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            batch_size=1024,
            early_stopping=True,
            max_iter=50,
            random_state=0,
        ),
    )
    return model


def main():
    X, y, event = load_data()
    print("Rows:", X.shape[0], " Features:", X.shape[1])

    print("\nMLP")
    mlp_prediction = predict_leave_one_event_out(make_mlp, X, y, event)
    all_rows = results_by_event(y, mlp_prediction, event, "mlp")

    print_table(all_rows)

    out_path = new_output_path("mlp_results")
    save_table(all_rows, out_path)
    print("\nSaved table to", out_path)


if __name__ == "__main__":
    main()
