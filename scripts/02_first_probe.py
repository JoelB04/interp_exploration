"""
Session 2: your first probe, end to end.

Question: is truth linearly decodable from the residual stream, at which layer,
and does the readout position you discovered in session 1 matter?

Deliverable: a table of AUROC by layer, for two probe types and two readout
positions. That is it. Resist scope creep.

Run from the repo root:  python scripts/02_first_probe.py
(DATA_DIR is relative to the working directory, so the root matters.)
"""

import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from acts import load, get_acts  # noqa: E402

DATA_DIR = "data"
MAX_EXAMPLES = 400  # you are on CPU. Raise this once you have a GPU.
BASE = "https://raw.githubusercontent.com/saprmarks/geometry_of_truth/main/datasets"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch(name: str) -> pd.DataFrame:
    """Marks & Tegmark's geometry_of_truth datasets."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        url = f"{BASE}/{name}.csv"
        print(f"downloading {name}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise SystemExit(
                f"Could not fetch {url} ({e}).\n"
                "Clone https://github.com/saprmarks/geometry_of_truth and copy "
                f"datasets/{name}.csv into ./data/ manually."
            )
    return pd.read_csv(path)


def prepare(name: str, seed: int = 0):
    df = fetch(name)
    print(f"\n{name}: {len(df)} rows, columns = {list(df.columns)}")
    print(df.head(3).to_string())

    assert "statement" in df.columns and "label" in df.columns, \
        "expected 'statement' and 'label' columns -- inspect the head above and adapt"

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df.iloc[:MAX_EXAMPLES]

    balance = df["label"].mean()
    print(f"  using {len(df)} rows, {balance:.1%} positive")
    # A probe on an imbalanced set can score well by learning the prior.
    # AUROC is robust to this, accuracy is not. Know which you are reporting.

    return df["statement"].tolist(), df["label"].to_numpy()


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def diff_of_means(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The field's default estimator, and the fix for your session-1 confound.

    Topic information lives in both class means and cancels in the subtraction,
    PROVIDED topics are balanced across classes. That proviso is doing real work
    and is exactly what broke your four-sentence similarity metric.
    """
    d = acts[labels == 1].mean(axis=0) - acts[labels == 0].mean(axis=0)
    return d / np.linalg.norm(d)


def eval_layer(train_acts, train_y, test_acts, test_y):
    """AUROC for both probe types at one layer."""
    d = diff_of_means(train_acts, train_y)
    auroc_dom = roc_auc_score(test_y, test_acts @ d)
    # No intercept needed: AUROC depends only on ranking, and a constant offset
    # does not change ranking. Worth convincing yourself of this.

    mu, sigma = train_acts.mean(0), train_acts.std(0) + 1e-8
    lr = LogisticRegression(max_iter=2000, C=0.1)
    lr.fit((train_acts - mu) / sigma, train_y)
    auroc_lr = roc_auc_score(test_y, lr.decision_function((test_acts - mu) / sigma))

    return auroc_dom, auroc_lr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model, tok, device = load()
    statements, labels = prepare("cities")

    n_train = int(0.7 * len(statements))
    # NOTE: this is a random row split. In cities.csv the same city often appears
    # in both a true and a false statement, so a random split can put both halves
    # of a pair across the train/test boundary. That is not classic label leakage
    # but it does make this an easier test than it looks. If the dataframe has a
    # city or entity column, try splitting on that instead and see if AUROC drops.
    # Finding out whether it does is a five-minute experiment and a real result.

    results = {}
    for mode in ["raw", "chat"]:
        print(f"\n=== readout mode: {mode} ===")
        acts = get_acts(statements, model, tok, device, mode=mode).numpy()
        n_layers = acts.shape[1] - 1

        rows = []
        for layer in range(n_layers + 1):
            a = acts[:, layer, :]
            dom, lr = eval_layer(a[:n_train], labels[:n_train], a[n_train:], labels[n_train:])
            rows.append((layer, dom, lr))
        results[mode] = rows

    print("\n" + "=" * 58)
    print(f"{'layer':>6} | {'raw dom':>8} {'raw lr':>8} | {'chat dom':>9} {'chat lr':>8}")
    print("-" * 58)
    for i in range(len(results["raw"])):
        l, rd, rl = results["raw"][i]
        _, cd, cl = results["chat"][i]
        print(f"{l:>6} | {rd:>8.3f} {rl:>8.3f} | {cd:>9.3f} {cl:>8.3f}")

    for mode in results:
        best = max(results[mode], key=lambda r: r[1])
        print(f"\nbest diff-of-means layer ({mode}): {best[0]} at AUROC {best[1]:.3f}")

    np.save("results_session2.npy", results, allow_pickle=True)


if __name__ == "__main__":
    main()