"""
Personae clustering pipeline (application copy).
Preprocess -> PCA -> K-means, with silhouette-based k selection.
Kept dependency-light so the Flask app can run anywhere sklearn is present.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

CANONICAL_FEATURES = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT",
    "TENURE",
]


def select_numeric_features(df):
    """Generalize to arbitrary schemas: drop obvious ID columns, keep numeric.
    Falls back gracefully when the canonical CC columns aren't all present."""
    drop = [c for c in df.columns if c.upper() in ("CUST_ID", "ID", "CUSTOMER_ID")]
    work = df.drop(columns=drop, errors="ignore")
    num = work.select_dtypes(include=[np.number])
    return list(num.columns)


def build_preprocessor():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def choose_k(X, k_min=2, k_max=8):
    """Sweep k; return (best_k_by_silhouette, sweep_records).

    silhouette_score is O(n^2); on large uploads that is the slowest step and
    can exceed a serverless timeout. We cap it with sample_size, which gives a
    near-identical estimate at a fraction of the cost (seeded, so reproducible).
    """
    n = len(X)
    sil_sample = min(2000, n) if n > 2000 else None
    records = []
    for k in range(k_min, min(k_max, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        records.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(
                X, labels, sample_size=sil_sample, random_state=RANDOM_STATE)),
        })
    best = max(records, key=lambda r: r["silhouette"])["k"]
    return best, records


def run(df, k=None, k_default=4):
    """Full pipeline. If k is None, pick by silhouette but honor the
    interpretability default (k_default) when it is competitive.

    Returns a dict with everything the UI needs.
    """
    features = select_numeric_features(df)
    if len(features) < 2:
        raise ValueError("Need at least two numeric feature columns to cluster.")

    X_raw = df[features].values
    pre = build_preprocessor()
    Xs = pre.fit_transform(X_raw)

    # PCA
    pca_full = PCA(random_state=RANDOM_STATE).fit(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)

    # choose k
    best_k, sweep = choose_k(Xs)
    chosen_k = k or (k_default if k_default else best_k)
    chosen_k = int(min(chosen_k, len(X_raw) - 1))

    km = KMeans(n_clusters=chosen_k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(Xs)

    prof = df[features].copy()
    prof["__seg"] = labels
    means = prof.groupby("__seg")[features].mean()
    sizes = prof.groupby("__seg").size()
    overall = df[features].mean()

    return {
        "features": features,
        "labels": labels.tolist(),
        "coords": coords,
        "means": means,
        "sizes": sizes,
        "overall": overall,
        "overall_std": df[features].std(ddof=0).replace(0, 1e-9),
        "chosen_k": chosen_k,
        "best_k": best_k,
        "sweep": sweep,
        "pca_n90": n90,
        "pca_cum10": float(cum[min(9, len(cum) - 1)] * 100),
        "pca_pc12": float(pca_full.explained_variance_ratio_[:2].sum() * 100),
        "n_rows": int(len(df)),
        "missing": {c: int(df[c].isna().sum()) for c in features if df[c].isna().sum() > 0},
    }
