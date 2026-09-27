"""LightGBM pair-model parameters and gain-based feature selection."""
import lightgbm as lgb
import numpy as np

PARAMS = dict(
    objective="binary", metric="average_precision", learning_rate=0.03,
    num_leaves=31, min_child_samples=40, feature_fraction=0.30,
    bagging_fraction=0.80, bagging_freq=1, lambda_l1=0.0, lambda_l2=5.0,
    max_depth=-1, verbose=-1, n_jobs=5, seed=42,
    # 63 statt 255 Bins: bei ~2500 Merkmalen dominiert das Binning die Laufzeit,
    # und die Aufloesung ist fuer diese Merkmale unkritisch.
    max_bin=63, min_data_in_bin=5,
)

def select_features(X, y, names, train_mask, k=900, seed=0, rounds=250, sub=60000,
                    n_jobs=None):
    """Merkmalsvorauswahl ueber den Gain eines schnellen Modells.

    Bei 372 Positiven und ueber 3.000 Merkmalen ist das keine Rechenoptimierung,
    sondern Regularisierung. WICHTIG: `train_mask` darf nur Trainingszeilen des
    jeweiligen Folds enthalten — eine Auswahl auf allen Daten liesse
    Validierungslabels ueber die Merkmalsmenge in ihre eigene Vorhersage
    zurueckfliessen und macht die OOF-Messung optimistisch.
    """
    rng = np.random.default_rng(seed)
    m = train_mask.copy()
    unl = np.flatnonzero(m & (y < 0))
    if len(unl) > sub:
        m[rng.choice(unl, len(unl) - sub, replace=False)] = False
    p = dict(PARAMS); p.update(feature_fraction=0.5, learning_rate=0.08, num_leaves=31, seed=seed)
    if n_jobs is not None:
        p["n_jobs"] = n_jobs
    # Wenn alle Zeilen Training sind, waere X[flatnonzero(m)] eine vollstaendige
    # zweite Kopie der Matrix — bei 123k x 4003 float32 sind das 2 GB, die hier
    # nichts bringen.
    Xs = X if m.all() else X[np.flatnonzero(m)]
    d = lgb.Dataset(Xs, (y[m] == 1).astype(int), feature_name=names)
    mdl = lgb.train(p, d, num_boost_round=rounds)
    del d
    gain = mdl.feature_importance("gain")
    keep = np.argsort(-gain)[:k]
    return np.sort(keep)
