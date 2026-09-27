"""Three-family LightGBM classifier fitted to confirmed positive pairs."""
import lightgbm as lgb
import numpy as np

FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]

PARAMS = dict(objective="multiclass", num_class=3, metric="multi_logloss",
              learning_rate=0.04, num_leaves=15, min_child_samples=15,
              feature_fraction=0.30, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, n_jobs=6, seed=42)

def full_fit(X, fam, names, n_rounds):
    m = np.isin(fam, FAMS)
    yy = np.array([FAMS.index(f) for f in fam[m]])
    d = lgb.Dataset(X[m], yy, feature_name=names)
    return lgb.train(PARAMS, d, num_boost_round=n_rounds)
