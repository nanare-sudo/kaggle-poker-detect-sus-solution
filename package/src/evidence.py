"""Evidence feature selection and LambdaRank parameters."""
import numpy as np

RANK_PARAMS = dict(
    objective="lambdarank", metric="map", eval_at=[5], lambdarank_truncation_level=30,
    learning_rate=0.05, num_leaves=31, min_child_samples=30, feature_fraction=0.35,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=4, seed=42,
)

EVID_DROP_PREFIXES = ("cov_",)

def evid_keep(names):
    """Spaltenmaske der PH-Rohmerkmale fuer den Evidenz-Ranker (vor `enrich`)."""
    return np.array([not n.startswith(EVID_DROP_PREFIXES) for n in names])
