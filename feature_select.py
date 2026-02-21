import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier


def _fisher_scores(X, y):
    classes = np.unique(y)
    global_mean = np.mean(X, axis=0)

    between = np.zeros(X.shape[1], dtype=float)
    within = np.zeros(X.shape[1], dtype=float)

    for c in classes:
        Xc = X[y == c]
        n_c = Xc.shape[0]
        if n_c == 0:
            continue
        mean_c = np.mean(Xc, axis=0)
        var_c = np.var(Xc, axis=0)

        between += n_c * (mean_c - global_mean) ** 2
        within += n_c * var_c

    return between / (within + 1e-12)


def _topk_mask(scores, k):
    k = min(k, scores.shape[0])
    idx = np.argsort(scores)[::-1][:k]
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[idx] = True
    return mask


def select_features(X, y, top_k=15, return_mask=False, fusion_rule="intersection"):
    n_features = X.shape[1]
    top_k = min(top_k, n_features)

    # FIS: Fisher Score top-k
    fisher = _fisher_scores(X, y)
    fis_mask = _topk_mask(fisher, top_k)

    # IGS: Information Gain Score (mutual information) top-k
    igs = mutual_info_classif(X, y, random_state=42)
    igs_mask = _topk_mask(igs, top_k)

    # RFE with Gradient Boosting top-k
    gb = GradientBoostingClassifier(random_state=42)
    rfe_mask = RFE(gb, n_features_to_select=top_k).fit(X, y).support_

    if fusion_rule == "intersection":
        mask = fis_mask & igs_mask & rfe_mask
    elif fusion_rule == "vote2":
        votes = fis_mask.astype(int) + igs_mask.astype(int) + rfe_mask.astype(int)
        mask = votes >= 2
    elif fusion_rule == "union":
        mask = fis_mask | igs_mask | rfe_mask
    else:
        raise ValueError("fusion_rule must be one of: intersection, vote2, union")

    # Safety fallback in case the chosen fusion returns an empty set.
    if np.sum(mask) == 0:
        votes = fis_mask.astype(int) + igs_mask.astype(int) + rfe_mask.astype(int)
        mask = votes >= 2

    X_selected = X[:, mask]

    if return_mask:
        return X_selected, mask
    return X_selected
