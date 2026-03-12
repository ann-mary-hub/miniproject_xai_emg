import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from feature_extract import get_feature_names
from feature_select import select_features
from innovation import compute_mdri


def apply_distribution_normalization(X, feature_names):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    minmax_features = {
        "Mean",
        "Variance",
        "Integrated EMG (IEMG)",
        "Total Power (TP)",
        "Signal Duration (SGD)",
        "Mean Frequency (MNF)",
        "Median Frequency (MDF)",
    }
    robust_features = {
        "Skewness",
        "Kurtosis",
        "Zero Crossings (ZC)",
        "Willison Amplitude (WAMP)",
        "Variance of Central Frequency (VCF)",
        "Max Wavelet Coefficient (MWC)",
    }

    Xn = X.astype(np.float64).copy()
    for idx, fname in enumerate(feature_names):
        col = Xn[:, idx:idx + 1]
        if fname in minmax_features:
            scaler = MinMaxScaler()
            Xn[:, idx:idx + 1] = scaler.fit_transform(col)
        elif fname in robust_features:
            scaler = RobustScaler()
            Xn[:, idx:idx + 1] = scaler.fit_transform(col)
        else:
            scaler = StandardScaler()
            Xn[:, idx:idx + 1] = scaler.fit_transform(col)
    return Xn


def find_latest_cache(cache_dir):
    if not os.path.isdir(cache_dir):
        return None
    files = [
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if f.startswith("feature_cache_") and f.endswith(".npz")
    ]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def _deterministic_split_indices(labels, test_size, seed):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_test = int(np.round(len(idx) * test_size))
        n_test = max(1, min(n_test, len(idx) - 1))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def _deterministic_group_split(labels, groups, test_size, seed):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        class_groups = np.unique(groups[idx])
        rng.shuffle(class_groups)
        n_test = int(np.round(len(class_groups) * test_size))
        n_test = max(1, min(n_test, len(class_groups) - 1))
        test_groups = set(class_groups[:n_test])
        for i in idx:
            if groups[i] in test_groups:
                test_idx.append(i)
            else:
                train_idx.append(i)
    return np.array(train_idx), np.array(test_idx)


def local_occlusion_contrib(model, x_scaled, pred_idx):
    base_prob = model.predict(x_scaled.reshape(1, -1, 1), verbose=0)[0, pred_idx]
    contrib = np.zeros(x_scaled.shape[0], dtype=float)
    for i in range(x_scaled.shape[0]):
        x_mut = x_scaled.copy()
        x_mut[i] = 0.0
        mut_prob = model.predict(x_mut.reshape(1, -1, 1), verbose=0)[0, pred_idx]
        contrib[i] = base_prob - mut_prob
    return contrib


def mdri_for_sample(model, labels, x_scaled):
    probs = model.predict(x_scaled.reshape(1, -1, 1), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    contrib = local_occlusion_contrib(model, x_scaled, pred_idx)
    return compute_mdri(labels=labels, probs=probs, contrib=contrib)


def sensitivity_analysis(model, labels, Xte, yte, feature_names, out_dir):
    max_per_class = int(os.getenv("MDRI_SENS_SAMPLES", "5"))
    delta = float(os.getenv("MDRI_SENS_DELTA", "0.5"))
    max_features = int(os.getenv("MDRI_SENS_FEATURES", "8"))

    class_names = ["Healthy", "Myopathy", "Neuropathy"]
    idx_by_class = {c: [] for c in class_names}
    for i, y in enumerate(yte):
        label = labels[int(y)]
        if label in idx_by_class and len(idx_by_class[label]) < max_per_class:
            idx_by_class[label].append(i)

    # Limit features to speed up sensitivity runs.
    use_features = feature_names[:max_features] if len(feature_names) > max_features else feature_names
    rows = []
    total_samples = sum(len(v) for v in idx_by_class.values())
    total_steps = total_samples * len(use_features) * 2
    step = 0
    print(f"MDRI sensitivity: {total_samples} samples, {len(use_features)} features, {total_steps} steps")
    for label in class_names:
        for i in idx_by_class.get(label, []):
            x = Xte[i].copy()
            base_mdri = mdri_for_sample(model, labels, x)["mdri"]
            for f_idx, fname in enumerate(use_features):
                for sign in (-1.0, 1.0):
                    step += 1
                    if step % 100 == 0 or step == total_steps:
                        print(f"MDRI sensitivity progress: {step}/{total_steps}")
                    x_mut = x.copy()
                    x_mut[f_idx] = x_mut[f_idx] + sign * delta
                    mdri_mut = mdri_for_sample(model, labels, x_mut)["mdri"]
                    rows.append(
                        {
                            "true_label": label,
                            "feature": fname,
                            "delta_sign": "plus" if sign > 0 else "minus",
                            "mdri_base": base_mdri,
                            "mdri_mut": mdri_mut,
                            "mdri_change": mdri_mut - base_mdri,
                        }
                    )

    out_csv = os.path.join(out_dir, "mdri_sensitivity.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("true_label,feature,delta_sign,mdri_base,mdri_mut,mdri_change\n")
        for r in rows:
            f.write(
                f"{r['true_label']},{r['feature']},{r['delta_sign']},"
                f"{r['mdri_base']:.4f},{r['mdri_mut']:.4f},{r['mdri_change']:.4f}\n"
            )

    # Aggregate: mean absolute change per feature per class.
    agg = {}
    for r in rows:
        key = (r["true_label"], r["feature"])
        agg.setdefault(key, []).append(abs(r["mdri_change"]))

    summary_rows = []
    for (label, feat), vals in agg.items():
        summary_rows.append(
            {"true_label": label, "feature": feat, "mean_abs_change": float(np.mean(vals))}
        )

    summary_csv = os.path.join(out_dir, "mdri_sensitivity_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("true_label,feature,mean_abs_change\n")
        for r in summary_rows:
            f.write(f"{r['true_label']},{r['feature']},{r['mean_abs_change']:.6f}\n")

    # Plot per class: top 10 features by mean abs change.
    for label in class_names:
        feats = [r for r in summary_rows if r["true_label"] == label]
        feats.sort(key=lambda x: x["mean_abs_change"], reverse=True)
        top = feats[:10]
        if not top:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([t["feature"] for t in top], [t["mean_abs_change"] for t in top], color="#F58518")
        ax.set_title(f"MDRI Sensitivity (Top Features) - {label}")
        ax.set_ylabel("Mean |ΔMDRI|")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"mdri_sensitivity_{label.lower()}.png")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

    print(f"Saved MDRI sensitivity CSV: {out_csv}")
    print(f"Saved MDRI sensitivity summary: {summary_csv}")
    print("Saved MDRI sensitivity plots per class.")


def main():
    cache_dir = "cache"
    model_path = os.getenv("MODEL_PATH", "cnn_lstm_emg_paper28.h5")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    test_split = float(os.getenv("TEST_SPLIT", "0.2"))
    split_seed = int(os.getenv("SPLIT_SEED", "42"))
    use_deterministic = os.getenv("USE_DETERMINISTIC_SPLIT", "0") == "1"
    use_smote = os.getenv("USE_SMOTE", "1") == "1"
    smote_scope = os.getenv("SMOTE_SCOPE", "all")
    paper_exact = os.getenv("PAPER_EXACT_MODE", "1") == "1"
    paper_use_all = os.getenv("PAPER_USE_ALL_FEATURES", "0") == "1"
    top_k = int(os.getenv("TOP_K", "15"))
    plot_kind = os.getenv("MDRI_PLOT", "box_scatter").lower()
    run_sensitivity = os.getenv("MDRI_SENSITIVITY", "1") == "1"

    cache_path = find_latest_cache(cache_dir)
    if not cache_path:
        raise FileNotFoundError("No feature cache found. Run main_driver.py first.")

    cache = np.load(cache_path, allow_pickle=True)
    X = cache["X"]
    y = cache["y"]
    labels = cache["labels"].tolist()
    file_paths = cache["files"].tolist() if "files" in cache.files else None
    base_paths = cache["base_files"].tolist() if "base_files" in cache.files else file_paths

    all_feature_names = get_feature_names()
    X_norm = apply_distribution_normalization(X, all_feature_names)

    paper_fsf_names = [
        "Standard Deviation (SD)",
        "Integrated EMG (IEMG)",
        "RMS Envelope (RMSE)",
        "Variance of Central Frequency (VCF)",
        "Motor Unit Action Potential (MUAP)",
        "Wavelet Entropy (WE)",
        "Max Wavelet Coefficient (MWC)",
        "Total Power (TP)",
        "Slope Sign Changes (SSC)",
        "Willison Amplitude (WAMP)",
        "Turn Count (TC)",
        "Lempel-Ziv Complexity (LZC)",
        "Hjorth Activity (HA)",
        "Hjorth Mobility (HM)",
        "Hjorth Complexity (HC)",
    ]

    if paper_exact:
        if paper_use_all:
            selected_mask = np.ones(len(all_feature_names), dtype=bool)
        else:
            selected_mask = np.array([name in paper_fsf_names for name in all_feature_names], dtype=bool)
    else:
        _, selected_mask = select_features(
            X_norm, y, top_k=top_k, return_mask=True, fusion_rule="union"
        )

    X_sel = X_norm[:, selected_mask]

    if use_deterministic:
        if base_paths is not None:
            train_idx, test_idx = _deterministic_group_split(y, base_paths, test_split, split_seed)
        else:
            train_idx, test_idx = _deterministic_split_indices(y, test_split, split_seed)
        Xtr_raw, ytr_raw = X_sel[train_idx], y[train_idx]
        Xte_raw, yte_raw = X_sel[test_idx], y[test_idx]
    else:
        from sklearn.model_selection import train_test_split
        Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split(
            X_sel, y,
            test_size=test_split,
            stratify=y,
            random_state=split_seed,
        )

    if use_smote:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(
            sampling_strategy={int(c): 200 for c in np.unique(y)},
            random_state=split_seed,
            k_neighbors=5,
        )
        if smote_scope == "all" and not use_deterministic:
            X_bal, y_bal = smote.fit_resample(X_sel, y)
            from sklearn.model_selection import train_test_split
            Xtr, Xte, ytr, yte = train_test_split(
                X_bal, y_bal,
                test_size=test_split,
                stratify=y_bal,
                random_state=split_seed,
            )
        else:
            Xtr, ytr = smote.fit_resample(Xtr_raw, ytr_raw)
            Xte, yte = Xte_raw, yte_raw
    else:
        Xtr, ytr = Xtr_raw, ytr_raw
        Xte, yte = Xte_raw, yte_raw

    model = load_model(model_path)
    mdri_rows = []
    for i in range(Xte.shape[0]):
        x = Xte[i]
        probs = model.predict(x.reshape(1, -1, 1), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        contrib = local_occlusion_contrib(model, x, pred_idx)
        mdri_result = compute_mdri(labels=labels, probs=probs, contrib=contrib)
        mdri_rows.append(
            {
                "mdri": mdri_result["mdri"],
                "true_label": labels[int(yte[i])],
                "pred_label": labels[pred_idx],
                "pathology_probability": mdri_result["pathology_probability"],
                "contribution_severity": mdri_result["contribution_severity"],
                "risk_level": mdri_result["risk_level"],
            }
        )
        if (i + 1) % 25 == 0 or (i + 1) == Xte.shape[0]:
            print(f"MDRI computed for {i + 1}/{Xte.shape[0]} samples")

    out_csv = os.path.join(results_dir, "mdri_distribution.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("mdri,true_label,pred_label,pathology_probability,contribution_severity,risk_level\n")
        for row in mdri_rows:
            f.write(
                f"{row['mdri']:.4f},{row['true_label']},{row['pred_label']},"
                f"{row['pathology_probability']:.6f},{row['contribution_severity']:.6f},{row['risk_level']}\n"
            )

    class_order = ["Healthy", "Myopathy", "Neuropathy"]
    data_by_class = {k: [] for k in class_order}
    for row in mdri_rows:
        if row["true_label"] in data_by_class:
            data_by_class[row["true_label"]].append(row["mdri"])

    if plot_kind == "hist":
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharex=True, sharey=True, constrained_layout=True)
        bins = np.linspace(0, 100, 21)
        for ax, cls in zip(axes, class_order):
            vals = np.array(data_by_class[cls], dtype=float)
            ax.hist(vals, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
            ax.set_title(cls, fontsize=11, pad=8)
            ax.set_xlabel("MDRI Score", fontsize=10)
            ax.grid(axis="y", alpha=0.2, linewidth=0.7)
            ax.set_xlim(0, 100)
        axes[0].set_ylabel("Count", fontsize=10)
        fig.suptitle("MDRI Distribution Across Classes", fontsize=13, y=1.06)
    else:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        if plot_kind in ("box", "box_scatter"):
            ax.boxplot(
                [data_by_class[c] for c in class_order],
                labels=class_order,
                showfliers=False,
            )
            if plot_kind == "box_scatter":
                rng = np.random.RandomState(42)
                for i, c in enumerate(class_order, start=1):
                    vals = np.array(data_by_class[c], dtype=float)
                    if vals.size == 0:
                        continue
                    jitter = rng.uniform(-0.12, 0.12, size=vals.size)
                    ax.scatter(
                        np.full_like(vals, i, dtype=float) + jitter,
                        vals,
                        s=12,
                        alpha=0.6,
                        color="#4C78A8",
                        edgecolors="none",
                    )
            ax.set_title("MDRI Distribution Across Classes", fontsize=12, pad=8)
            ax.set_ylabel("MDRI Score (0–100)", fontsize=10)
            ax.grid(axis="y", alpha=0.2, linewidth=0.7)
        else:
            parts = ax.violinplot(
                [data_by_class[c] for c in class_order],
                showmeans=True,
                showextrema=True,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("#4C78A8")
                pc.set_alpha(0.6)
            ax.set_xticks(np.arange(1, len(class_order) + 1))
            ax.set_xticklabels(class_order)
            ax.set_title("MDRI Distribution Across Classes", fontsize=12, pad=8)
            ax.set_ylabel("MDRI Score (0–100)", fontsize=10)
            ax.grid(axis="y", alpha=0.2, linewidth=0.7)

        fig.tight_layout()
    out_png = os.path.join(results_dir, f"mdri_distribution_{plot_kind}.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    summary_path = os.path.join(results_dir, "mdri_distribution_meta.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_path": cache_path,
                "model_path": model_path,
                "plot": plot_kind,
                "test_split": test_split,
                "split_seed": split_seed,
                "use_smote": use_smote,
                "smote_scope": smote_scope,
                "paper_exact_mode": paper_exact,
                "paper_use_all_features": paper_use_all,
                "top_k": top_k,
            },
            f,
            indent=2,
        )

    print(f"Saved MDRI CSV: {out_csv}")
    print(f"Saved MDRI plot: {out_png}")

    if run_sensitivity:
        sensitivity_analysis(model, labels, Xte, yte, feature_names=[n for n, k in zip(all_feature_names, selected_mask) if k], out_dir=results_dir)


if __name__ == "__main__":
    main()
