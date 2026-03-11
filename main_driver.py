import os
import random
import sys
import json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess import preprocess
from feature_extract import extract_features, get_feature_names
from feature_select import select_features
from explainability import run_pfi, run_shap, run_lime, run_pdp
from train_cnn_lstm import build_model

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Avoid Windows console encoding issues when filenames contain non-ASCII chars.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------- CONFIG ----------------
SEED = 42
TRAIN_MODEL = os.getenv("TRAIN_MODEL", "1") == "1"
MODEL_PATH = os.getenv("MODEL_PATH", "cnn_lstm_emg_paper28.h5")
PIPELINE_PATH = os.getenv(
    "PIPELINE_PATH",
    f"{os.path.splitext(MODEL_PATH)[0]}_pipeline.npz"
)
BASE_PATH = "data"
LABELS_ENV = os.getenv("CLASS_NAMES", "").strip()
LABELS = [x.strip() for x in LABELS_ENV.split(",") if x.strip()] if LABELS_ENV else []
EVAL_MODE = os.getenv("EVAL_MODE", "paper_reproduction")  # "paper_reproduction" or "strict"
FUSION_RULE = os.getenv("FUSION_RULE", "union")  # "intersection", "vote2", "union"
TOP_K = int(os.getenv("TOP_K", "15"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.1"))
USE_CALLBACKS = os.getenv("USE_CALLBACKS", "0") == "1"
RUN_EXPLAINABILITY = os.getenv("RUN_EXPLAINABILITY", "1") == "1"
NUM_TRIALS = int(os.getenv("NUM_TRIALS", "1"))
ENSEMBLE_TOP_N = int(os.getenv("ENSEMBLE_TOP_N", "1"))
PAPER_EXACT_MODE = os.getenv("PAPER_EXACT_MODE", "1") == "1"
PAPER_USE_ALL_FEATURES = os.getenv("PAPER_USE_ALL_FEATURES", "0") == "1"
PAPER_CLASS_COUNTS = [int(x) for x in os.getenv("PAPER_CLASS_COUNTS", "50,98,93").split(",")]
SMOTE_TARGET_PER_CLASS = int(os.getenv("SMOTE_TARGET_PER_CLASS", "200"))
# Paper protocol: SMOTE to 200 samples/class before 80:20 stratified split.
USE_SMOTE = os.getenv("USE_SMOTE", "1") == "1"
USE_DETERMINISTIC_SPLIT = os.getenv("USE_DETERMINISTIC_SPLIT", "1") == "1"
TEST_SPLIT = float(os.getenv("TEST_SPLIT", "0.2"))
SPLIT_SEED = int(os.getenv("SPLIT_SEED", "42"))
SMOTE_SCOPE = os.getenv("SMOTE_SCOPE", "train" if USE_DETERMINISTIC_SPLIT else "all")
USE_WINDOW_SEGMENTATION = os.getenv("USE_WINDOW_SEGMENTATION", "1") == "1"
WINDOW_SEC = float(os.getenv("WINDOW_SEC", "1.0"))
WINDOW_OVERLAP = float(os.getenv("WINDOW_OVERLAP", "0.0"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))

USE_FEATURE_CACHE = True
CACHE_DIR = "cache"
CACHE_VERSION = (
    f"paper28_exact_{EVAL_MODE}_{FUSION_RULE}_k{TOP_K}_smote{SMOTE_TARGET_PER_CLASS}"
    f"_win{WINDOW_SEC}_ov{WINDOW_OVERLAP}_seg{int(USE_WINDOW_SEGMENTATION)}"
)
CACHE_PATH = os.path.join(
    CACHE_DIR,
    f"feature_cache_{CACHE_VERSION}.npz"
)
RESULTS_DIR = "results"
REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")
CM_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
CURVES_PATH = os.path.join(RESULTS_DIR, "training_curves.png")
# ---------------------------------------


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def apply_distribution_normalization(X, feature_names):
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
    norm_meta = {}
    for idx, fname in enumerate(feature_names):
        col = Xn[:, idx:idx + 1]
        if fname in minmax_features:
            scaler = MinMaxScaler()
            Xn[:, idx:idx + 1] = scaler.fit_transform(col)
            norm_meta[idx] = ("minmax", scaler)
        elif fname in robust_features:
            scaler = RobustScaler()
            Xn[:, idx:idx + 1] = scaler.fit_transform(col)
            norm_meta[idx] = ("robust", scaler)
        else:
            scaler = StandardScaler()
            Xn[:, idx:idx + 1] = scaler.fit_transform(col)
            norm_meta[idx] = ("zscore", scaler)
    return Xn, norm_meta


def apply_saved_normalization(X, norm_meta):
    Xt = X.astype(np.float64).copy()
    for idx, (_, scaler) in norm_meta.items():
        Xt[:, idx:idx + 1] = scaler.transform(Xt[:, idx:idx + 1])
    return Xt


set_global_seed(SEED)


def segment_signal(x, fs, win_sec, overlap):
    if win_sec <= 0:
        return [x]
    win_len = int(round(win_sec * fs))
    if win_len <= 1 or len(x) < win_len:
        return []
    step = int(round(win_len * (1.0 - overlap)))
    if step <= 0:
        step = win_len
    segments = []
    for start in range(0, len(x) - win_len + 1, step):
        segments.append(x[start:start + win_len])
    return segments

top_level_dirs = sorted(
    [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
)
dataset_groups = []
labels_from_nested = set()
for group in top_level_dirs:
    group_path = os.path.join(BASE_PATH, group)
    child_dirs = sorted(
        [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
    )
    group_has_label_dirs = False
    for child in child_dirs:
        child_path = os.path.join(group_path, child)
        if any(f.endswith(".asc") for f in os.listdir(child_path)):
            group_has_label_dirs = True
            labels_from_nested.add(child)
    if group_has_label_dirs:
        dataset_groups.append(group)

USE_NESTED_GROUPS = len(dataset_groups) > 0
if not LABELS:
    if USE_NESTED_GROUPS:
        LABELS = sorted(labels_from_nested)
    else:
        LABELS = sorted(top_level_dirs)
if len(LABELS) < 2:
    raise ValueError(
        f"Need at least 2 class folders under '{BASE_PATH}'. Found: {LABELS}"
    )
print(f"Classes used for training: {LABELS}")
if USE_NESTED_GROUPS:
    print(f"Combining data from groups: {dataset_groups}")

label_encoder = LabelEncoder()
label_encoder.fit(LABELS)
class_names = list(label_encoder.classes_)
class_ids = np.arange(len(class_names))


# ---------------- DATA LOADING / CACHE ----------------
use_cache = USE_FEATURE_CACHE and os.path.exists(CACHE_PATH)
if use_cache:
    print(f"Loading cached features from: {CACHE_PATH}")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    if "labels" not in cache.files or "files" not in cache.files:
        print("Cache is legacy (missing labels/files). Rebuilding features from raw files.")
        use_cache = False
    else:
        cached_labels = cache["labels"].tolist()
        if list(cached_labels) != class_names:
            raise ValueError(
                f"Cache labels {cached_labels} do not match current labels {class_names}. "
                "Delete cache or change CLASS_NAMES."
            )
if use_cache:
    X = cache["X"]
    y = cache["y"]
    file_paths = cache["files"].tolist()
    if "base_files" in cache.files:
        base_paths = cache["base_files"].tolist()
    else:
        base_paths = list(file_paths)
else:
    X, y_labels, file_paths, base_paths = [], [], [], []
    class_sample_counter = {label: 0 for label in LABELS}
    skipped_files = []
    class_target_counts = {}
    if PAPER_EXACT_MODE:
        if len(PAPER_CLASS_COUNTS) != len(LABELS):
            raise ValueError(
                f"PAPER_CLASS_COUNTS must define {len(LABELS)} counts, got {len(PAPER_CLASS_COUNTS)}."
            )
        class_target_counts = {label: PAPER_CLASS_COUNTS[i] for i, label in enumerate(LABELS)}

    for label in LABELS:
        files = []
        if USE_NESTED_GROUPS:
            for group in dataset_groups:
                folder = os.path.join(BASE_PATH, group, label)
                if os.path.isdir(folder):
                    group_files = sorted(
                        [
                            os.path.join(folder, f)
                            for f in os.listdir(folder)
                            if f.endswith(".asc")
                        ]
                    )
                    files.extend(group_files)
        else:
            folder = os.path.join(BASE_PATH, label)
            files = sorted(
                [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.endswith(".asc")
                ]
            )

        target_count = class_target_counts.get(label) if PAPER_EXACT_MODE else None
        for i, file in enumerate(files):
            if target_count is not None and class_sample_counter[label] >= target_count:
                break
            print(f"{label} [{i+1}/{len(files)}] -> {os.path.basename(file)}")

            try:
                signal = np.loadtxt(file)
            except Exception as e:
                skipped_files.append((file, str(e)))
                print(f"Skipping unreadable file: {file} | {e}")
                continue
            signal = preprocess(signal)
            if USE_WINDOW_SEGMENTATION:
                segments = segment_signal(signal, 4096, WINDOW_SEC, WINDOW_OVERLAP)
                if not segments:
                    continue
                for seg_idx, seg in enumerate(segments):
                    feats = extract_features(seg)
                    X.append(feats)
                    y_labels.append(label)
                    file_paths.append(f"{file}::seg{seg_idx}")
                    base_paths.append(file)
                    class_sample_counter[label] += 1
            else:
                feats = extract_features(signal)
                X.append(feats)
                y_labels.append(label)
                file_paths.append(file)
                base_paths.append(file)
                class_sample_counter[label] += 1

    X = np.array(X)
    y = label_encoder.transform(y_labels)

    print("Samples per class before balancing:")
    for lbl in LABELS:
        print(f"{lbl}: {class_sample_counter[lbl]}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} unreadable files.")
        for fp, err in skipped_files[:5]:
            print(f" - {fp}: {err}")

    if USE_FEATURE_CACHE:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(
            CACHE_PATH,
            X=X,
            y=y,
            labels=np.array(class_names, dtype=object),
            files=np.array(file_paths, dtype=object),
            base_files=np.array(base_paths, dtype=object),
        )
        print(f"Saved feature cache to: {CACHE_PATH}")

print("Feature extraction completed.")


# ---------------- FEATURE SELECTION / SPLIT / SCALING ----------------
all_feature_names = get_feature_names()
if X.shape[1] != len(all_feature_names):
    raise ValueError(
        f"Feature count mismatch: extracted {X.shape[1]} vs named {len(all_feature_names)}."
    )

X_norm, norm_meta = apply_distribution_normalization(X, all_feature_names)

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

if PAPER_EXACT_MODE:
    if PAPER_USE_ALL_FEATURES:
        selected_mask = np.ones(len(all_feature_names), dtype=bool)
    else:
        selected_mask = np.array([name in paper_fsf_names for name in all_feature_names], dtype=bool)
else:
    _, selected_mask = select_features(
        X_norm, y, top_k=TOP_K, return_mask=True, fusion_rule=FUSION_RULE
    )

X_sel = X_norm[:, selected_mask]

if USE_DETERMINISTIC_SPLIT and SMOTE_SCOPE != "train":
    print("Deterministic file split forces SMOTE_SCOPE='train' to preserve file lists.")
    SMOTE_SCOPE = "train"

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

if USE_DETERMINISTIC_SPLIT:
    if USE_WINDOW_SEGMENTATION:
        train_idx, test_idx = _deterministic_group_split(y, base_paths, TEST_SPLIT, SPLIT_SEED)
    else:
        train_idx, test_idx = _deterministic_split_indices(y, TEST_SPLIT, SPLIT_SEED)
    split_meta = {
        "seed": SPLIT_SEED,
        "test_split": TEST_SPLIT,
        "train_files": [file_paths[i] for i in train_idx],
        "test_files": [file_paths[i] for i in test_idx],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    split_path = os.path.join(RESULTS_DIR, "split_files.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    print(f"Saved deterministic split list: {split_path}")

    Xtr_raw, ytr_raw = X_sel[train_idx], y[train_idx]
    Xte_raw, yte_raw = X_sel[test_idx], y[test_idx]
else:
    Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split(
        X_sel, y,
        test_size=TEST_SPLIT,
        stratify=y,
        random_state=SPLIT_SEED,
    )

if USE_SMOTE:
    smote = SMOTE(
        sampling_strategy={int(c): SMOTE_TARGET_PER_CLASS for c in class_ids},
        random_state=SEED,
        k_neighbors=5,
    )
    if SMOTE_SCOPE == "all" and not USE_DETERMINISTIC_SPLIT:
        X_bal, y_bal = smote.fit_resample(X_sel, y)
        Xtr_scaled, Xte_scaled, ytr, yte = train_test_split(
            X_bal, y_bal,
            test_size=TEST_SPLIT,
            stratify=y_bal,
            random_state=SPLIT_SEED,
        )
        print("Samples per class after SMOTE (before split):")
        for c in class_ids:
            print(f"{class_names[c]}: {int(np.sum(y_bal == c))}")
        print("Samples per class after 80/20 split (train):")
        for c in class_ids:
            print(f"{class_names[c]}: {int(np.sum(ytr == c))}")
        print("Samples per class after 80/20 split (test):")
        for c in class_ids:
            print(f"{class_names[c]}: {int(np.sum(yte == c))}")
        # Skip the generic print below in this branch.
        ytr_counts_printed = True
    else:
        Xtr_scaled, ytr = smote.fit_resample(Xtr_raw, ytr_raw)
        Xte_scaled, yte = Xte_raw, yte_raw
    if "ytr_counts_printed" not in locals():
        print("Samples per class after SMOTE balancing (train set):")
        for c in class_ids:
            print(f"{class_names[c]}: {int(np.sum(ytr == c))}")
else:
    Xtr_scaled, Xte_scaled, ytr, yte = Xtr_raw, Xte_raw, ytr_raw, yte_raw
    print("SMOTE disabled; using original class distribution.")

feature_names = [name for name, keep in zip(all_feature_names, selected_mask) if keep]
if PAPER_EXACT_MODE and PAPER_USE_ALL_FEATURES:
    print(f"Selected {len(feature_names)} features for training (paper mode: all features):")
else:
    print(f"Selected {len(feature_names)} features for training:")
print(feature_names)

Xtr = Xtr_scaled.reshape((Xtr_scaled.shape[0], Xtr_scaled.shape[1], 1))
Xte = Xte_scaled.reshape((Xte_scaled.shape[0], Xte_scaled.shape[1], 1))

# Save a deterministic inference bundle to avoid feature-pipeline drift in app/CLI.
norm_method = np.array([norm_meta[i][0] for i in range(len(all_feature_names))], dtype=object)
norm_a = np.zeros(len(all_feature_names), dtype=np.float64)
norm_b = np.ones(len(all_feature_names), dtype=np.float64)
for i in range(len(all_feature_names)):
    method, scaler_i = norm_meta[i]
    if method == "minmax":
        norm_a[i] = float(scaler_i.data_min_[0])
        norm_b[i] = float(scaler_i.data_max_[0])
    elif method == "robust":
        norm_a[i] = float(scaler_i.center_[0])
        norm_b[i] = float(scaler_i.scale_[0])
    else:
        norm_a[i] = float(scaler_i.mean_[0])
        norm_b[i] = float(scaler_i.scale_[0])

segments_per_file_meta = -1 if USE_WINDOW_SEGMENTATION else 1
np.savez_compressed(
    PIPELINE_PATH,
    selected_mask=selected_mask.astype(np.uint8),
    scaler_mean=np.zeros(np.sum(selected_mask), dtype=np.float64),
    scaler_scale=np.ones(np.sum(selected_mask), dtype=np.float64),
    norm_method=norm_method,
    norm_a=norm_a,
    norm_b=norm_b,
    labels=np.array(class_names, dtype=object),
    feature_names=np.array(feature_names, dtype=object),
    top_k=np.array([TOP_K], dtype=np.int32),
    fusion_rule=np.array([FUSION_RULE], dtype=object),
    eval_mode=np.array([EVAL_MODE], dtype=object),
    segments_per_file=np.array([segments_per_file_meta], dtype=np.int32),
)
print(f"Saved inference pipeline: {PIPELINE_PATH}")


# ---------------- MODEL TRAIN / LOAD ----------------
history = None
if TRAIN_MODEL:
    print("Training CNN-LSTM model for current feature pipeline...")
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(ytr),
        y=ytr
    )
    class_weights = {int(c): float(w) for c, w in zip(np.unique(ytr), class_weights_arr)}

    callbacks = []
    if USE_CALLBACKS:
        callbacks = [
            EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
            ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", mode="max", save_best_only=True),
        ]

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(Xtr_scaled, ytr), start=1):
        X_fold_tr = Xtr_scaled[tr_idx].reshape((-1, Xtr_scaled.shape[1], 1))
        X_fold_va = Xtr_scaled[va_idx].reshape((-1, Xtr_scaled.shape[1], 1))
        y_fold_tr = ytr[tr_idx]
        y_fold_va = ytr[va_idx]
        fold_model = build_model(Xtr.shape[1:], learning_rate=LEARNING_RATE)
        fold_model.fit(
            X_fold_tr,
            y_fold_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            class_weight=class_weights,
            verbose=0,
        )
        fold_prob = fold_model.predict(X_fold_va, verbose=0)
        fold_pred = np.argmax(fold_prob, axis=1)
        fold_acc = float(np.mean(fold_pred == y_fold_va))
        cv_scores.append(fold_acc)
        print(f"CV fold {fold_idx}/{CV_FOLDS} accuracy: {fold_acc:.4f}")
    print(f"CV mean accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    model = build_model(Xtr.shape[1:], learning_rate=LEARNING_RATE)
    history = model.fit(
        Xtr,
        ytr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
else:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Pretrained model not found at '{MODEL_PATH}'. "
            "Set TRAIN_MODEL=True once to create it."
        )

    print("Loading pre-trained CNN-LSTM model...")
    model = load_model(MODEL_PATH)

    model_input_len = int(model.input_shape[1])
    current_input_len = int(Xtr.shape[1])
    if model_input_len != current_input_len:
        raise ValueError(
            f"Pretrained model expects {model_input_len} features, but current pipeline has {current_input_len}. "
            "Set TRAIN_MODEL=True once to train a compatible model."
        )


# ---------------- EVALUATION OUTPUTS ----------------
os.makedirs(RESULTS_DIR, exist_ok=True)

y_prob = model.predict(Xte, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

report = classification_report(
    yte,
    y_pred,
    labels=class_ids,
    target_names=class_names,
    digits=4,
    zero_division=0,
)
print("\nClassification Report:\n")
print(report)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

# Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2.
y_onehot = np.zeros((yte.shape[0], len(class_ids)), dtype=np.float64)
y_onehot[np.arange(yte.shape[0]), yte] = 1.0
brier_score = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

# Paper-reported aggregate metrics for easier comparison.
acc = np.mean(y_pred == yte)
kappa = cohen_kappa_score(yte, y_pred)
mcc = matthews_corrcoef(yte, y_pred)
auc_ovo = roc_auc_score(yte, y_prob, multi_class="ovo")
metrics_summary = (
    f"\nAccuracy: {acc:.4f}\n"
    f"Cohen's Kappa: {kappa:.4f}\n"
    f"MCC: {mcc:.4f}\n"
    f"AUC (OvO): {auc_ovo:.4f}\n"
    f"Brier Score (multiclass): {brier_score:.6f}\n"
)
print(metrics_summary)
with open(REPORT_PATH, "a", encoding="utf-8") as f:
    f.write(metrics_summary)
print(f"Saved classification report: {REPORT_PATH}")

cm = confusion_matrix(yte, y_pred, labels=class_ids)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    cmap="Blues",
    ax=ax_cm,
    colorbar=False,
)
plt.title("Confusion Matrix")
plt.tight_layout()
fig_cm.savefig(CM_PATH, dpi=300)
plt.close(fig_cm)
print(f"Saved confusion matrix: {CM_PATH}")

if history is not None:
    fig_hist, ax_hist = plt.subplots(1, 2, figsize=(10, 4))

    ax_hist[0].plot(history.history.get("loss", []), label="Train Loss")
    ax_hist[0].plot(history.history.get("val_loss", []), label="Val Loss")
    ax_hist[0].set_title("Loss")
    ax_hist[0].set_xlabel("Epoch")
    ax_hist[0].set_ylabel("Loss")
    ax_hist[0].legend()

    ax_hist[1].plot(history.history.get("accuracy", []), label="Train Accuracy")
    ax_hist[1].plot(history.history.get("val_accuracy", []), label="Val Accuracy")
    ax_hist[1].set_title("Accuracy")
    ax_hist[1].set_xlabel("Epoch")
    ax_hist[1].set_ylabel("Accuracy")
    ax_hist[1].legend()

    plt.tight_layout()
    fig_hist.savefig(CURVES_PATH, dpi=300)
    plt.close(fig_hist)
    print(f"Saved training curves: {CURVES_PATH}")


# ---------------- EXPLAINABILITY ----------------
if RUN_EXPLAINABILITY:
    run_pfi(model, Xte, yte, feature_names)
    run_shap(model, Xte, feature_names)
    run_lime(model, Xte, feature_names)   # optional (slow)
    run_pdp(model, Xte, feature_names)



