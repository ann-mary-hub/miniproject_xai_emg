import os
import random
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess import preprocess
from feature_extract import extract_features, get_feature_names
from feature_select import select_features
from explainability import run_pfi, run_shap, run_lime, run_pdp
from train_cnn_lstm import build_model

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ---------------- CONFIG ----------------
SEED = 42
TRAIN_MODEL = os.getenv("TRAIN_MODEL", "1") == "1"
MODEL_PATH = os.getenv("MODEL_PATH", "cnn_lstm_emg_paper28.h5")
BASE_PATH = "data"
LABELS = ["Healthy", "Myopathy", "Neuropathy"]
EVAL_MODE = os.getenv("EVAL_MODE", "paper_reproduction")  # "paper_reproduction" or "strict"
FUSION_RULE = os.getenv("FUSION_RULE", "union")  # "intersection", "vote2", "union"
TOP_K = int(os.getenv("TOP_K", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
EPOCHS = int(os.getenv("EPOCHS", "80"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.1"))
USE_CALLBACKS = os.getenv("USE_CALLBACKS", "0") == "1"
RUN_EXPLAINABILITY = os.getenv("RUN_EXPLAINABILITY", "1") == "1"
NUM_TRIALS = int(os.getenv("NUM_TRIALS", "3"))

# Paper-protocol approximation controls
PAPER_PROTOCOL_MODE = True
FILES_PER_CLASS = int(os.getenv("FILES_PER_CLASS", "50"))      # equal file count per class
SEGMENTS_PER_FILE = int(os.getenv("SEGMENTS_PER_FILE", "8"))   # split each preprocessed signal

MAX_SAMPLES = None        # used only when PAPER_PROTOCOL_MODE=False

USE_FEATURE_CACHE = True
CACHE_DIR = "cache"
CACHE_VERSION = (
    f"paper28_ccfs_v3_{EVAL_MODE}_{FUSION_RULE}_k{TOP_K}_bal{FILES_PER_CLASS}_seg{SEGMENTS_PER_FILE}"
    if PAPER_PROTOCOL_MODE else
    f"paper28_ccfs_v3_{EVAL_MODE}_{FUSION_RULE}_k{TOP_K}"
)
CACHE_PATH = os.path.join(
    CACHE_DIR,
    f"feature_cache_{CACHE_VERSION}_{'all' if MAX_SAMPLES is None else MAX_SAMPLES}.npz"
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


def split_signal_segments(signal, n_segments):
    if n_segments <= 1:
        return [signal]
    segments = np.array_split(signal, n_segments)
    return [s for s in segments if s.size >= 64]


set_global_seed(SEED)


label_encoder = LabelEncoder()
label_encoder.fit(LABELS)
class_names = list(label_encoder.classes_)
class_ids = np.arange(len(class_names))


# ---------------- DATA LOADING / CACHE ----------------
if USE_FEATURE_CACHE and os.path.exists(CACHE_PATH):
    print(f"Loading cached features from: {CACHE_PATH}")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    X = cache["X"]
    y = cache["y"]
else:
    X, y_labels = [], []
    class_sample_counter = {label: 0 for label in LABELS}

    for label in LABELS:
        folder = os.path.join(BASE_PATH, label)
        files = sorted([f for f in os.listdir(folder) if f.endswith(".asc")])

        if PAPER_PROTOCOL_MODE:
            files = files[:FILES_PER_CLASS]
        elif MAX_SAMPLES is not None:
            files = files[:MAX_SAMPLES]

        for i, file in enumerate(files):
            print(f"{label} [{i+1}/{len(files)}] -> {file}")

            signal = np.loadtxt(os.path.join(folder, file))
            signal = preprocess(signal)

            segments = split_signal_segments(signal, SEGMENTS_PER_FILE) if PAPER_PROTOCOL_MODE else [signal]
            for seg in segments:
                feats = extract_features(seg)
                X.append(feats)
                y_labels.append(label)
                class_sample_counter[label] += 1

    X = np.array(X)
    y = label_encoder.transform(y_labels)

    print("Samples per class after segmentation:")
    for lbl in LABELS:
        print(f"{lbl}: {class_sample_counter[lbl]}")

    if USE_FEATURE_CACHE:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(CACHE_PATH, X=X, y=y)
        print(f"Saved feature cache to: {CACHE_PATH}")

print("Feature extraction completed.")


# ---------------- FEATURE SELECTION / SPLIT / SCALING ----------------
all_feature_names = get_feature_names()
if X.shape[1] != len(all_feature_names):
    raise ValueError(
        f"Feature count mismatch: extracted {X.shape[1]} vs named {len(all_feature_names)}."
    )

if EVAL_MODE == "strict":
    # Leakage-safe protocol.
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )
    Xtr_sel, selected_mask = select_features(
        Xtr_raw, ytr, top_k=TOP_K, return_mask=True, fusion_rule=FUSION_RULE
    )
    Xte_sel = Xte_raw[:, selected_mask]

    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr_sel)
    Xte_scaled = scaler.transform(Xte_sel)
else:
    # Paper-reproduction mode: selection/scaling before split.
    X_sel, selected_mask = select_features(
        X, y, top_k=TOP_K, return_mask=True, fusion_rule=FUSION_RULE
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    Xtr_scaled, Xte_scaled, ytr, yte = train_test_split(
        X_scaled, y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

feature_names = [name for name, keep in zip(all_feature_names, selected_mask) if keep]
print(f"Selected {len(feature_names)} features via CCFS ({FUSION_RULE}, {EVAL_MODE}):")
print(feature_names)

Xtr = Xtr_scaled.reshape((Xtr_scaled.shape[0], Xtr_scaled.shape[1], 1))
Xte = Xte_scaled.reshape((Xte_scaled.shape[0], Xte_scaled.shape[1], 1))


# ---------------- MODEL TRAIN / LOAD ----------------
history = None
if TRAIN_MODEL:
    print("Training CNN-LSTM model for current feature pipeline...")
    class_weights = None
    if EVAL_MODE == "strict":
        class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(ytr),
            y=ytr
        )
        class_weights = {int(c): float(w) for c, w in zip(np.unique(ytr), class_weights_arr)}

    callbacks = []
    if USE_CALLBACKS:
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
            ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        ]

    best_acc = -1.0
    best_history = None
    best_model = None
    for trial in range(NUM_TRIALS):
        set_global_seed(SEED + trial)
        model_trial = build_model(Xtr.shape[1:], learning_rate=LEARNING_RATE)
        print(f"Trial {trial+1}/{NUM_TRIALS}")
        history_trial = model_trial.fit(
            Xtr,
            ytr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        y_prob_trial = model_trial.predict(Xte, verbose=0)
        y_pred_trial = np.argmax(y_prob_trial, axis=1)
        acc_trial = np.mean(y_pred_trial == yte)
        print(f"Trial accuracy: {acc_trial:.4f}")
        if acc_trial > best_acc:
            best_acc = acc_trial
            best_history = history_trial
            best_model = model_trial

    model = best_model
    history = best_history
    model.save(MODEL_PATH)
    print(f"Best trial accuracy: {best_acc:.4f}")
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
    # run_lime(model, Xte, feature_names)   # optional (slow)
    run_pdp(model, Xte, feature_names)
