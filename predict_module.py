import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import preprocess
from feature_extract import extract_features


DEFAULT_LABELS = ["Healthy", "Myopathy", "Neuropathy"]


def split_signal_segments(signal, n_segments):
    if n_segments <= 1:
        return [signal]
    segments = np.array_split(signal, n_segments)
    return [s for s in segments if s.size >= 64]


def find_pipeline_file(search_dir):
    if not os.path.isdir(search_dir):
        raise FileNotFoundError(f"Directory not found: {search_dir}")
    files = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.endswith("_pipeline.npz")
    ]
    if not files:
        raise FileNotFoundError(
            "No pipeline bundle found. Run main_driver.py once to generate *_pipeline.npz."
        )
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def find_pipeline_files(search_dir):
    if not os.path.isdir(search_dir):
        return []
    files = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.endswith("_pipeline.npz")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def peek_pipeline_feature_count(pipeline_path):
    try:
        pipe = np.load(pipeline_path, allow_pickle=True)
        if "feature_names" in pipe.files:
            return int(len(pipe["feature_names"]))
    except Exception:
        pass
    return None


def find_compatible_pipeline(model_expected_len, search_dir, preferred_path=None):
    candidates = []
    if preferred_path:
        candidates.append(preferred_path)
    candidates.extend([p for p in find_pipeline_files(search_dir) if p != preferred_path])
    for p in candidates:
        feat_len = peek_pipeline_feature_count(p)
        if feat_len == int(model_expected_len):
            return p
    return None


def load_inference_bundle(pipeline_path):
    pipe = np.load(pipeline_path, allow_pickle=True)
    required = ["selected_mask", "scaler_mean", "scaler_scale", "feature_names", "labels"]
    missing = [k for k in required if k not in pipe.files]
    if missing:
        raise ValueError(f"Invalid pipeline bundle '{pipeline_path}'. Missing keys: {missing}")

    selected_mask = pipe["selected_mask"].astype(bool)
    scaler_mean = pipe["scaler_mean"].astype(float)
    scaler_scale = pipe["scaler_scale"].astype(float)
    scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    feature_names = pipe["feature_names"].tolist()
    labels = pipe["labels"].tolist() if "labels" in pipe.files else DEFAULT_LABELS
    default_segments = (
        int(pipe["segments_per_file"][0])
        if "segments_per_file" in pipe.files
        else 1
    )
    norm_method = pipe["norm_method"].tolist() if "norm_method" in pipe.files else None
    norm_a = pipe["norm_a"].astype(float) if "norm_a" in pipe.files else None
    norm_b = pipe["norm_b"].astype(float) if "norm_b" in pipe.files else None
    return (
        selected_mask,
        scaler_mean,
        scaler_scale,
        feature_names,
        labels,
        default_segments,
        norm_method,
        norm_a,
        norm_b,
    )


def normalize_features(feats, scaler_mean, scaler_scale, norm_method=None, norm_a=None, norm_b=None):
    if norm_method is None or norm_a is None or norm_b is None:
        return (feats - scaler_mean) / scaler_scale

    feats_n = feats.astype(float).copy()
    for i, method in enumerate(norm_method):
        col = feats_n[:, i]
        if method == "minmax":
            denom = (norm_b[i] - norm_a[i]) if (norm_b[i] - norm_a[i]) != 0 else 1.0
            feats_n[:, i] = (col - norm_a[i]) / denom
        elif method == "robust":
            denom = norm_b[i] if norm_b[i] != 0 else 1.0
            feats_n[:, i] = (col - norm_a[i]) / denom
        else:
            denom = norm_b[i] if norm_b[i] != 0 else 1.0
            feats_n[:, i] = (col - norm_a[i]) / denom
    return feats_n


def transform_with_pipeline(
    feats,
    selected_mask,
    scaler_mean,
    scaler_scale,
    norm_method=None,
    norm_a=None,
    norm_b=None,
):
    """
    Backward/forward compatible feature transform.
    - New pipelines: per-feature normalization metadata for full feature vector.
    - Legacy pipelines: scaler stats over selected features only.
    """
    if norm_method is not None and norm_a is not None and norm_b is not None:
        feats_norm = normalize_features(feats, scaler_mean, scaler_scale, norm_method, norm_a, norm_b)
        return feats_norm[:, selected_mask]

    full_dim = feats.shape[1]
    scaler_dim = int(np.asarray(scaler_mean).shape[0])
    sel_dim = int(np.sum(selected_mask))

    if scaler_dim == full_dim:
        feats_norm = normalize_features(feats, scaler_mean, scaler_scale, None, None, None)
        return feats_norm[:, selected_mask]
    if scaler_dim == sel_dim:
        feats_sel = feats[:, selected_mask]
        return normalize_features(feats_sel, scaler_mean, scaler_scale, None, None, None)

    raise ValueError(
        f"Incompatible pipeline dimensions: full={full_dim}, selected={sel_dim}, scaler={scaler_dim}"
    )


def explain_prediction(model, x_scaled, pred_idx, feature_names, top_k=6):
    """
    Occlusion-style local explanation:
    For each feature, replace it with baseline (0 after standardization)
    and measure drop in predicted class probability.
    """
    base_prob = model.predict(x_scaled.reshape(1, -1, 1), verbose=0)[0, pred_idx]
    contrib = np.zeros(x_scaled.shape[0], dtype=float)

    for i in range(x_scaled.shape[0]):
        x_mut = x_scaled.copy()
        x_mut[i] = 0.0
        mut_prob = model.predict(x_mut.reshape(1, -1, 1), verbose=0)[0, pred_idx]
        contrib[i] = base_prob - mut_prob

    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    lines = []
    lines.append("Top reasons for this prediction:")
    for idx in order:
        direction = "supports" if contrib[idx] >= 0 else "opposes"
        lines.append(
            f"- {feature_names[idx]}: {contrib[idx]:+.4f} ({direction} predicted class)"
        )
    return lines


def predict_file(
    input_file,
    model_path,
    pipeline_path,
    segments_per_file=None,
):
    model = load_model(model_path)
    model_expected = int(model.input_shape[1])
    compatible_path = find_compatible_pipeline(
        model_expected_len=model_expected,
        search_dir=".",
        preferred_path=pipeline_path,
    )
    if compatible_path is None:
        raise ValueError(
            f"No compatible pipeline found for model '{model_path}' (expects {model_expected} features)."
        )
    if pipeline_path != compatible_path:
        print(f"[Info] Auto-switched pipeline to compatible bundle: {compatible_path}")
        pipeline_path = compatible_path

    (
        selected_mask,
        scaler_mean,
        scaler_scale,
        selected_feature_names,
        labels,
        default_segments,
        norm_method,
        norm_a,
        norm_b,
    ) = load_inference_bundle(pipeline_path)
    if segments_per_file is None:
        segments_per_file = default_segments

    signal = np.loadtxt(input_file)
    signal = preprocess(signal)
    segments = split_signal_segments(signal, segments_per_file)
    if not segments:
        raise ValueError("No valid segments produced from input signal.")

    feats = np.array([extract_features(seg) for seg in segments], dtype=float)
    feats_scaled = transform_with_pipeline(
        feats,
        selected_mask,
        scaler_mean,
        scaler_scale,
        norm_method,
        norm_a,
        norm_b,
    )

    probs_segments = model.predict(
        feats_scaled.reshape(feats_scaled.shape[0], feats_scaled.shape[1], 1), verbose=0
    )
    probs = probs_segments.mean(axis=0)
    pred_idx = int(np.argmax(probs))

    pred_label = labels[pred_idx]

    # Explain based on averaged standardized feature vector.
    x_explain = feats_scaled.mean(axis=0)
    reason_lines = explain_prediction(
        model=model,
        x_scaled=x_explain,
        pred_idx=pred_idx,
        feature_names=selected_feature_names,
        top_k=6,
    )

    print("\nPrediction Result")
    print(f"- File: {input_file}")
    print(f"- Predicted class: {pred_label}")
    print("- Class probabilities:")
    for i, p in enumerate(probs):
        print(f"  - {labels[i]}: {p:.4f}")

    print("\nHuman-readable explanation")
    print(
        "The model compared EMG feature patterns from this signal to learned class patterns. "
        "The following features had the strongest local effect on the predicted class:"
    )
    for line in reason_lines:
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Predict EMG class for one .asc file and explain why."
    )
    parser.add_argument("--input", required=True, help="Path to input .asc file")
    parser.add_argument("--model", default="cnn_lstm_disease_deltoid_brachii.h5", help="Model path")
    parser.add_argument("--pipeline", default=None, help="Inference pipeline .npz path")
    parser.add_argument(
        "--segments-per-file",
        type=int,
        default=None,
        help="Segments used for prediction-time averaging",
    )
    args = parser.parse_args()

    pipeline_path = args.pipeline if args.pipeline else find_pipeline_file(".")
    predict_file(
        input_file=args.input,
        model_path=args.model,
        pipeline_path=pipeline_path,
        segments_per_file=args.segments_per_file,
    )


if __name__ == "__main__":
    main()
