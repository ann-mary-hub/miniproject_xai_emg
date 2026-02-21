import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

from preprocess import preprocess
from feature_extract import extract_features, get_feature_names
from feature_select import select_features


LABELS = ["Healthy", "Myopathy", "Neuropathy"]


def split_signal_segments(signal, n_segments):
    if n_segments <= 1:
        return [signal]
    segments = np.array_split(signal, n_segments)
    return [s for s in segments if s.size >= 64]


def find_cache_file(cache_dir):
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    files = [
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if f.startswith("feature_cache_") and f.endswith(".npz")
    ]
    if not files:
        raise FileNotFoundError(
            "No feature cache found. Run main_driver.py once to generate cache."
        )
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def build_inference_pipeline(cache_path, top_k, fusion_rule):
    cache = np.load(cache_path, allow_pickle=True)
    X = cache["X"]
    y = cache["y"]

    all_feature_names = get_feature_names()
    if X.shape[1] != len(all_feature_names):
        raise ValueError(
            f"Feature count mismatch in cache: {X.shape[1]} vs {len(all_feature_names)} names."
        )

    X_sel, selected_mask = select_features(
        X, y, top_k=top_k, return_mask=True, fusion_rule=fusion_rule
    )
    scaler = StandardScaler().fit(X_sel)
    selected_feature_names = [
        n for n, keep in zip(all_feature_names, selected_mask) if keep
    ]
    return selected_mask, scaler, selected_feature_names


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
    cache_path,
    top_k,
    fusion_rule,
    segments_per_file,
):
    model = load_model(model_path)
    selected_mask, scaler, selected_feature_names = build_inference_pipeline(
        cache_path, top_k=top_k, fusion_rule=fusion_rule
    )

    signal = np.loadtxt(input_file)
    signal = preprocess(signal)
    segments = split_signal_segments(signal, segments_per_file)
    if not segments:
        raise ValueError("No valid segments produced from input signal.")

    feats = np.array([extract_features(seg) for seg in segments], dtype=float)
    feats_sel = feats[:, selected_mask]
    feats_scaled = scaler.transform(feats_sel)

    probs_segments = model.predict(
        feats_scaled.reshape(feats_scaled.shape[0], feats_scaled.shape[1], 1), verbose=0
    )
    probs = probs_segments.mean(axis=0)
    pred_idx = int(np.argmax(probs))

    label_encoder = LabelEncoder().fit(LABELS)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

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
        print(f"  - {LABELS[i]}: {p:.4f}")

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
    parser.add_argument("--model", default="cnn_lstm_emg_paper28.h5", help="Model path")
    parser.add_argument("--cache", default=None, help="Feature cache .npz path")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K for feature selection")
    parser.add_argument(
        "--fusion-rule",
        default="union",
        choices=["intersection", "vote2", "union"],
        help="Feature fusion rule",
    )
    parser.add_argument(
        "--segments-per-file",
        type=int,
        default=8,
        help="Segments used for prediction-time averaging",
    )
    args = parser.parse_args()

    cache_path = args.cache if args.cache else find_cache_file("cache")
    predict_file(
        input_file=args.input,
        model_path=args.model,
        cache_path=cache_path,
        top_k=args.top_k,
        fusion_rule=args.fusion_rule,
        segments_per_file=args.segments_per_file,
    )


if __name__ == "__main__":
    main()
