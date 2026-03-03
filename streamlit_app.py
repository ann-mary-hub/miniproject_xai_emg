import io
import os
import numpy as np
import pandas as pd
import streamlit as st
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
        return None
    files = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.endswith("_pipeline.npz")
    ]
    if not files:
        return None
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


def find_model_file(search_dir):
    if not os.path.isdir(search_dir):
        return None
    files = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.endswith(".h5")
    ]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def find_preferred_model(search_dir):
    # Prefer the legacy deltoid+brachii model for your current app use-case.
    preferred = os.path.join(search_dir, "cnn_lstm_disease_deltoid_brachii.h5")
    if os.path.exists(preferred):
        return preferred
    return find_model_file(search_dir)


def peek_default_segments(pipeline_path, fallback=1):
    try:
        if not pipeline_path or not os.path.exists(pipeline_path):
            return fallback
        pipe = np.load(pipeline_path, allow_pickle=True)
        if "segments_per_file" in pipe.files:
            return int(pipe["segments_per_file"][0])
    except Exception:
        pass
    return fallback


def peek_pipeline_feature_count(pipeline_path):
    try:
        if not pipeline_path or not os.path.exists(pipeline_path):
            return None
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


@st.cache_resource
def load_keras_model(model_path):
    return load_model(model_path)


@st.cache_data
def load_inference_pipeline(pipeline_path):
    pipe = np.load(pipeline_path, allow_pickle=True)
    required = ["selected_mask", "scaler_mean", "scaler_scale", "feature_names", "labels"]
    missing = [k for k in required if k not in pipe.files]
    if missing:
        raise ValueError(f"Invalid pipeline bundle. Missing keys: {missing}")

    selected_mask = pipe["selected_mask"].astype(bool)
    scaler_mean = pipe["scaler_mean"].astype(float)
    scaler_scale = pipe["scaler_scale"].astype(float)
    scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    selected_names = pipe["feature_names"].tolist()
    labels = pipe["labels"].tolist() if "labels" in pipe.files else DEFAULT_LABELS
    default_segments = int(pipe["segments_per_file"][0]) if "segments_per_file" in pipe.files else 1
    norm_method = pipe["norm_method"].tolist() if "norm_method" in pipe.files else None
    norm_a = pipe["norm_a"].astype(float) if "norm_a" in pipe.files else None
    norm_b = pipe["norm_b"].astype(float) if "norm_b" in pipe.files else None
    return (
        selected_mask,
        scaler_mean,
        scaler_scale,
        selected_names,
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


def local_occlusion_explanation(model, x_scaled, pred_idx, feature_names, top_n=8):
    base_prob = model.predict(x_scaled.reshape(1, -1, 1), verbose=0)[0, pred_idx]
    contrib = np.zeros(x_scaled.shape[0], dtype=float)

    for i in range(x_scaled.shape[0]):
        x_mut = x_scaled.copy()
        x_mut[i] = 0.0
        mut_prob = model.predict(x_mut.reshape(1, -1, 1), verbose=0)[0, pred_idx]
        contrib[i] = base_prob - mut_prob

    order = np.argsort(np.abs(contrib))[::-1][:top_n]
    rows = []
    for idx in order:
        rows.append(
            {
                "feature": feature_names[idx],
                "contribution": float(contrib[idx]),
                "effect": "supports prediction" if contrib[idx] >= 0 else "opposes prediction",
            }
        )
    return pd.DataFrame(rows), contrib


def parse_uploaded_asc(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return np.loadtxt(io.StringIO(text))


def main():
    st.set_page_config(page_title="EMG Prediction + Explanation", layout="wide")
    st.title("EMG File Prediction and Explanation")
    st.write("Upload an `.asc` EMG file to get class prediction and a human-readable reason report.")

    default_pipeline = find_pipeline_file(".")
    default_model = find_preferred_model(".")
    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model path", value=default_model or "cnn_lstm_emg_paper28.h5")
        pipeline_path = st.text_input("Pipeline path", value=default_pipeline or "")

    uploaded = st.file_uploader("Upload EMG .asc file", type=["asc"])
    if uploaded is None:
        st.info("Choose an `.asc` file to start analysis.")
        return

    if not model_path or not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return
    if not pipeline_path or not os.path.exists(pipeline_path):
        st.error("Pipeline file not found. Run `main_driver.py` first to generate *_pipeline.npz.")
        return

    try:
        model = load_keras_model(model_path)
        model_expected = int(model.input_shape[1])
        pipeline_feature_len = peek_pipeline_feature_count(pipeline_path)
        if pipeline_feature_len != model_expected:
            compatible = find_compatible_pipeline(
                model_expected_len=model_expected,
                search_dir=".",
                preferred_path=pipeline_path,
            )
            if compatible is None:
                st.error(
                    "No compatible pipeline found for this model. "
                    f"Model expects {model_expected} features, selected pipeline has "
                    f"{pipeline_feature_len if pipeline_feature_len is not None else 'unknown'}."
                )
                return
            pipeline_path = compatible

        (
            selected_mask,
            scaler_mean,
            scaler_scale,
            selected_names,
            labels,
            default_segments,
            norm_method,
            norm_a,
            norm_b,
        ) = load_inference_pipeline(pipeline_path=pipeline_path)
    except Exception as e:
        st.error(f"Failed to load model/pipeline: {e}")
        return

    expected = int(model.input_shape[1])
    if expected != len(selected_names):
        st.error(
            "Model and selected features are inconsistent. "
            f"Model expects {expected}, current pipeline gives {len(selected_names)}."
        )
        return

    # Always use the pipeline-defined segment count to keep inference consistent with training.
    segments_per_file = int(default_segments)
    st.sidebar.caption(f"Using pipeline segments_per_file = {segments_per_file}")

    try:
        raw_signal = parse_uploaded_asc(uploaded)
        pre_signal = preprocess(raw_signal)
        segments = split_signal_segments(pre_signal, int(segments_per_file))
        if not segments:
            st.error("No valid segments generated from input signal.")
            return

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
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")
        return

    probs_per_seg = model.predict(
        feats_scaled.reshape(feats_scaled.shape[0], feats_scaled.shape[1], 1), verbose=0
    )
    probs = probs_per_seg.mean(axis=0)
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Class", pred_label)
    c2.metric("Confidence", f"{probs[pred_idx] * 100:.2f}%")

    prob_df = pd.DataFrame({"Class": labels, "Probability": probs})
    st.bar_chart(prob_df.set_index("Class"))

    # Local explanation based on average feature vector over segments.
    x_explain = feats_scaled.mean(axis=0)
    local_df, contrib = local_occlusion_explanation(
        model=model,
        x_scaled=x_explain,
        pred_idx=pred_idx,
        feature_names=selected_names,
        top_n=8,
    )

    st.subheader("Why This Prediction?")
    st.write(
        f"The model predicted **{pred_label}** because the following features had the strongest "
        "local influence for this specific file:"
    )
    st.dataframe(local_df, use_container_width=True)

    st.caption(
        "Positive contribution supports the predicted class; negative contribution opposes it."
    )

    with st.expander("Signal preview"):
        st.line_chart(pd.DataFrame({"raw": raw_signal[:3000]}))
        st.line_chart(pd.DataFrame({"preprocessed": pre_signal[:3000]}))


if __name__ == "__main__":
    main()
