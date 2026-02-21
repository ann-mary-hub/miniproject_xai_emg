import io
import os
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

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


@st.cache_resource
def load_keras_model(model_path):
    return load_model(model_path)


@st.cache_data
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
    selected_names = [n for n, keep in zip(all_feature_names, selected_mask) if keep]
    return selected_mask, scaler, selected_names


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

    default_cache = find_cache_file("cache")
    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model path", value="cnn_lstm_emg_paper28.h5")
        cache_path = st.text_input("Cache path", value=default_cache or "")
        top_k = st.number_input("Top-K features", min_value=1, max_value=50, value=20, step=1)
        fusion_rule = st.selectbox("Fusion rule", options=["union", "vote2", "intersection"], index=0)
        segments_per_file = st.number_input("Segments per file", min_value=1, max_value=20, value=8, step=1)

    uploaded = st.file_uploader("Upload EMG .asc file", type=["asc"])
    if uploaded is None:
        st.info("Choose an `.asc` file to start analysis.")
        return

    if not model_path or not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return
    if not cache_path or not os.path.exists(cache_path):
        st.error("Cache file not found. Run `main_driver.py` first to generate feature cache.")
        return

    try:
        model = load_keras_model(model_path)
        selected_mask, scaler, selected_names = build_inference_pipeline(
            cache_path=cache_path,
            top_k=int(top_k),
            fusion_rule=fusion_rule,
        )
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

    try:
        raw_signal = parse_uploaded_asc(uploaded)
        pre_signal = preprocess(raw_signal)
        segments = split_signal_segments(pre_signal, int(segments_per_file))
        if not segments:
            st.error("No valid segments generated from input signal.")
            return

        feats = np.array([extract_features(seg) for seg in segments], dtype=float)
        feats_sel = feats[:, selected_mask]
        feats_scaled = scaler.transform(feats_sel)
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")
        return

    probs_per_seg = model.predict(
        feats_scaled.reshape(feats_scaled.shape[0], feats_scaled.shape[1], 1), verbose=0
    )
    probs = probs_per_seg.mean(axis=0)
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Class", pred_label)
    c2.metric("Confidence", f"{probs[pred_idx] * 100:.2f}%")

    prob_df = pd.DataFrame({"Class": LABELS, "Probability": probs})
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
