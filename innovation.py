import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _safe_normalize(v):
    arr = np.asarray(v, dtype=float)
    s = float(np.sum(np.abs(arr)))
    if s <= 1e-12:
        return np.zeros_like(arr)
    return arr / s


def risk_level_from_mdri(mdri):
    if mdri < 35:
        return "Low"
    if mdri < 60:
        return "Moderate"
    if mdri < 80:
        return "High"
    return "Critical"


def compute_mdri(labels, probs, contrib):
    labels_l = [str(x).lower() for x in labels]
    healthy_idx = labels_l.index("healthy") if "healthy" in labels_l else 0

    pathology_prob = float(1.0 - probs[healthy_idx])
    contrib = np.asarray(contrib, dtype=float)
    contrib_mag = float(np.sum(np.maximum(contrib, 0.0)) / (np.sum(np.abs(contrib)) + 1e-12))

    mdri = 100.0 * (0.7 * pathology_prob + 0.3 * contrib_mag)
    mdri = float(np.clip(mdri, 0.0, 100.0))

    return {
        "mdri": mdri,
        "pathology_probability": pathology_prob,
        "contribution_severity": contrib_mag,
        "risk_level": risk_level_from_mdri(mdri),
    }


def build_counterfactual_suggestions(feature_names, x_scaled, contrib, max_items=5):
    x_scaled = np.asarray(x_scaled, dtype=float)
    contrib = np.asarray(contrib, dtype=float)
    strength = np.abs(contrib)
    if np.all(strength <= 1e-12):
        return []

    idx = np.argsort(strength)[::-1]
    suggestions = []
    for i in idx:
        if len(suggestions) >= max_items:
            break
        c = float(contrib[i])
        if c <= 0:
            continue
        x = float(x_scaled[i])
        delta = -0.25 * x if abs(x) >= 0.05 else -0.15
        expected_gain = min(25.0, 100.0 * abs(c) / (np.sum(np.abs(contrib)) + 1e-12))
        direction = "decrease" if delta < 0 else "increase"
        suggestions.append(
            {
                "feature": feature_names[i],
                "current_z": x,
                "recommended_delta_z": delta,
                "action": direction,
                "expected_risk_reduction_pct": expected_gain,
            }
        )
    return suggestions


def _wrap_text(text, width=95):
    words = str(text).split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def generate_medical_report_pdf(
    file_name,
    labels,
    probs,
    predicted_label,
    confidence,
    mdri_result,
    top_feature_df,
    counterfactual_suggestions,
):
    buf = io.BytesIO()
    probs = np.asarray(probs, dtype=float)
    conf_pct = float(confidence) * 100.0

    with PdfPages(buf) as pdf:
        fig1 = plt.figure(figsize=(8.27, 11.69))
        ax1 = fig1.add_subplot(111)
        ax1.axis("off")

        y = 0.98
        line = 0.03
        ax1.text(0.01, y, "EMG Clinical Decision Support Report", fontsize=16, fontweight="bold", va="top")
        y -= line * 1.3
        ax1.text(0.01, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, va="top")
        y -= line
        ax1.text(0.01, y, f"File: {file_name}", fontsize=9, va="top")

        y -= line * 1.4
        ax1.text(0.01, y, "Prediction Summary", fontsize=12, fontweight="bold", va="top")
        y -= line
        ax1.text(0.02, y, f"Predicted class: {predicted_label}", fontsize=10, va="top")
        y -= line
        ax1.text(0.02, y, f"Confidence: {conf_pct:.2f}%", fontsize=10, va="top")
        y -= line
        ax1.text(0.02, y, f"MDRI: {mdri_result['mdri']:.2f}/100 ({mdri_result['risk_level']} risk)", fontsize=10, va="top")

        y -= line * 1.4
        ax1.text(0.01, y, "Class Probabilities", fontsize=12, fontweight="bold", va="top")
        y -= line
        for lbl, p in zip(labels, probs):
            ax1.text(0.02, y, f"{lbl}: {100.0 * float(p):.2f}%", fontsize=10, va="top")
            y -= line

        y -= line * 0.6
        ax1.text(0.01, y, "Top Explainability Drivers", fontsize=12, fontweight="bold", va="top")
        y -= line
        if isinstance(top_feature_df, pd.DataFrame) and not top_feature_df.empty:
            show_n = min(8, len(top_feature_df))
            for i in range(show_n):
                row = top_feature_df.iloc[i]
                txt = f"{i+1}. {row['feature']} | contribution={float(row['contribution']):.4f} | {row['effect']}"
                for wrapped in _wrap_text(txt, width=95):
                    ax1.text(0.02, y, wrapped, fontsize=9, va="top")
                    y -= line * 0.92
        else:
            ax1.text(0.02, y, "No explainability rows available.", fontsize=9, va="top")
            y -= line

        y -= line * 0.5
        ax1.text(0.01, y, "Counterfactual Health Suggestions", fontsize=12, fontweight="bold", va="top")
        y -= line
        if counterfactual_suggestions:
            for i, item in enumerate(counterfactual_suggestions, start=1):
                txt = (
                    f"{i}. {item['feature']}: {item['action']} by "
                    f"{abs(float(item['recommended_delta_z'])):.2f} z-units "
                    f"(expected MDRI reduction ~{float(item['expected_risk_reduction_pct']):.1f}%)."
                )
                for wrapped in _wrap_text(txt, width=95):
                    ax1.text(0.02, y, wrapped, fontsize=9, va="top")
                    y -= line * 0.92
        else:
            ax1.text(0.02, y, "No counterfactual action needed for current risk profile.", fontsize=9, va="top")

        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = plt.figure(figsize=(11.69, 8.27))
        gs = fig2.add_gridspec(1, 2, width_ratios=[1, 1.2])
        ax_prob = fig2.add_subplot(gs[0, 0])
        ax_mdri = fig2.add_subplot(gs[0, 1])

        ax_prob.bar(labels, probs * 100.0, color=["#4C78A8", "#F58518", "#E45756"][:len(labels)])
        ax_prob.set_ylim(0, 100)
        ax_prob.set_ylabel("Probability (%)")
        ax_prob.set_title("Class Probability Distribution")
        ax_prob.tick_params(axis="x", rotation=20)

        mdri = float(mdri_result["mdri"])
        ax_mdri.barh(["MDRI"], [mdri], color="#E45756" if mdri >= 60 else "#F2CF5B" if mdri >= 35 else "#54A24B")
        ax_mdri.set_xlim(0, 100)
        ax_mdri.set_title("Muscle Degeneration Risk Index")
        ax_mdri.set_xlabel("Risk Score (0-100)")
        ax_mdri.text(mdri + 1, 0, f"{mdri:.2f}", va="center", fontsize=10)

        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()

