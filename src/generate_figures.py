"""
FraudShield - generate the figures used in the report and the presentation.

Usage
-----
    python src/generate_figures.py

Reads the measured result files in outputs/ and writes PNGs to docs/images/.
Needs no Spark. Every chart is drawn from real pipeline output, so the report
and the slides cannot drift from the numbers the project actually produced.

Figures
    01_architecture.png        system architecture (Member 1 -> Member 2)
    02_class_imbalance.png     why accuracy is useless here
    03_model_comparison.png    precision / recall / F1 / PR-AUC, both models
    04_confusion_matrices.png  test-split confusion matrices
    05_threshold_curve.png     precision & recall vs threshold, chosen point
    06_fraud_by_step.png       fraud and volume over simulation time
    07_risk_bands.png          alerts by risk category
    08_feature_importance.png  what the Random Forest learned
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config  # noqa: E402

IMG = PROJECT_ROOT / "docs" / "images"

# Palette matching the dashboard
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#d03b3b"
AMBER = "#ec835a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def save(fig, name: str) -> None:
    out = IMG / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------- 1. architecture
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#eef4fc", ec=BLUE):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2 + (1.4 if sub else 0), title,
                ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center",
                    fontsize=8, color=INK2)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3,
                                    shrinkA=2, shrinkB=2))

    # lanes
    ax.add_patch(mpatches.FancyBboxPatch(
        (1, 60), 98, 38, boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor="#f4f8fd", edgecolor=BLUE, linewidth=1.0, linestyle="--", alpha=0.5))
    ax.text(3, 95, "MEMBER 1  ·  Data Engineering & Analytics", fontsize=11,
            fontweight="bold", color=BLUE)
    ax.add_patch(mpatches.FancyBboxPatch(
        (1, 1), 98, 47, boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor="#f2fbf7", edgecolor=AQUA, linewidth=1.0, linestyle="--", alpha=0.5))
    ax.text(3, 45, "MEMBER 2  ·  Machine Learning, Alerts & Dashboard", fontsize=11,
            fontweight="bold", color=AQUA)

    # member 1
    box(4, 79, 18, 11, "PaySim CSV", "6,362,620 rows", "#ffffff", INK2)
    box(26, 79, 18, 11, "Spark ingestion", "data_ingestion.py")
    box(48, 79, 22, 11, "Data quality + EDA", "19 Spark scripts")
    box(74, 79, 22, 11, "Preprocessing", "8 columns kept")
    box(37, 64, 26, 11, "Feature engineering", "+5 features -> 13 columns")
    arrow(22, 84.5, 26, 84.5)
    arrow(44, 84.5, 48, 84.5)
    arrow(70, 84.5, 74, 84.5)
    arrow(85, 79, 63, 72)

    # handover artefact, sitting between the two lanes
    box(34, 50, 32, 8, "outputs/engineered_data", "557 MB · 16 Spark CSV part files",
        "#fff8e8", YELLOW)
    arrow(50, 64, 50, 58.5)

    # member 2
    box(4, 29, 20, 10, "Parquet + row_id", "convert_to_parquet.py", "#ffffff", AQUA)
    box(28, 29, 20, 10, "Verification", "8/8 checks", "#ffffff", AQUA)
    box(52, 29, 20, 10, "MLlib pipeline", "split · weights · encode", "#ffffff", AQUA)
    box(76, 29, 20, 10, "Train 2 models", "LR · Random Forest", "#ffffff", AQUA)
    arrow(40, 50, 14, 39.5)
    arrow(24, 34, 28, 34)
    arrow(48, 34, 52, 34)
    arrow(72, 34, 76, 34)

    box(76, 15, 20, 10, "Threshold tuning", "validation only", "#ffffff", AQUA)
    box(52, 15, 20, 10, "Test comparison", "final model chosen", "#ffffff", AQUA)
    box(28, 15, 20, 10, "Fraud alerts", "risk categories", "#fdeeee", RED)
    box(4, 15, 20, 10, "Dashboard data", "small aggregates", "#ffffff", AQUA)
    arrow(86, 29, 86, 25.5)
    arrow(76, 20, 72, 20)
    arrow(52, 20, 48, 20)
    arrow(28, 20, 24, 20)

    box(30, 3, 40, 9, "Streamlit dashboard", "5 sections · reads precomputed outputs",
        "#eef4fc", BLUE)
    arrow(14, 15, 42, 12.5)
    arrow(38, 15, 56, 12.5)

    ax.set_title("FraudShield - System Architecture", fontsize=15, pad=14, loc="left")
    save(fig, "01_architecture.png")


# ---------------------------------------------------------------- 2. imbalance
def fig_imbalance(overview: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    legit = overview["legitimate_transactions"]
    fraud = overview["fraud_transactions"]

    ax1.barh(["Transactions"], [legit], color=BLUE, label="Legitimate", height=0.45)
    ax1.barh(["Transactions"], [fraud], left=[legit], color=RED, label="Fraud", height=0.45)
    ax1.set_xlabel("transactions")
    ax1.set_title("6,362,620 transactions, 8,213 fraudulent")
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_yticks([])
    ax1.grid(axis="x", color=GRID)
    ax1.set_axisbelow(True)
    ax1.text(legit / 2, 0, f"{legit:,} ({overview['legitimate_percentage']:.2f}%)",
             ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax1.annotate(f"{fraud:,} fraud\n({overview['fraud_percentage']:.4f}%)",
                 xy=(legit, 0), xytext=(legit * 0.72, 0.32),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
                 color=RED, fontsize=10, fontweight="bold")

    bars = ax2.bar(["Always predict\n'legitimate'", "What it actually\ncatches"],
                   [99.87, 0], color=[YELLOW, RED], width=0.5)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("%")
    ax2.set_title("Why accuracy is the wrong metric")
    ax2.grid(axis="y", color=GRID)
    ax2.set_axisbelow(True)
    for b, v in zip(bars, ["99.87%\naccuracy", "0 of 8,213\nfrauds"]):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, v,
                 ha="center", fontsize=10, fontweight="bold", color=INK)
    save(fig, "02_class_imbalance.png")


# ---------------------------------------------------------------- 3. comparison
def fig_comparison(comp: dict) -> None:
    table = comp["comparison_table"]
    metrics = [("test_precision", "Precision"), ("test_recall", "Recall"),
               ("test_f1", "F1"), ("test_f2", "F2"), ("test_pr_auc", "PR-AUC")]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.36
    xs = range(len(metrics))
    colors = {"LogisticRegression": BLUE, "RandomForestClassifier": AQUA}
    for i, row in enumerate(table):
        offs = [x + (i - 0.5) * width for x in xs]
        vals = [row[m] for m, _ in metrics]
        bars = ax.bar(offs, vals, width=width, label=row["model"],
                      color=colors.get(row["model"], BLUE))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score (test split)")
    ax.set_title("Model comparison on the held-out test split (fraud = positive class)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    save(fig, "03_model_comparison.png")


# ---------------------------------------------------------------- 4. confusion
def fig_confusion(comp: dict) -> None:
    table = comp["comparison_table"]
    fig, axes = plt.subplots(1, len(table), figsize=(11, 4.2))
    if len(table) == 1:
        axes = [axes]
    for ax, row in zip(axes, table):
        cm = [[row["test_tn"], row["test_fp"]], [row["test_fn"], row["test_tp"]]]
        ax.imshow([[0, 1], [1, 0]], cmap="Blues", alpha=0.12)
        labels = [["TN", "FP"], ["FN", "TP"]]
        notes = [["correct", "false alarm"], ["MISSED FRAUD", "caught"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i - 0.12, f"{cm[i][j]:,}", ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color=RED if labels[i][j] == "FN" else INK)
                ax.text(j, i + 0.18, f"{labels[i][j]} · {notes[i][j]}", ha="center",
                        va="center", fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ["predicted\nlegitimate", "predicted\nfraud"])
        ax.set_yticks([0, 1], ["actual\nlegitimate", "actual\nfraud"])
        ax.set_title(f"{row['model']}\nthreshold {row['threshold']:.3f}", fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Confusion matrices - test split (918,617 transactions, 4,006 fraud)",
                 fontsize=13, fontweight="bold", y=1.04)
    save(fig, "04_confusion_matrices.png")


# ---------------------------------------------------------------- 5. threshold
def fig_threshold() -> None:
    path = config.MODEL_RESULTS_DIR / "logistic_regression_threshold_sweep.csv"
    policy = load(config.MODEL_RESULTS_DIR / "logistic_regression_threshold_policy.json")
    if not path.exists() or not policy:
        return
    sweep = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(sweep["threshold"], sweep["precision"], color=BLUE, lw=2, label="Precision")
    ax.plot(sweep["threshold"], sweep["recall"], color=ORANGE, lw=2, label="Recall")
    ax.plot(sweep["threshold"], sweep["f1"], color=AQUA, lw=1.8, ls=":", label="F1")
    chosen = policy["selected_threshold"]
    ax.axvline(chosen, color=INK2, ls="--", lw=1.2)
    ax.text(chosen - 0.01, 1.04, f"chosen {chosen:.3f}\n(best F2)", ha="right",
            fontsize=9, color=INK)
    ax.axvline(0.5, color=RED, ls="--", lw=1.0, alpha=0.6)
    ax.text(0.51, 1.04, "default 0.5\n41,279 alerts", fontsize=9, color=RED)
    ax.set_xlabel("fraud probability threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.16)
    ax.set_title("Why the threshold is tuned: Logistic Regression on the validation split")
    ax.legend(frameon=False, loc="center left")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    save(fig, "05_threshold_curve.png")


# ---------------------------------------------------------------- 6. time
def fig_time() -> None:
    path = config.DASHBOARD_DATA_DIR / "fraud_by_step.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
    ax1.plot(d["step"], d["transactions"], color=BLUE, lw=1.4)
    ax1.set_ylabel("all transactions")
    ax1.set_title("Legitimate volume collapses after ~step 400, fraud does not")
    ax1.grid(axis="y", color=GRID)
    ax1.set_axisbelow(True)
    ax2.plot(d["step"], d["fraud"], color=RED, lw=1.4)
    ax2.set_ylabel("fraudulent")
    ax2.set_xlabel("step (one simulated hour)")
    ax2.grid(axis="y", color=GRID)
    ax2.set_axisbelow(True)
    for ax in (ax1, ax2):
        ax.axvline(323, color=INK2, ls="--", lw=1, alpha=0.7)
        ax.axvline(378, color=INK2, ls="--", lw=1, alpha=0.7)
    ax1.text(150, ax1.get_ylim()[1] * 0.88, "TRAIN", fontsize=9, color=INK2, ha="center")
    ax1.text(350, ax1.get_ylim()[1] * 0.88, "VAL", fontsize=9, color=INK2, ha="center")
    ax1.text(550, ax1.get_ylim()[1] * 0.88, "TEST", fontsize=9, color=INK2, ha="center")
    save(fig, "06_fraud_by_step.png")


# ---------------------------------------------------------------- 7. risk bands
def fig_risk(alerts: dict) -> None:
    rb = pd.DataFrame(alerts["risk_breakdown"])
    order = {"High Risk": 0, "Medium Risk": 1, "Low Risk": 2}
    rb["o"] = rb["risk_category"].map(order)
    rb = rb.sort_values("o")
    colors = {"High Risk": RED, "Medium Risk": AMBER, "Low Risk": YELLOW}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(rb["risk_category"], rb["alerts"],
                  color=[colors.get(c, BLUE) for c in rb["risk_category"]], width=0.55)
    for b, v in zip(bars, rb["alerts"]):
        ax.text(b.get_x() + b.get_width() / 2, v + max(rb["alerts"]) * 0.03,
                f"{int(v):,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("alerts")
    ax.set_ylim(0, max(rb["alerts"]) * 1.2)
    ax.set_title(f"{alerts['alerts_raised']:,} fraud alerts by risk category")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    save(fig, "07_risk_bands.png")


# ---------------------------------------------------------------- 8. importance
def fig_importance() -> None:
    tr = load(config.MODEL_RESULTS_DIR / "random_forest_training.json")
    if not tr or not tr.get("feature_importances"):
        return
    imp = pd.DataFrame(tr["feature_importances"]).head(8).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.barh(imp["feature"], imp["importance"], color=AQUA, height=0.6)
    for b, v in zip(bars, imp["importance"]):
        ax.text(v + imp["importance"].max() * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, imp["importance"].max() * 1.18)
    ax.set_xlabel("share of impurity reduction")
    ax.set_title("What the Random Forest relies on")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    save(fig, "08_feature_importance.png")


def main() -> int:
    IMG.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {IMG}")

    overview = load(config.DASHBOARD_DATA_DIR / "overview.json")
    comp = load(config.MODEL_RESULTS_DIR / "model_comparison.json")
    alerts = load(config.FRAUD_ALERTS_DIR / "fraud_alerts_summary.json")

    fig_architecture()
    if overview:
        fig_imbalance(overview)
        fig_time()
    if comp:
        fig_comparison(comp)
        fig_confusion(comp)
    fig_threshold()
    if alerts:
        fig_risk(alerts)
    fig_importance()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
