# plot_unified_roc.py  (no cross-file label check)
# ------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

FILES = {
    "RF (Limited)"        : "results/roc_curve/rf_5_predictions.csv",
    "RF (Full)"           : "results/roc_curve/rf_predictions.csv",
    "CatBoost (Limited)"  : "results/roc_curve/catboost_5_predictions.csv",
    "CatBoost (Full)"     : "results/roc_curve/catboost_predictions.csv",
    "MLP (Limited)"       : "results/roc_curve/mlp_5_predictions.csv",
    "MLP (Full)"          : "results/roc_curve/mlp_predictions.csv",
    "DistilBERT (Limited)": "results/roc_curve/evaluation_predictions_with_scores_5_cpt.csv",
    "DistilBERT (Full)"   : "results/roc_curve/evaluation_predictions_with_scores_all_cpt.csv",
}

DATA_DIR = "."   # change if needed
SAVE_PNG = "results/roc_curve/all_models_roc.png"

roc_data = {}  # run → (fpr, tpr, auc, n)

for run_name, rel_path in FILES.items():
    path = os.path.join(DATA_DIR, rel_path)
    df   = pd.read_csv(path)

    if not {"true_label", "predicted_probability"}.issubset(df.columns):
        raise ValueError(f"{rel_path} is missing required columns.")

    y_true = df["true_label"].values
    y_prob = df["predicted_probability"].values

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    roc_data[run_name] = (fpr, tpr, roc_auc, len(y_true))

# ------------------------ PLOT -----------------------------
plt.figure(figsize=(6, 6))

for name, (fpr, tpr, roc_auc, n) in roc_data.items():
    plt.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC {roc_auc:.2f}, n={n})")

plt.plot([0, 1], [0, 1], "k--", lw=1)  # chance line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Unified ROC – Eight Model/Feature Configurations\n(each model evaluated on its own test subset)")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(SAVE_PNG, dpi=300)
plt.show()

print(f"Unified ROC plot saved to {SAVE_PNG}")