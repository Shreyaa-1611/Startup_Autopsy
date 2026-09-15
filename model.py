"""
model.py — Runway Prediction Model
Predicts a startup's lifespan (years from founding to shutdown) from its
funding level, sector, and documented cause of death.

Given the small sample (n=34 startups with disclosed funding), this script
deliberately uses Leave-One-Out Cross-Validation (LOOCV) rather than a
train/test split — with only 34 rows, a single split would be unstable and
easy to game by luck. LOOCV trains on 33 startups and tests on the 1 left
out, repeated 34 times, giving the most honest estimate of real-world
performance this dataset can support.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------- style (matches analysis.py / portfolio) ----------
INK, SLATE, AMBER, TEAL, LINE, PAPER = "#14202B", "#55606B", "#C97A1E", "#2F6F63", "#D3D8D0", "#EFF1EC"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SLATE, "ytick.color": SLATE, "font.family": "monospace", "font.size": 11,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
})

# ---------- 1. Load & prepare ----------
df = pd.read_csv("startup_shutdowns_india_clean.csv")
fdf = df.dropna(subset=["funding_usd_mn"]).copy()          # only startups with disclosed funding
fdf["log_funding"] = np.log1p(fdf["funding_usd_mn"])         # log-transform: funding is right-skewed

# Group sectors that appear fewer than 2 times into "Other" so one-hot
# encoding doesn't create noisy single-startup columns
sector_counts = fdf["sector"].value_counts()
fdf["sector_grp"] = fdf["sector"].where(fdf["sector"].isin(sector_counts[sector_counts >= 2].index), "Other")

FEATURES = ["log_funding", "sector_grp", "cause_category"]
TARGET = "lifespan_years"
X = fdf[FEATURES]
y = fdf[TARGET]

preprocessor = ColumnTransformer(
    [("categorical", OneHotEncoder(handle_unknown="ignore"), ["sector_grp", "cause_category"])],
    remainder="passthrough",  # passes log_funding through unchanged
)

models = {
    "Linear Regression": Pipeline([("prep", preprocessor), ("model", LinearRegression())]),
    "Random Forest": Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(
        n_estimators=300, max_depth=4, random_state=42))]),
}

# ---------- 2. Leave-One-Out Cross-Validation ----------
loo = LeaveOneOut()
results = {}

for name, pipe in models.items():
    preds, actuals = [], []
    for train_idx, test_idx in loo.split(X):
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.append(pipe.predict(X.iloc[test_idx])[0])
        actuals.append(y.iloc[test_idx].values[0])
    preds, actuals = np.array(preds), np.array(actuals)
    results[name] = {
        "mae": mean_absolute_error(actuals, preds),
        "r2": r2_score(actuals, preds),
        "preds": preds,
        "actuals": actuals,
    }

baseline_mae = mean_absolute_error(y, np.full(len(y), y.mean()))

print("=" * 60)
print("RUNWAY PREDICTION MODEL — Leave-One-Out Cross-Validation")
print("=" * 60)
print(f"Baseline (always predict the mean, {y.mean():.2f} yrs): MAE = {baseline_mae:.2f} years")
for name, r in results.items():
    improvement = (1 - r["mae"] / baseline_mae) * 100
    print(f"{name:20s}: MAE = {r['mae']:.2f} years | R\u00b2 = {r['r2']:.3f} | {improvement:+.0f}% vs. baseline")

# Best model = lower MAE
best_name = min(results, key=lambda k: results[k]["mae"])
best = results[best_name]
print(f"\nBest model: {best_name}")

# ---------- 3. Fig 8: Predicted vs. Actual (LOOCV) ----------
fig, ax = plt.subplots(figsize=(6.5, 6))
lims = [0, max(best["actuals"].max(), best["preds"].max()) + 1]
ax.plot(lims, lims, linestyle="--", color=SLATE, linewidth=1, label="Perfect prediction")
ax.scatter(best["actuals"], best["preds"], color=AMBER, s=80, edgecolor=INK, linewidth=0.5, zorder=3)
ax.set_xlabel("Actual lifespan (years)")
ax.set_ylabel("Predicted lifespan (years, LOOCV)")
ax.set_title(f"{best_name}: Predicted vs. Actual Lifespan\n(Leave-One-Out CV, MAE = {best['mae']:.2f} yrs)",
             loc="left", fontsize=12, fontweight="bold")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/08_predicted_vs_actual.png", dpi=160)
plt.close(fig)

# ---------- 4. Fig 9: Feature importance (Random Forest, fit on full data) ----------
rf_pipe = models["Random Forest"]
rf_pipe.fit(X, y)
ohe = rf_pipe.named_steps["prep"].named_transformers_["categorical"]
feature_names = list(ohe.get_feature_names_out(["sector_grp", "cause_category"])) + ["log_funding"]
importances = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=feature_names)
top_importances = importances.sort_values(ascending=True).tail(8)

fig, ax = plt.subplots(figsize=(8, 4.5))
clean_labels = [l.replace("sector_grp_", "Sector: ").replace("cause_category_", "Cause: ").replace("log_funding", "Funding raised (log)")
                for l in top_importances.index]
ax.barh(clean_labels, top_importances.values, color=TEAL)
ax.set_xlabel("Relative importance")
ax.set_title("What Predicts How Long A Startup Survives?", loc="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/09_feature_importance.png", dpi=160)
plt.close(fig)

# ---------- 5. Save the trained model + a metrics summary ----------
joblib.dump(rf_pipe, "runway_model.pkl")

with open("model_metrics.txt", "w") as f:
    f.write("RUNWAY PREDICTION MODEL — RESULTS SUMMARY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Training data: {len(fdf)} startups with disclosed funding\n")
    f.write(f"Validation method: Leave-One-Out Cross-Validation (LOOCV)\n\n")
    f.write(f"Baseline (predict mean lifespan): MAE = {baseline_mae:.2f} years\n")
    for name, r in results.items():
        improvement = (1 - r["mae"] / baseline_mae) * 100
        f.write(f"{name}: MAE = {r['mae']:.2f} years | R2 = {r['r2']:.3f} | {improvement:+.0f}% vs baseline\n")
    f.write(f"\nBest model: {best_name}\n")
    f.write("\nTop feature importances (Random Forest):\n")
    for feat, val in importances.sort_values(ascending=False).head(6).items():
        f.write(f"  {feat}: {val:.3f}\n")

print("\nFigures saved to figures/08_predicted_vs_actual.png and figures/09_feature_importance.png")
print("Model saved to runway_model.pkl")
print("Metrics summary saved to model_metrics.txt")
