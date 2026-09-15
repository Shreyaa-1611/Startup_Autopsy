import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------- style ----------
INK = "#14202B"
SLATE = "#55606B"
AMBER = "#C97A1E"
TEAL = "#2F6F63"
LINE = "#D3D8D0"
PAPER = "#EFF1EC"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "font.family": "monospace",
    "font.size": 11,
    "axes.grid": True,
    "grid.color": LINE,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

CAT_COLORS = {
    "Capital / Funding Crunch": AMBER,
    "No Product-Market Fit": TEAL,
    "Regulatory Shock": "#9C4221",
    "Founder / Legal / Insolvency": "#7B3F61",
    "Scaling / Execution Failure": "#3D6A8A",
    "Weak Business Model / Economics": "#8A8F5C",
    "Other": SLATE,
}

# ---------- load & clean ----------
df = pd.read_csv("startup_shutdowns_india.csv")
df["lifespan_years"] = df["shutdown_year"] - df["founded_year"]

def categorize(cause: str) -> str:
    c = cause.lower()
    if "regulatory" in c:
        return "Regulatory Shock"
    if "founder" in c or "insolvency" in c or "legal turmoil" in c or "acquirer" in c:
        return "Founder / Legal / Insolvency"
    if "funding crunch" in c or "no follow-on" in c or "cash burn" in c:
        return "Capital / Funding Crunch"
    if "pmf" in c or "traction" in c or "retention" in c:
        return "No Product-Market Fit"
    if "business model" in c or "unit economics" in c or "biz model" in c or "commoditised" in c or "margins" in c:
        return "Weak Business Model / Economics"
    if "scal" in c or "pivot" in c or "demand" in c or "debt" in c:
        return "Scaling / Execution Failure"
    return "Other"

df["cause_category"] = df["primary_cause"].apply(categorize)
df.to_csv("startup_shutdowns_india_clean.csv", index=False)

# ---------- Fig 1: shutdowns per year ----------
fig, ax = plt.subplots(figsize=(7, 4))
counts = df["shutdown_year"].value_counts().sort_index()
bars = ax.bar(counts.index.astype(str), counts.values, color=AMBER, width=0.55)
ax.set_title("Documented Funded-Startup Shutdowns, By Year", loc="left", fontsize=13, fontweight="bold")
ax.set_ylabel("Startups")
for b in bars:
    ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, max(counts.values) * 1.2)
fig.tight_layout()
fig.savefig("figures/01_shutdowns_by_year.png", dpi=160)
plt.close(fig)

# ---------- Fig 2: cause category distribution ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
cat_counts = df["cause_category"].value_counts()
colors = [CAT_COLORS.get(c, SLATE) for c in cat_counts.index]
bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color=colors[::-1])
ax.set_title("What Actually Kills Funded Indian Startups (n=52)", loc="left", fontsize=12.5, fontweight="bold")
ax.set_xlabel("Number of startups")
for b in bars:
    ax.annotate(f"{int(b.get_width())}", (b.get_width(), b.get_y() + b.get_height()/2),
                ha="left", va="center", fontsize=10, xytext=(4, 0), textcoords="offset points")
fig.tight_layout()
fig.savefig("figures/02_cause_distribution.png", dpi=160)
plt.close(fig)

# ---------- Fig 3: cause category by year (grouped) ----------
fig, ax = plt.subplots(figsize=(9, 5))
pivot = pd.crosstab(df["cause_category"], df["shutdown_year"])
pivot = pivot.reindex(cat_counts.index)  # order by overall frequency
years = pivot.columns.tolist()
n_years = len(years)
x = np.arange(len(pivot.index))
width = 0.25
for i, yr in enumerate(years):
    ax.bar(x + (i - n_years/2 + 0.5) * width, pivot[yr].values, width,
           label=str(yr), color=[AMBER, TEAL, INK][i % 3])
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=25, ha="right", fontsize=9)
ax.set_title("Cause Of Death Shifts As The Ecosystem Matures", loc="left", fontsize=13, fontweight="bold")
ax.set_ylabel("Startups")
ax.legend(title="Year", frameon=False)
fig.tight_layout()
fig.savefig("figures/03_cause_by_year.png", dpi=160)
plt.close(fig)

# ---------- Fig 4: avg lifespan by cause category ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
life_by_cause = df.groupby("cause_category")["lifespan_years"].mean().sort_values()
colors = [CAT_COLORS.get(c, SLATE) for c in life_by_cause.index]
bars = ax.barh(life_by_cause.index, life_by_cause.values, color=colors)
ax.set_title("How Long Startups Survived Before Each Type Of Death", loc="left", fontsize=12, fontweight="bold")
ax.set_xlabel("Average years from founding to shutdown")
for b in bars:
    ax.annotate(f"{b.get_width():.1f} yrs", (b.get_width(), b.get_y() + b.get_height()/2),
                ha="left", va="center", fontsize=10, xytext=(4, 0), textcoords="offset points")
fig.tight_layout()
fig.savefig("figures/04_lifespan_by_cause.png", dpi=160)
plt.close(fig)

# ---------- Fig 5: funding vs lifespan scatter ----------
fdf = df.dropna(subset=["funding_usd_mn"]).copy()
fdf["log_funding"] = np.log1p(fdf["funding_usd_mn"])
corr = np.corrcoef(fdf["log_funding"], fdf["lifespan_years"])[0, 1]

fig, ax = plt.subplots(figsize=(8, 5.5))
for cat, color in CAT_COLORS.items():
    sub = fdf[fdf["cause_category"] == cat]
    if len(sub):
        ax.scatter(sub["funding_usd_mn"], sub["lifespan_years"], color=color, s=70,
                   alpha=0.85, edgecolor=INK, linewidth=0.4, label=cat)
ax.set_xscale("log")
ax.set_xlabel("Total funding raised, US$ millions (log scale)")
ax.set_ylabel("Years from founding to shutdown")
ax.set_title(f"Money Buys Time, Not Survival  (r = {corr:.2f} on log-funding)", loc="left", fontsize=13, fontweight="bold")
# annotate a few notable points
for _, row in fdf.nlargest(4, "funding_usd_mn").iterrows():
    ax.annotate(row["name"], (row["funding_usd_mn"], row["lifespan_years"]),
                fontsize=8, xytext=(6, 4), textcoords="offset points", color=SLATE)
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
fig.tight_layout()
fig.savefig("figures/05_funding_vs_lifespan.png", dpi=160)
plt.close(fig)

# ---------- Fig 6: total funding by cause category ----------
fig, ax = plt.subplots(figsize=(9.5, 4.5))
fund_by_cause = fdf.groupby("cause_category")["funding_usd_mn"].sum().sort_values()
colors = [CAT_COLORS.get(c, SLATE) for c in fund_by_cause.index]
bars = ax.barh(fund_by_cause.index, fund_by_cause.values, color=colors)
ax.set_title("Capital Deployed Into Startups That Ultimately Shut Down", loc="left", fontsize=12, fontweight="bold")
ax.set_xlabel("Total disclosed funding, US$ millions")
for b in bars:
    ax.annotate(f"${b.get_width():,.0f}M", (b.get_width(), b.get_y() + b.get_height()/2),
                ha="left", va="center", fontsize=10, xytext=(4, 0), textcoords="offset points")
fig.tight_layout()
fig.savefig("figures/06_funding_by_cause.png", dpi=160)
plt.close(fig)

# ---------- Fig 7: sector x cause heatmap ----------
top_sectors = df["sector"].value_counts().head(9).index
heat_df = df[df["sector"].isin(top_sectors)]
heat = pd.crosstab(heat_df["sector"], heat_df["cause_category"])
heat = heat.reindex(index=top_sectors)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heat.values, cmap="YlOrBr", aspect="auto")
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index, fontsize=9)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center",
                    color=INK if v < heat.values.max()/1.6 else "white", fontsize=9)
ax.set_title("Which Sectors Die From Which Causes", loc="left", fontsize=13, fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.7, label="Number of startups")
fig.tight_layout()
fig.savefig("figures/07_sector_cause_heatmap.png", dpi=160)
plt.close(fig)

print("All figures generated.")
print(f"log-funding vs lifespan correlation: {corr:.3f}")
print(df["cause_category"].value_counts())
