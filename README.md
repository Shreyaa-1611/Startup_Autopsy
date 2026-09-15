# Startup Autopsy: What Actually Kills Funded Indian Startups

A data analytics project by **Shreya Mishra** — B.Tech Computer Science, Chandigarh University.

## What this is

Most coverage of Indian startup shutdowns treats it as one story: "funding winter." This project builds a hand-compiled dataset of **52 real, named, funded Indian startups** that shut down between 2023–2025 — with sector, funding raised, founding/shutdown year, and the founders' own stated reason for closing — to test that assumption properly.

**Headline finding:** Capital crunch is the single biggest documented cause of death (31%), but Product-Market Fit failure and Regulatory Shock are statistically tied for second place (19% each). Regulation is a top-three killer of Indian startups, not a footnote — and it shows no sign of fading as the funding winter thaws.

## Files

| File | What it is |
|---|---|
| `startup_shutdowns_india.csv` | The raw dataset — 52 startups × 8 fields |
| `Startup_Autopsy.ipynb` | Full Jupyter notebook: cleaning, categorization, and all analysis |
| `Startup_Autopsy_Report.html` | A standalone, readable write-up of the findings — open directly in a browser, no Jupyter needed |
| `analysis.py` | The chart-generation script, if you want to regenerate figures |
| `figures/` | All 7 charts as standalone PNGs |

## Data source

All 52 startups are drawn from Inc42's annual "Startup Graveyard" reports:
- [2023: 15 startups](https://inc42.com/features/plagued-by-funding-winter-bad-economics-here-are-15-indian-startups-that-shutdown-in-2023/)
- [2024: 12 startups](https://inc42.com/features/2024s-startup-graveyard-12-indian-startups-that-shut-down-this-year/)
- [2025: 25 startups](https://inc42.com/features/25-indian-startups-shut-down-in-2025/)

Each entry reflects the founders' own public statement (LinkedIn post, press interview, or Inc42 reporting) on why the company shut down.

**This is not a census.** Tracxn data cited in the same reports shows over 28,000 Indian startups shut down in 2023–2024 alone. This dataset covers only the ones prominent or well-funded enough to be individually reported — a study of *documented, funded* shutdowns, not of startup failure in general.

## Tools used

Python · Pandas · NumPy · Matplotlib · Jupyter

## Key findings

1. Capital crunch causes 31% of documented shutdowns; No PMF and Regulatory Shock are tied for second at 19% each.
2. Regulatory risk is constant across all three years (4 → 2 → 4 shutdowns) — not a fading pandemic-era problem.
3. Founder/governance failures are a distinctly 2025 phenomenon (0 → 0 → 5), emerging as the ecosystem's biggest companies reach a scale where fraud and mismanagement become visible.
4. Funding buys time, not survival (r = 0.62, log-funding vs. lifespan) — a real but moderate relationship.
5. Some sectors have one dominant killer: Gaming & RMG dies almost exclusively from regulation; Enterprise SaaS and Healthtech die mostly from lack of product-market fit.

## Limitations

- Small, media-curated sample (n=52) — skews toward better-funded, Bengaluru-headquartered companies.
- One dominant cause assigned per startup, even where multiple factors contributed.
- Funding figures are approximate where originally reported in INR.

## Next steps

Extend with Tracxn's aggregate shutdown counts to weight these categories against the true base rate, and track 2026 shutdowns to see whether the Founder/Legal category keeps growing.
