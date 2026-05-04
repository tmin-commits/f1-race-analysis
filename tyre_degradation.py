"""
Tyre Degradation Analysis
=========================
Analyses tyre performance degradation across a race stint by compound.
Fits a linear regression to lap times per stint to estimate deg rate (seconds/lap).
"""

import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# FastF1 compound colour map (official)
COMPOUND_COLORS = {
    "SOFT":   "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD":   "#FFFFFF",
    "INTER":  "#39B54A",
    "WET":    "#0067FF",
}


def load_race(year: int, round_name: str) -> fastf1.core.Session:
    session = fastf1.get_session(year, round_name, "R")
    session.load(telemetry=False, weather=False, messages=False)
    return session


def get_stint_laps(session: fastf1.core.Session) -> pd.DataFrame:
    """Return clean lap data with stint info, filtered for representative laps."""
    laps = session.laps.copy()

    # Keep only accurate, non-pit laps
    laps = laps[laps["IsAccurate"]].copy()
    laps = laps[laps["PitOutTime"].isna() & laps["PitInTime"].isna()].copy()

    # Convert lap time to seconds
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    # Remove safety car affected laps (lap time > 1.5× median for that driver)
    def remove_outliers(grp):
        med = grp["LapTimeSec"].median()
        return grp[grp["LapTimeSec"] < med * 1.12]

    laps = laps.groupby("Driver", group_keys=False).apply(remove_outliers)

    # Stint lap number (lap within the current tyre stint)
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["StintLap"] = laps.groupby(["Driver", "Stint"]).cumcount() + 1

    return laps[["Driver", "LapNumber", "StintLap", "Stint", "Compound", "LapTimeSec"]]


def compute_degradation(laps: pd.DataFrame) -> pd.DataFrame:
    """Fit linear regression per compound and return degradation rates."""
    records = []
    for compound, grp in laps.groupby("Compound"):
        if compound not in COMPOUND_COLORS:
            continue
        # Need minimum data points
        if len(grp) < 8:
            continue
        slope, intercept, r, p, _ = stats.linregress(grp["StintLap"], grp["LapTimeSec"])
        records.append({
            "Compound":   compound,
            "DegRate":    round(slope, 3),       # seconds per lap
            "BaseTime":   round(intercept, 3),   # projected lap 1 time
            "R2":         round(r**2, 3),
            "SampleSize": len(grp)
        })
    return pd.DataFrame(records).sort_values("DegRate")


def plot_degradation(laps: pd.DataFrame, race_label: str, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0F0F0F")

    for ax in axes:
        ax.set_facecolor("#1A1A1A")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # ── LEFT: scatter + regression lines ──────────────────────────────────────
    ax = axes[0]
    legend_patches = []
    compounds_used = []

    for compound, grp in laps.groupby("Compound"):
        if compound not in COMPOUND_COLORS or len(grp) < 8:
            continue
        color = COMPOUND_COLORS[compound]
        ec = "#333333" if compound == "HARD" else color

        ax.scatter(grp["StintLap"], grp["LapTimeSec"],
                   color=color, edgecolors=ec, linewidths=0.5,
                   alpha=0.55, s=28, zorder=3)

        # Regression line
        x_range = np.linspace(grp["StintLap"].min(), grp["StintLap"].max(), 100)
        slope, intercept, *_ = stats.linregress(grp["StintLap"], grp["LapTimeSec"])
        ax.plot(x_range, slope * x_range + intercept,
                color=color, linewidth=2.5, zorder=4)

        legend_patches.append(mpatches.Patch(color=color, label=f"{compound}  ({slope:+.3f}s/lap)"))
        compounds_used.append(compound)

    ax.set_xlabel("Stint Lap Number", color="#AAAAAA", fontsize=11)
    ax.set_ylabel("Lap Time (seconds)", color="#AAAAAA", fontsize=11)
    ax.set_title("Tyre Degradation by Compound", color="#FFFFFF", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#888888")
    ax.legend(handles=legend_patches, facecolor="#222222", edgecolor="#444444",
              labelcolor="#DDDDDD", fontsize=10, loc="upper left")
    ax.grid(axis="y", color="#2A2A2A", linewidth=0.8, zorder=1)

    # ── RIGHT: bar chart of deg rates ─────────────────────────────────────────
    ax2 = axes[1]
    deg_df = compute_degradation(laps)

    bars = ax2.barh(deg_df["Compound"],
                    deg_df["DegRate"],
                    color=[COMPOUND_COLORS.get(c, "#888888") for c in deg_df["Compound"]],
                    edgecolor="#333333", height=0.55)

    # Value labels
    for bar, (_, row) in zip(bars, deg_df.iterrows()):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"+{row['DegRate']:.3f}s/lap  (n={row['SampleSize']})",
                 va="center", color="#CCCCCC", fontsize=10)

    ax2.set_xlabel("Degradation Rate (seconds per lap)", color="#AAAAAA", fontsize=11)
    ax2.set_title("Deg Rate Comparison", color="#FFFFFF", fontsize=13, fontweight="bold", pad=12)
    ax2.tick_params(colors="#888888")
    ax2.set_facecolor("#1A1A1A")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")
    ax2.axvline(0, color="#555555", linewidth=1)
    ax2.grid(axis="x", color="#2A2A2A", linewidth=0.8)
    ax2.set_xlim(left=0)

    fig.suptitle(f"Tyre Degradation Analysis — {race_label}",
                 color="#FFFFFF", fontsize=16, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close()
    print(f"  [✓] Saved: {output_path}")
    return compute_degradation(laps)
