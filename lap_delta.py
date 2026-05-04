"""
Lap Time Delta Analysis
=======================
Compares two drivers' lap times across a race, showing the cumulative
time gap and per-lap delta — the standard "gap chart" used in F1 broadcasts.
"""

import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_driver_laps(session: fastf1.core.Session, driver: str) -> pd.DataFrame:
    laps = session.laps.pick_drivers(driver).copy()
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    return laps[["LapNumber", "LapTimeSec", "Compound", "Stint"]].set_index("LapNumber")


def compute_delta(session: fastf1.core.Session, driver_a: str, driver_b: str) -> pd.DataFrame:
    """
    Returns per-lap delta and cumulative gap between two drivers.
    Positive cumulative = driver_a is AHEAD of driver_b.
    """
    a = get_driver_laps(session, driver_a).rename(columns={"LapTimeSec": "A"})
    b = get_driver_laps(session, driver_b).rename(columns={"LapTimeSec": "B"})

    df = a.join(b[["B"]], how="inner")
    df = df.dropna(subset=["A", "B"])

    # Per-lap delta: positive = driver_a faster this lap
    df["PerLapDelta"] = df["B"] - df["A"]

    # Cumulative gap: positive = driver_a is ahead overall
    df["CumGap"] = df["PerLapDelta"].cumsum()

    return df


def plot_delta(session: fastf1.core.Session,
               driver_a: str, driver_b: str,
               race_label: str, output_path: str):

    df = compute_delta(session, driver_a, driver_b)

    # Get team colours
    def get_color(driver):
        try:
            return "#" + session.get_driver(driver)["TeamColor"]
        except Exception:
            return "#FFFFFF"

    col_a = get_color(driver_a)
    col_b = get_color(driver_b)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1.2]})
    fig.patch.set_facecolor("#0F0F0F")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#1A1A1A")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.tick_params(colors="#888888")
        ax.grid(axis="y", color="#252525", linewidth=0.8)
        ax.grid(axis="x", color="#252525", linewidth=0.5)

    laps = df.index

    # ── TOP: cumulative gap ───────────────────────────────────────────────────
    above = df["CumGap"].clip(lower=0)
    below = df["CumGap"].clip(upper=0)

    ax1.fill_between(laps, above, 0, color=col_a, alpha=0.35, label=f"{driver_a} ahead")
    ax1.fill_between(laps, below, 0, color=col_b, alpha=0.35, label=f"{driver_b} ahead")
    ax1.plot(laps, df["CumGap"], color="#FFFFFF", linewidth=1.8, zorder=5)
    ax1.axhline(0, color="#555555", linewidth=1, zorder=4)

    # Annotate final gap
    final_gap = df["CumGap"].iloc[-1]
    winner = driver_a if final_gap > 0 else driver_b
    ax1.annotate(f"Final gap: {abs(final_gap):.1f}s ({winner} ahead)",
                 xy=(laps[-1], final_gap),
                 xytext=(-15, 15 if final_gap > 0 else -25),
                 textcoords="offset points",
                 color="#FFFFFF", fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="#888888"))

    ax1.set_ylabel(f"Cumulative Gap (s)\n← {driver_b} ahead  |  {driver_a} ahead →",
                   color="#AAAAAA", fontsize=10)
    ax1.set_title(f"{driver_a} vs {driver_b} — Race Gap Analysis",
                  color="#FFFFFF", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(facecolor="#222222", edgecolor="#444444", labelcolor="#DDDDDD", fontsize=10)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fs"))

    # ── BOTTOM: per-lap delta ─────────────────────────────────────────────────
    colors_bar = [col_a if v >= 0 else col_b for v in df["PerLapDelta"]]
    ax2.bar(laps, df["PerLapDelta"], color=colors_bar, alpha=0.8, width=0.8)
    ax2.axhline(0, color="#555555", linewidth=1)

    # Rolling average
    rolling = df["PerLapDelta"].rolling(5, center=True).mean()
    ax2.plot(laps, rolling, color="#FFFFFF", linewidth=1.5, linestyle="--",
             label="5-lap rolling avg", zorder=5)

    ax2.set_xlabel("Lap Number", color="#AAAAAA", fontsize=11)
    ax2.set_ylabel("Per-Lap Δ (s)", color="#AAAAAA", fontsize=10)
    ax2.legend(facecolor="#222222", edgecolor="#444444", labelcolor="#DDDDDD", fontsize=9)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+.2fs"))

    fig.suptitle(f"Lap Delta Analysis — {race_label}",
                 color="#FFFFFF", fontsize=16, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close()
    print(f"  [✓] Saved: {output_path}")

    # Print summary stats
    print(f"\n  ── Delta Summary: {driver_a} vs {driver_b} ──")
    print(f"  Avg per-lap delta:  {df['PerLapDelta'].mean():+.3f}s")
    print(f"  Laps {driver_a} faster: {(df['PerLapDelta'] > 0).sum()}")
    print(f"  Laps {driver_b} faster: {(df['PerLapDelta'] < 0).sum()}")
    print(f"  Peak {driver_a} advantage: {df['CumGap'].max():.1f}s")
    print(f"  Final gap: {abs(final_gap):.1f}s to {winner}")
