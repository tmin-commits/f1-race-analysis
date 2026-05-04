"""
Pit Stop Strategy Analysis
==========================
Visualises the tyre strategy of the top N finishers across a race —
the classic "strategy chart" used by Sky Sports F1 and teams internally.
"""

import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COMPOUND_COLORS = {
    "SOFT":   "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD":   "#EBEBEB",
    "INTER":  "#39B54A",
    "WET":    "#0067FF",
    "UNKNOWN":"#888888",
}

COMPOUND_ABBREV = {
    "SOFT": "S", "MEDIUM": "M", "HARD": "H",
    "INTER": "I", "WET": "W", "UNKNOWN": "?"
}


def build_strategy_df(session: fastf1.core.Session, top_n: int = 10) -> pd.DataFrame:
    """Return a tidy DataFrame of stint info for the top N finishers."""
    results = session.results.sort_values("ClassifiedPosition").head(top_n)
    laps = session.laps.copy()

    records = []
    for _, driver_row in results.iterrows():
        drv = driver_row["Abbreviation"]
        drv_laps = laps[laps["Driver"] == drv].sort_values("LapNumber")

        stint_groups = drv_laps.groupby("Stint")
        for stint_num, stint_laps in stint_groups:
            compound = stint_laps["Compound"].mode()[0] if not stint_laps["Compound"].isna().all() else "UNKNOWN"
            records.append({
                "Driver":    drv,
                "Position":  int(driver_row["ClassifiedPosition"]) if str(driver_row["ClassifiedPosition"]).isdigit() else 99,
                "Stint":     stint_num,
                "Compound":  compound,
                "StartLap":  int(stint_laps["LapNumber"].min()),
                "EndLap":    int(stint_laps["LapNumber"].max()),
                "StintLen":  len(stint_laps),
            })

    return pd.DataFrame(records).sort_values(["Position", "Stint"])


def plot_strategy(session: fastf1.core.Session,
                  race_label: str,
                  output_path: str,
                  top_n: int = 10):

    df = build_strategy_df(session, top_n=top_n)
    total_laps = session.laps["LapNumber"].max()
    drivers = df.drop_duplicates("Driver").sort_values("Position")["Driver"].tolist()

    fig, ax = plt.subplots(figsize=(16, max(7, len(drivers) * 0.75 + 2)))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#1A1A1A")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    y_positions = {drv: i for i, drv in enumerate(reversed(drivers))}

    for _, row in df.iterrows():
        drv = row["Driver"]
        y = y_positions[drv]
        color = COMPOUND_COLORS.get(row["Compound"], "#888888")
        ec = "#555555" if row["Compound"] == "HARD" else color

        bar = ax.barh(y=y,
                      left=row["StartLap"] - 1,
                      width=row["StintLen"],
                      height=0.62,
                      color=color,
                      edgecolor=ec,
                      linewidth=0.8)

        # Label the stint length inside the bar if wide enough
        if row["StintLen"] >= 6:
            abbrev = COMPOUND_ABBREV.get(row["Compound"], "?")
            text_color = "#1A1A1A" if row["Compound"] in ("MEDIUM", "HARD") else "#FFFFFF"
            ax.text(row["StartLap"] - 1 + row["StintLen"] / 2, y,
                    f"{abbrev} ({row['StintLen']})",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

        # Pit stop marker (vertical line at stint start, except lap 1)
        if row["StartLap"] > 1:
            ax.axvline(row["StartLap"] - 1, ymin=(y - 0.31) / len(drivers),
                       ymax=(y + 0.31) / len(drivers),
                       color="#FFFFFF", linewidth=1.2, alpha=0.6, zorder=5)

    # Driver labels on y-axis with position number
    pos_map = df.drop_duplicates("Driver").set_index("Driver")["Position"].to_dict()
    ytick_labels = []
    for drv in reversed(drivers):
        pos = pos_map.get(drv, "")
        ytick_labels.append(f"P{pos}  {drv}")

    ax.set_yticks(list(range(len(drivers))))
    ax.set_yticklabels(ytick_labels, color="#CCCCCC", fontsize=11, fontfamily="monospace")
    ax.set_xlabel("Lap Number", color="#AAAAAA", fontsize=11)
    ax.set_xlim(0, total_laps)
    ax.set_ylim(-0.5, len(drivers) - 0.5)
    ax.tick_params(colors="#888888")
    ax.grid(axis="x", color="#252525", linewidth=0.8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))

    # Legend
    patches = [
        mpatches.Patch(color=COMPOUND_COLORS[c], label=c,
                       edgecolor="#555555" if c == "HARD" else COMPOUND_COLORS[c])
        for c in ["SOFT", "MEDIUM", "HARD", "INTER", "WET"]
    ]
    ax.legend(handles=patches, loc="lower right",
              facecolor="#222222", edgecolor="#444444",
              labelcolor="#DDDDDD", fontsize=10, ncol=5)

    ax.set_title(f"Pit Stop Strategy — {race_label} (Top {top_n})",
                 color="#FFFFFF", fontsize=15, fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close()
    print(f"  [✓] Saved: {output_path}")

    # Print strategy summary
    print(f"\n  ── Strategy Summary ──")
    for drv in drivers:
        stints = df[df["Driver"] == drv].sort_values("Stint")
        summary = "  →  ".join(
            f"{COMPOUND_ABBREV.get(r['Compound'],'?')}({r['StintLen']}L)"
            for _, r in stints.iterrows()
        )
        pos = pos_map.get(drv, "?")
        print(f"  P{pos:<2} {drv}: {summary}")
