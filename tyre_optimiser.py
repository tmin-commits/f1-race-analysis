"""
Tyre Strategy Optimiser
=======================
Uses machine learning to model tyre degradation per compound and
simulate all possible pit stop strategies to find the optimal window.

What it does:
  1. Loads race data and extracts clean stint laps
  2. Fits a polynomial regression deg model per compound (fuel-corrected)
  3. Simulates every possible 1-stop and 2-stop strategy
  4. Scores each strategy by predicted total race time
  5. Compares optimal strategy vs what teams actually did
  6. Visualises results with strategy comparison chart

Usage:
    python tyre_optimiser.py
    python tyre_optimiser.py --year 2024 --race "Bahrain" --driver VER
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from itertools import combinations

fastf1.Cache.enable_cache("cache/")

COMPOUND_COLORS = {
    "SOFT":   "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD":   "#EBEBEB",
    "INTER":  "#39B54A",
    "WET":    "#0067FF",
}

# Approximate pit stop time loss (seconds) — varies by track
PIT_LOSS = {
    "Bahrain": 22.0, "Monaco": 18.0, "British": 21.5,
    "Italian": 19.0, "Japanese": 23.0, "Spanish": 21.0,
}
DEFAULT_PIT_LOSS = 21.0

# Fuel burn rate: ~1.6 kg/lap, each kg = ~0.03s lap time improvement
FUEL_EFFECT = 1.6 * 0.03  # seconds per lap improvement from fuel burn


def load_session(year, race):
    session = fastf1.get_session(year, race, "R")
    session.load(telemetry=False, weather=False, messages=False)
    return session


def extract_stint_data(session):
    """Extract clean stint laps across all drivers."""
    laps = session.laps.copy()
    laps = laps[laps["IsAccurate"]].copy()
    laps = laps[laps["PitOutTime"].isna() & laps["PitInTime"].isna()].copy()
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["StintLap"] = laps.groupby(["Driver", "Stint"]).cumcount() + 1

    # Fuel correction: remove fuel effect to isolate tyre deg
    total_laps = laps["LapNumber"].max()
    laps["FuelCorrectedLapTime"] = (
        laps["LapTimeSec"] - (total_laps - laps["LapNumber"]) * FUEL_EFFECT
    )

    # Remove outliers per driver
    def remove_outliers(grp):
        med = grp["FuelCorrectedLapTime"].median()
        return grp[grp["FuelCorrectedLapTime"] < med * 1.08]

    laps = laps.groupby("Driver", group_keys=False).apply(remove_outliers)
    return laps[["Driver", "LapNumber", "StintLap", "Stint",
                 "Compound", "LapTimeSec", "FuelCorrectedLapTime"]]


def fit_deg_models(laps):
    """
    Fit polynomial regression deg model per compound.
    Returns dict: compound -> (model, base_laptime, r2)
    """
    models = {}
    print("\n  ── Degradation Models ──")
    for compound, grp in laps.groupby("Compound"):
        if compound not in COMPOUND_COLORS or len(grp) < 10:
            continue
        X = grp["StintLap"].values.reshape(-1, 1)
        y = grp["FuelCorrectedLapTime"].values

        # Polynomial degree 2 — captures initial warm-up + deg curve
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # Base lap time = predicted time at lap 1 of stint
        base = model.predict([[1]])[0]
        models[compound] = {"model": model, "base": base, "r2": r2,
                            "max_stint": int(grp["StintLap"].max())}

        deg_rate = model.predict([[6]])[0] - model.predict([[1]])[0]
        print(f"  {compound:<8} base: {base:.2f}s  "
              f"deg over 5 laps: +{deg_rate:.3f}s  R²: {r2:.3f}")

    return models


def predict_stint_time(model_info, num_laps):
    """Predict total stint time for a given number of laps."""
    model = model_info["model"]
    laps_arr = np.arange(1, num_laps + 1).reshape(-1, 1)
    return model.predict(laps_arr).sum()


def simulate_strategies(models, total_laps, pit_loss, compounds_available):
    """
    Simulate all valid 1-stop and 2-stop strategies.
    Rule: must use at least 2 different compounds (F1 regulation).
    Returns list of strategy dicts sorted by total time.
    """
    results = []
    available = [c for c in compounds_available if c in models]

    # ── 1-stop strategies ────────────────────────────────────────────────────
    for pit_lap in range(5, total_laps - 5):
        for c1 in available:
            for c2 in available:
                if c1 == c2:
                    continue  # Must use 2 compounds
                stint1 = pit_lap
                stint2 = total_laps - pit_lap

                if stint1 < 3 or stint2 < 3:
                    continue
                if stint1 > models[c1]["max_stint"] * 1.4:
                    continue
                if stint2 > models[c2]["max_stint"] * 1.4:
                    continue

                t1 = predict_stint_time(models[c1], stint1)
                t2 = predict_stint_time(models[c2], stint2)
                total = t1 + t2 + pit_loss

                results.append({
                    "stops":    1,
                    "strategy": f"{c1}({stint1}L) → {c2}({stint2}L)",
                    "pit_laps": [pit_lap],
                    "compounds": [c1, c2],
                    "total_time": total,
                    "t_stints":  [t1, t2],
                })

    # ── 2-stop strategies ────────────────────────────────────────────────────
    for pit1 in range(5, total_laps - 10):
        for pit2 in range(pit1 + 5, total_laps - 5):
            for c1 in available:
                for c2 in available:
                    for c3 in available:
                        # Must use at least 2 different compounds
                        if len({c1, c2, c3}) < 2:
                            continue
                        s1 = pit1
                        s2 = pit2 - pit1
                        s3 = total_laps - pit2
                        if s1 < 3 or s2 < 3 or s3 < 3:
                            continue

                        t1 = predict_stint_time(models[c1], s1)
                        t2 = predict_stint_time(models[c2], s2)
                        t3 = predict_stint_time(models[c3], s3)
                        total = t1 + t2 + t3 + (pit_loss * 2)

                        results.append({
                            "stops":    2,
                            "strategy": f"{c1}({s1}L) → {c2}({s2}L) → {c3}({s3}L)",
                            "pit_laps": [pit1, pit2],
                            "compounds": [c1, c2, c3],
                            "total_time": total,
                            "t_stints":  [t1, t2, t3],
                        })

    return sorted(results, key=lambda x: x["total_time"])


def get_actual_strategy(session, driver):
    """Get the actual strategy used by the driver."""
    laps = session.laps.pick_drivers(driver).sort_values("LapNumber")
    stints = []
    for stint_num, grp in laps.groupby("Stint"):
        compound = grp["Compound"].mode()[0] if not grp["Compound"].isna().all() else "UNKNOWN"
        stints.append({
            "compound": compound,
            "laps":     len(grp),
            "start_lap": int(grp["LapNumber"].min()),
        })
    return stints


def format_time(seconds):
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


def plot_optimiser(models, top_strategies, actual_strategy,
                   driver, race_label, total_laps, output_path):

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0A0A0A")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]

    for ax in axes:
        ax.set_facecolor("#131313")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2A2A")
        ax.tick_params(colors="#888888", labelsize=9)

    # ── CHART 1: Deg curves per compound ─────────────────────────────────────
    ax = axes[0]
    x = np.arange(1, 35)
    for compound, info in models.items():
        y = info["model"].predict(x.reshape(-1, 1))
        # Normalise to show relative deg from base
        y_rel = y - y[0]
        color = COMPOUND_COLORS.get(compound, "#888888")
        ec = "#555" if compound == "HARD" else color
        ax.plot(x, y_rel, color=color, linewidth=2.5, label=compound)

    ax.set_xlabel("Stint Lap", color="#AAAAAA", fontsize=10)
    ax.set_ylabel("Lap Time Increase (s)", color="#AAAAAA", fontsize=10)
    ax.set_title("Tyre Degradation Model\n(fuel-corrected, relative to lap 1)",
                 color="#FFFFFF", fontsize=11, fontweight="bold")
    ax.legend(facecolor="#1A1A1A", edgecolor="#333", labelcolor="#DDD", fontsize=9)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.grid(axis="y", color="#1E1E1E", linewidth=0.6)

    # ── CHART 2: Top strategies ranked ───────────────────────────────────────
    ax = axes[1]
    top_n = min(12, len(top_strategies))
    labels = [s["strategy"] for s in top_strategies[:top_n]]
    times  = [s["total_time"] for s in top_strategies[:top_n]]
    best   = times[0]
    gaps   = [t - best for t in times]
    colors_bar = ["#C0392B" if i == 0 else "#2A2A2A" for i in range(top_n)]

    bars = ax.barh(range(top_n), gaps, color=colors_bar,
                   edgecolor="#333", height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"{i+1}. {l[:38]}" for i, l in enumerate(labels)],
                       fontsize=7.5, color="#CCCCCC", fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Time vs Best Strategy (s)", color="#AAAAAA", fontsize=10)
    ax.set_title("Top Strategy Rankings",
                 color="#FFFFFF", fontsize=11, fontweight="bold")

    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"+{gap:.1f}s" if gap > 0 else "OPTIMAL",
                va="center", fontsize=7.5,
                color="#C0392B" if gap == 0 else "#888888")

    ax.grid(axis="x", color="#1E1E1E", linewidth=0.6)
    ax.set_facecolor("#131313")

    # ── CHART 3: Optimal vs Actual strategy bar ───────────────────────────────
    ax = axes[2]
    optimal = top_strategies[0]

    def draw_strategy_bar(ax, stints_data, y, height, label):
        x_start = 0
        for stint in stints_data:
            compound = stint["compound"]
            laps = stint["laps"]
            color = COMPOUND_COLORS.get(compound, "#888888")
            ec = "#555" if compound == "HARD" else color
            ax.barh(y, laps, left=x_start, height=height,
                    color=color, edgecolor=ec, linewidth=0.8)
            if laps >= 5:
                text_color = "#1A1A1A" if compound in ("MEDIUM", "HARD") else "#FFFFFF"
                ax.text(x_start + laps/2, y,
                        f"{compound[:1]}({laps}L)",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)
            if x_start > 0:
                ax.axvline(x_start, ymin=(y-height/2+0.5)/3,
                          ymax=(y+height/2+0.5)/3,
                          color="#FFFFFF", linewidth=1.2, alpha=0.5)
            x_start += laps
        ax.text(-2, y, label, ha="right", va="center",
                color="#CCCCCC", fontsize=10, fontweight="bold")

    # Optimal strategy
    opt_stints = []
    pit_laps = [0] + optimal["pit_laps"] + [total_laps]
    for i, compound in enumerate(optimal["compounds"]):
        opt_stints.append({
            "compound": compound,
            "laps": pit_laps[i+1] - pit_laps[i]
        })
    draw_strategy_bar(ax, opt_stints, 2, 0.6, "OPTIMAL")

    # Actual strategy
    draw_strategy_bar(ax, actual_strategy, 1, 0.6, "ACTUAL")

    # Time difference
    actual_time_approx = sum(
        predict_stint_time(models.get(s["compound"],
                          list(models.values())[0]), s["laps"])
        for s in actual_strategy
        if s["compound"] in models
    ) + (len(actual_strategy) - 1) * DEFAULT_PIT_LOSS

    time_saving = actual_time_approx - optimal["total_time"]

    ax.set_xlim(0, total_laps + 2)
    ax.set_ylim(0.3, 2.7)
    ax.set_xlabel("Lap Number", color="#AAAAAA", fontsize=10)
    ax.set_title(
        f"Optimal vs Actual Strategy — {driver}  "
        f"({'Model suggests saving ~' + format_time(abs(time_saving)) if time_saving > 0 else 'Actual was near-optimal!'})",
        color="#FFFFFF", fontsize=11, fontweight="bold")
    ax.set_yticks([])
    ax.grid(axis="x", color="#1E1E1E", linewidth=0.6)

    # Compound legend
    patches = [mpatches.Patch(color=COMPOUND_COLORS[c], label=c)
               for c in ["SOFT", "MEDIUM", "HARD"] if c in models]
    ax.legend(handles=patches, loc="upper right",
              facecolor="#1A1A1A", edgecolor="#333",
              labelcolor="#DDD", fontsize=9)

    fig.suptitle(f"Tyre Strategy Optimiser — {race_label}  |  {driver}",
                 color="#FFFFFF", fontsize=16, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0A0A0A")
    plt.close()
    print(f"\n  [✓] Saved: {output_path}")


def run(year, race, driver):
    os.makedirs("cache",   exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Tyre Strategy Optimiser: {year} {race} GP — {driver}")
    print(f"{'='*55}")

    print("\n[ 1/4 ] Loading race session...")
    session = load_session(year, race)
    race_label = f"{year} {race} GP"
    total_laps = int(session.laps["LapNumber"].max())
    pit_loss   = PIT_LOSS.get(race, DEFAULT_PIT_LOSS)
    print(f"  Total laps: {total_laps}  |  Pit loss: {pit_loss}s")

    print("\n[ 2/4 ] Extracting stint data & fitting ML models...")
    laps = extract_stint_data(session)
    models = fit_deg_models(laps)

    if len(models) < 2:
        print("  [!] Not enough compound data. Try a different race.")
        return

    compounds_available = list(models.keys())
    print(f"  Compounds modelled: {compounds_available}")

    print("\n[ 3/4 ] Simulating strategies...")
    strategies = simulate_strategies(models, total_laps,
                                     pit_loss, compounds_available)
    print(f"  Simulated {len(strategies):,} strategies")

    print("\n  ── Top 5 Optimal Strategies ──")
    for i, s in enumerate(strategies[:5]):
        print(f"  #{i+1}  {s['strategy']}")
        print(f"       Predicted time: {format_time(s['total_time'])}  "
              f"(+{s['total_time']-strategies[0]['total_time']:.1f}s vs best)")

    print("\n[ 4/4 ] Getting actual strategy & plotting...")
    actual = get_actual_strategy(session, driver)
    actual_str = " -> ".join(str(s["compound"]) + "(" + str(s["laps"]) + "L)" for s in actual)
    print(f"  Actual: {actual_str}")

    output_path = f"outputs/{year}_{race.replace(' ','_')}_strategy_optimiser_{driver}.png"
    plot_optimiser(models, strategies, actual, driver,
                   race_label, total_laps, output_path)

    print(f"\n{'='*55}")
    print(f"  Done! Chart saved to {output_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",   type=int, default=2024)
    parser.add_argument("--race",   type=str, default="Bahrain")
    parser.add_argument("--driver", type=str, default="VER")
    args = parser.parse_args()

    run(args.year, args.race, args.driver.upper())
