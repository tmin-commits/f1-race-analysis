"""
Qualifying Lap Comparison
=========================
Overlays two drivers' fastest qualifying laps with:
  - Speed trace (km/h vs distance)
  - Throttle & brake traces
  - Gear map
  - Mini-sector colour bands (who is faster where)
  - Cumulative time delta across the lap

Usage:
    python quali_comparison.py
    python quali_comparison.py --year 2024 --race "Monaco" --driver-a VER --driver-b LEC
"""

import argparse
import os
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

fastf1.Cache.enable_cache("cache/")

DRIVER_COLORS = {
    "VER": "#3671C6", "PER": "#3671C6",
    "HAM": "#27F4D2", "RUS": "#27F4D2",
    "LEC": "#E8002D", "SAI": "#E8002D",
    "NOR": "#FF8000", "PIA": "#FF8000",
    "ALO": "#358C75", "STR": "#358C75",
    "OCO": "#B6BABD", "GAS": "#B6BABD",
    "ALB": "#64C4FF", "SAR": "#64C4FF",
    "TSU": "#356CAB", "RIC": "#356CAB",
    "HUL": "#B62FC3", "MAG": "#B62FC3",
    "BOT": "#C92D4B", "ZHO": "#C92D4B",
}

def get_color(driver):
    return DRIVER_COLORS.get(driver.upper(), "#FFFFFF")


def load_quali(year, race):
    session = fastf1.get_session(year, race, "Q")
    session.load(telemetry=True, weather=False, messages=False)
    return session


def get_fastest_lap_telemetry(session, driver):
    lap = session.laps.pick_drivers(driver).pick_fastest()
    tel = lap.get_telemetry().add_distance()
    tel["Driver"] = driver
    return lap, tel


def resample_to_distance(tel, distance_points):
    """Resample telemetry to common distance axis."""
    from scipy.interpolate import interp1d
    out = {}
    for col in ["Speed", "Throttle", "Brake", "nGear", "DRS"]:
        if col in tel.columns:
            f = interp1d(tel["Distance"], tel[col],
                        bounds_error=False, fill_value="extrapolate")
            out[col] = f(distance_points)
    return out


def plot_quali_comparison(year, race, driver_a, driver_b, output_path):
    print(f"\n{'='*55}")
    print(f"  Quali Comparison: {driver_a} vs {driver_b} — {year} {race} GP")
    print(f"{'='*55}\n")

    print("[ 1/3 ] Loading qualifying session...")
    session = load_quali(year, race)
    race_label = f"{year} {race} GP — Qualifying"

    print("[ 2/3 ] Extracting fastest lap telemetry...")
    lap_a, tel_a = get_fastest_lap_telemetry(session, driver_a)
    lap_b, tel_b = get_fastest_lap_telemetry(session, driver_b)

    lap_time_a = lap_a["LapTime"]
    lap_time_b = lap_b["LapTime"]

    print(f"  {driver_a}: {lap_time_a}")
    print(f"  {driver_b}: {lap_time_b}")

    # Common distance axis
    max_dist = min(tel_a["Distance"].max(), tel_b["Distance"].max())
    dist = np.linspace(0, max_dist, 1000)

    data_a = resample_to_distance(tel_a, dist)
    data_b = resample_to_distance(tel_b, dist)

    # Cumulative time delta
    speed_a = np.clip(data_a["Speed"], 1, None)
    speed_b = np.clip(data_b["Speed"], 1, None)
    dt = np.diff(dist) / 1000  # km
    time_a = dt / (speed_a[:-1] / 3600)
    time_b = dt / (speed_b[:-1] / 3600)
    delta = np.cumsum(time_b - time_a)
    delta = np.insert(delta, 0, 0)

    col_a = get_color(driver_a)
    col_b = get_color(driver_b)

    print("[ 3/3 ] Plotting...")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0A0A0A")
    gs = GridSpec(6, 1, figure=fig,
                  height_ratios=[0.5, 3, 1.5, 1.5, 1, 2],
                  hspace=0.08)

    axes = [fig.add_subplot(gs[i]) for i in range(6)]
    for ax in axes:
        ax.set_facecolor("#131313")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2A2A")
        ax.tick_params(colors="#666666", labelsize=9)
        ax.set_xlim(0, max_dist)
        ax.grid(axis="x", color="#1E1E1E", linewidth=0.6)

    # ── ROW 0: Mini sector bands ─────────────────────────────────────────────
    ax = axes[0]
    faster = np.where(delta < 0, 1, 0)  # 1 = driver_a faster
    for i in range(len(dist) - 1):
        color = col_a if faster[i] == 1 else col_b
        ax.axvspan(dist[i], dist[i+1], color=color, alpha=0.85, linewidth=0)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("Faster", color="#666666", fontsize=8, rotation=0,
                  labelpad=35, va="center")

    # Legend patches
    patch_a = mpatches.Patch(color=col_a, label=f"{driver_a} faster")
    patch_b = mpatches.Patch(color=col_b, label=f"{driver_b} faster")
    ax.legend(handles=[patch_a, patch_b], loc="upper right",
              facecolor="#1A1A1A", edgecolor="#333333",
              labelcolor="#CCCCCC", fontsize=9, ncol=2)

    # ── ROW 1: Speed trace ───────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(dist, data_a["Speed"], color=col_a, linewidth=1.8,
            label=f"{driver_a}  ({str(lap_time_a)[10:19]})", zorder=5)
    ax.plot(dist, data_b["Speed"], color=col_b, linewidth=1.8,
            label=f"{driver_b}  ({str(lap_time_b)[10:19]})", zorder=5, alpha=0.85)

    ax.set_ylabel("Speed (km/h)", color="#AAAAAA", fontsize=10)
    ax.legend(facecolor="#1A1A1A", edgecolor="#333333",
              labelcolor="#DDDDDD", fontsize=11, loc="lower right")
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.suptitle(f"Qualifying Lap Comparison — {race_label}",
                 color="#FFFFFF", fontsize=16, fontweight="bold", y=0.98)

    # ── ROW 2: Throttle ──────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(dist, data_a["Throttle"], color=col_a, linewidth=1.2, alpha=0.9)
    ax.plot(dist, data_b["Throttle"], color=col_b, linewidth=1.2, alpha=0.7)
    ax.set_ylabel("Throttle %", color="#AAAAAA", fontsize=9)
    ax.set_ylim(-5, 105)
    ax.set_xticks([])

    # ── ROW 3: Brake ─────────────────────────────────────────────────────────
    ax = axes[3]
    if "Brake" in data_a and "Brake" in data_b:
        ax.fill_between(dist, data_a["Brake"].astype(float) * 100,
                        color=col_a, alpha=0.6, label=driver_a)
        ax.fill_between(dist, data_b["Brake"].astype(float) * 100,
                        color=col_b, alpha=0.4, label=driver_b)
    ax.set_ylabel("Brake", color="#AAAAAA", fontsize=9)
    ax.set_ylim(-5, 115)
    ax.set_xticks([])

    # ── ROW 4: Gear ──────────────────────────────────────────────────────────
    ax = axes[4]
    if "nGear" in data_a:
        ax.plot(dist, data_a["nGear"], color=col_a, linewidth=1.2,
                drawstyle="steps-post")
        ax.plot(dist, data_b["nGear"], color=col_b, linewidth=1.2,
                drawstyle="steps-post", alpha=0.7)
    ax.set_ylabel("Gear", color="#AAAAAA", fontsize=9)
    ax.set_ylim(0, 9)
    ax.set_yticks(range(1, 9))
    ax.set_xticks([])

    # ── ROW 5: Time delta ────────────────────────────────────────────────────
    ax = axes[5]
    above = np.where(delta > 0, delta, 0)
    below = np.where(delta < 0, delta, 0)

    ax.fill_between(dist, above, 0, color=col_b, alpha=0.4,
                    label=f"{driver_b} ahead in sector")
    ax.fill_between(dist, below, 0, color=col_a, alpha=0.4,
                    label=f"{driver_a} ahead in sector")
    ax.plot(dist, delta, color="#FFFFFF", linewidth=1.5, zorder=5)
    ax.axhline(0, color="#444444", linewidth=1)

    # Final delta annotation
    final = delta[-1]
    faster_driver = driver_a if final < 0 else driver_b
    ax.annotate(f"End: {abs(final):.3f}s ({faster_driver} ahead)",
                xy=(dist[-1], final),
                xytext=(-120, 15 if final > 0 else -25),
                textcoords="offset points",
                color="#FFFFFF", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#888888"))

    ax.set_ylabel("Δ Time (s)", color="#AAAAAA", fontsize=9)
    ax.set_xlabel("Distance (m)", color="#AAAAAA", fontsize=10)
    ax.legend(facecolor="#1A1A1A", edgecolor="#333333",
              labelcolor="#DDDDDD", fontsize=9, loc="lower left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.2f}s"))

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0A0A0A")
    plt.close()
    print(f"\n  [✓] Saved: {output_path}")

    # Summary
    gap = abs((lap_time_a - lap_time_b).total_seconds())
    faster_name = driver_a if lap_time_a < lap_time_b else driver_b
    print(f"\n  ── Quali Summary ──")
    print(f"  {driver_a}: {str(lap_time_a)[10:19]}")
    print(f"  {driver_b}: {str(lap_time_b)[10:19]}")
    print(f"  Gap: {gap:.3f}s  ({faster_name} faster)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",     type=int, default=2024)
    parser.add_argument("--race",     type=str, default="Bahrain")
    parser.add_argument("--driver-a", type=str, default="VER")
    parser.add_argument("--driver-b", type=str, default="LEC")
    args = parser.parse_args()

    os.makedirs("cache",   exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    plot_quali_comparison(
        year=args.year,
        race=args.race,
        driver_a=args.driver_a.upper(),
        driver_b=args.driver_b.upper(),
        output_path=f"outputs/{args.year}_{args.race.replace(' ','_')}_quali_{args.driver_a}_vs_{args.driver_b}.png"
    )
