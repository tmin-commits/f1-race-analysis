"""
F1 Race Analysis — Main Runner
================================
Run all analyses for a given race in one command.

Usage:
    python main.py
    python main.py --year 2024 --race "Monaco" --driver-a VER --driver-b LEC

Author: Tomin T Thomas Vaidyan
"""

import argparse
import os
import fastf1

# Enable caching (avoids re-downloading data)
fastf1.Cache.enable_cache("cache/")

from analysis.tyre_degradation import load_race, get_stint_laps, plot_degradation
from analysis.lap_delta import plot_delta
from analysis.pit_strategy import plot_strategy


def run_all(year: int, race: str, driver_a: str, driver_b: str, top_n: int = 10):
    os.makedirs("cache", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    race_label = f"{year} {race} GP"

    print(f"\n{'='*55}")
    print(f"  F1 Analysis: {race_label}")
    print(f"{'='*55}\n")

    # ── Load session ──────────────────────────────────────────────────────────
    print("[ 1/3 ] Loading session data...")
    session = load_race(year, race)
    print(f"  [✓] Loaded: {session.event['EventName']} {year}\n")

    # ── Analysis 1: Tyre Degradation ──────────────────────────────────────────
    print("[ 2/3 ] Tyre Degradation Analysis...")
    stint_laps = get_stint_laps(session)
    deg_df = plot_degradation(
        stint_laps,
        race_label=race_label,
        output_path=f"outputs/{year}_{race.replace(' ','_')}_tyre_degradation.png"
    )
    print("\n  Degradation rates:")
    print(deg_df.to_string(index=False))

    # ── Analysis 2: Lap Delta ─────────────────────────────────────────────────
    print(f"\n[ 3/3 ] Lap Delta: {driver_a} vs {driver_b}...")
    try:
        plot_delta(
            session,
            driver_a=driver_a,
            driver_b=driver_b,
            race_label=race_label,
            output_path=f"outputs/{year}_{race.replace(' ','_')}_{driver_a}_vs_{driver_b}.png"
        )
    except Exception as e:
        print(f"  [!] Delta analysis skipped: {e}")

    # ── Analysis 3: Pit Strategy ──────────────────────────────────────────────
    print(f"\n[ 4/4 ] Pit Stop Strategy (Top {top_n})...")
    try:
        plot_strategy(
            session,
            race_label=race_label,
            output_path=f"outputs/{year}_{race.replace(' ','_')}_strategy.png",
            top_n=top_n
        )
    except Exception as e:
        print(f"  [!] Strategy chart skipped: {e}")

    print(f"\n{'='*55}")
    print(f"  All outputs saved to /outputs/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Race Analysis")
    parser.add_argument("--year",     type=int,   default=2024,      help="Season year")
    parser.add_argument("--race",     type=str,   default="Bahrain",  help="Race name")
    parser.add_argument("--driver-a", type=str,   default="VER",     help="Driver A code (e.g. VER)")
    parser.add_argument("--driver-b", type=str,   default="LEC",     help="Driver B code (e.g. LEC)")
    parser.add_argument("--top-n",    type=int,   default=10,        help="Top N finishers for strategy chart")
    args = parser.parse_args()

    run_all(
        year=args.year,
        race=args.race,
        driver_a=args.driver_a.upper(),
        driver_b=args.driver_b.upper(),
        top_n=args.top_n
    )
