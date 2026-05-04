# 🏎️ F1 Race Analysis — Telemetry & Strategy Dashboard

A Python-based motorsport data analysis project using the **FastF1** library to perform the same kind of analyses conducted by F1 engineering teams: tyre degradation modelling, lap delta comparisons, and pit stop strategy visualisation.

Built as part of a self-directed transition into **motorsport data analysis**.

---

## 📊 What It Analyses

### 1. Tyre Degradation
- Extracts all race stint data, filters out safety car laps and pit-in/out laps
- Fits a **linear regression** to lap times per compound to estimate the degradation rate (seconds/lap)
- Visualises scatter + regression lines per compound alongside a bar chart of deg rates

### 2. Lap Time Delta — Driver vs Driver
- Computes the **per-lap time delta** and **cumulative race gap** between any two drivers
- Includes a 5-lap rolling average to smooth out traffic/VSC noise
- Clearly shows when strategy swings or pace differences changed the race outcome

### 3. Pit Stop Strategy Chart
- Reconstructs the **full tyre strategy** of the top N finishers
- Colour-coded by compound (official F1 colours: red/yellow/white)
- Marks pit stop laps with vertical markers and labels stint lengths

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/f1-race-analysis.git
cd f1-race-analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis (defaults to 2024 Bahrain GP, VER vs LEC)
python main.py

# Custom race and drivers
python main.py --year 2024 --race "Monaco" --driver-a VER --driver-b LEC --top-n 10
```

Outputs are saved to the `/outputs/` folder. FastF1 caches data in `/cache/` so subsequent runs are instant.

---

## 🗂️ Project Structure

```
f1-race-analysis/
├── main.py                        # Main runner — run this
├── requirements.txt
├── analysis/
│   ├── tyre_degradation.py        # Deg rate modelling (linear regression)
│   ├── lap_delta.py               # Driver vs driver gap chart
│   └── pit_strategy.py            # Strategy visualisation
├── cache/                         # FastF1 cache (auto-created)
└── outputs/                       # All saved charts (auto-created)
```

---

## 🔧 Key Technical Approaches

| Analysis | Method |
|---|---|
| Outlier removal | Per-driver median filter (removes SC laps, first/last laps of stint) |
| Deg rate | `scipy.stats.linregress` on StintLap vs LapTimeSec |
| Cumulative gap | Rolling sum of per-lap deltas between two drivers |
| Strategy chart | `matplotlib.barh` with stint groupings from FastF1 lap data |

---

## 📈 Example Outputs

> *Run the script to generate your own charts for any 2018–2024 F1 race*

Charts are styled with a dark background consistent with professional telemetry tool aesthetics (MoTeC i2, ATLAS).

---

## 📚 Skills Demonstrated

- Time-series data processing with Pandas
- Statistical modelling (linear regression) on real sensor/timing data
- Data cleaning and outlier filtering (SC laps, pit-in/out laps)
- Multi-subplot, publication-quality visualisation with Matplotlib
- Working with official F1 timing data via FastF1 API

---

## 🔄 Extending This Project

Ideas for future additions:
- Sector time heatmap (Driver × Sector × Lap)
- Tyre compound performance window analysis (optimal stint length per compound)
- Undercut/overcut opportunity detection
- Telemetry overlay: throttle, brake, speed traces for two drivers on the same corner

---

## 📖 References

- [FastF1 Documentation](https://docs.fastf1.dev/)
- Milliken & Milliken — *Race Car Vehicle Dynamics*
- [MoTeC i2 Pro](https://www.motec.com.au/i2/i2overview/)

---

*By Tomin T Thomas Vaidyan — Data Analyst transitioning into Motorsport Engineering*
