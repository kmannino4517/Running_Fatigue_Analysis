"""
src/load_data.py
Loads raw run data and computes derived fatigue/efficiency metrics.
Add new runs to data/runs/raw.csv — this script handles the rest.
"""
 
import pandas as pd
import numpy as np
 
 
def load_runs(path="data/runs/raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
 
    # --- Derived fatigue metrics ---
    # HR drift: did your heart work harder as the run went on?
    df["hr_drift_pct"] = (
        (df["second_hr"] - df["first_hr"]) / df["first_hr"] * 100
    ).round(1)
 
    # Pace decay: did you slow down? (positive = slower, negative = sped up)
    df["pace_decay_pct"] = (
        (df["second_pace"] - df["first_pace"]) / df["first_pace"] * 100
    ).round(1)
 
    # Aerobic decoupling: HR drift without matching pace change
    # High decoupling = body working harder to hold same pace = fatigue signal
    df["aerobic_decoupling"] = (df["hr_drift_pct"] - df["pace_decay_pct"]).round(1)
 
    # Efficiency index: speed per unit of HR — higher is more efficient
    df["speed_mph"] = (60 / df["avg_pace_min_per_mile"]).round(3)
    df["efficiency_index"] = (df["speed_mph"] / df["avg_hr"] * 100).round(3)
 
    # Environmental stress: combined heat/UV proxy
    df["heat_load"] = df["temperature_f"] + df["uv"] * 3
 
    # Recovery score: simple composite (lower RHR + more sleep = better)
    # Normalized so 0 = worst in dataset, 1 = best — updates automatically as you log more
    rhr_norm = 1 - (df["rhr"] - df["rhr"].min()) / (df["rhr"].max() - df["rhr"].min())
    sleep_norm = (df["sleep_hours"] - df["sleep_hours"].min()) / (
        df["sleep_hours"].max() - df["sleep_hours"].min()
    )
    df["recovery_score"] = ((rhr_norm + sleep_norm) / 2).round(2)
 
    # Days since last run (rest/loading context)
    df["days_since_prior"] = df["date"].diff().dt.days
 
    # Run label for chart axes
    df["label"] = df["date"].dt.strftime("%-m/%-d")
 
    return df
 
 
if __name__ == "__main__":
    df = load_runs()
    print(df[["label", "run_type", "hr_drift_pct", "pace_decay_pct",
               "aerobic_decoupling", "efficiency_index", "recovery_score"]].to_string(index=False))
  
