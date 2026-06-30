"""
src/visualize.py
All charts for the running fatigue project live here.
Each function takes the enriched dataframe from load_data.py and saves a PNG.

To add a new chart:
  1. Write a new plot_X() function below
  2. Call it in generate_all() at the bottom
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

# --- Style ---
sns.set_theme(style="whitegrid", font="DejaVu Sans")
PALETTE = {"easy": "#2a78d6", "workout": "#e34948"}
FIG_DIR = "docs/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    path = f"{FIG_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Chart 1 ─────────────────────────────────────────────────────────────────
def plot_hr_drift(df):
    """
    Bar chart: HR drift % per run, colored by run type.
    Positive = HR rose (fatigue), negative = HR fell (good aerobic control).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE.get(t, "#888") for t in df["run_type"]]
    bars = ax.bar(df["label"], df["hr_drift_pct"], color=colors, edgecolor="none", width=0.6)
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax.set_title("Heart Rate Drift per Run", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("HR drift (%)")
    ax.set_xlabel("Date")

    # Annotate each bar with its value
    for bar, val in zip(bars, df["hr_drift_pct"]):
        ypos = val + 0.5 if val >= 0 else val - 1.5
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=8, color="#444")

    legend = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    _save(fig, "1_hr_drift")


# ── Chart 2 ─────────────────────────────────────────────────────────────────
def plot_pace_decay(df):
    """
    Bar chart: pace decay % per run.
    Positive = slowed down (fatigue), negative = sped up (structured workout).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE.get(t, "#888") for t in df["run_type"]]
    bars = ax.bar(df["label"], df["pace_decay_pct"], color=colors, edgecolor="none", width=0.6)
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax.set_title("Pace Decay per Run  (positive = slowed, negative = sped up)", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Pace decay (%)")
    ax.set_xlabel("Date")

    for bar, val in zip(bars, df["pace_decay_pct"]):
        ypos = val + 0.3 if val >= 0 else val - 1.2
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=8, color="#444")

    legend = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    _save(fig, "2_pace_decay")


# ── Chart 3 ─────────────────────────────────────────────────────────────────
def plot_aerobic_decoupling(df):
    """
    Aerobic decoupling = HR drift minus pace decay.
    A high positive value means HR rose without a matching pace slow-down —
    your body was working harder to hold the same pace. Classic fatigue signal.
    Garmin calls this 'Pw:Hr' for power; we're doing a pace-based equivalent.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE.get(t, "#888") for t in df["run_type"]]
    bars = ax.bar(df["label"], df["aerobic_decoupling"], color=colors, edgecolor="none", width=0.6)
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax.axhline(5, color="#e34948", linewidth=0.8, linestyle=":", alpha=0.6)  # rough threshold
    ax.text(len(df) - 0.5, 5.4, "decoupling threshold (~5%)", fontsize=8, color="#e34948", ha="right")
    ax.set_title("Aerobic Decoupling per Run\n(HR drift − pace decay: high = heart working harder without slowing down)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Decoupling (%)")
    ax.set_xlabel("Date")

    for bar, val in zip(bars, df["aerobic_decoupling"]):
        ypos = val + 0.4 if val >= 0 else val - 1.5
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1f}", ha="center", va="bottom", fontsize=8, color="#444")

    legend = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    _save(fig, "3_aerobic_decoupling")


# ── Chart 4 ─────────────────────────────────────────────────────────────────
def plot_sleep_vs_drift(df):
    """
    Scatter: sleep hours vs HR drift, labeled by date.
    The hypothesis: worse sleep → higher drift. Tests the recovery story.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [PALETTE.get(t, "#888") for t in df["run_type"]]
    ax.scatter(df["sleep_hours"], df["hr_drift_pct"], c=colors, s=80, edgecolors="white", linewidths=0.5, zorder=3)

    for _, row in df.iterrows():
        ax.annotate(row["label"], (row["sleep_hours"], row["hr_drift_pct"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8, color="#555")

    # Trend line (only if enough variance in x)
    if df["sleep_hours"].std() > 0.5:
        m, b = np.polyfit(df["sleep_hours"], df["hr_drift_pct"], 1)
        x_line = np.linspace(df["sleep_hours"].min() - 0.5, df["sleep_hours"].max() + 0.5, 100)
        ax.plot(x_line, m * x_line + b, color="#aaa", linewidth=1, linestyle="--", zorder=1)
        r = df["sleep_hours"].corr(df["hr_drift_pct"])
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=10, color="#666", va="top")

    ax.axhline(0, color="#ddd", linewidth=0.7)
    ax.set_title("Sleep Hours vs Heart Rate Drift", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Sleep hours (night before)")
    ax.set_ylabel("HR drift (%)")
    legend = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    _save(fig, "4_sleep_vs_drift")


# ── Chart 5 ─────────────────────────────────────────────────────────────────
def plot_heat_vs_hr(df):
    """
    Scatter: heat load vs avg HR, bubble sized by RPE.
    Tests: does environmental stress push HR up independently of effort?
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [PALETTE.get(t, "#888") for t in df["run_type"]]
    sizes = (df["rpe"] * 20).clip(20, 200)
    ax.scatter(df["heat_load"], df["avg_hr"], c=colors, s=sizes, alpha=0.75,
               edgecolors="white", linewidths=0.5, zorder=3)

    for _, row in df.iterrows():
        ax.annotate(row["label"], (row["heat_load"], row["avg_hr"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8, color="#555")

    r = df["heat_load"].corr(df["avg_hr"])
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=10, color="#666", va="top")

    ax.set_title("Heat Load vs Avg HR  (bubble size = RPE)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Heat load  (temp °F + UV × 3)")
    ax.set_ylabel("Avg heart rate (bpm)")
    legend = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=legend, fontsize=9)
    _save(fig, "5_heat_vs_hr")


# ── Chart 6 ─────────────────────────────────────────────────────────────────
def plot_efficiency_over_time(df):
    """
    Line chart: efficiency index over time, split by shoe.
    Efficiency = speed / HR * 100. Higher = doing more work per heartbeat.
    Separating by shoe lets you start comparing vomero vs mizuno over time.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    shoe_colors = {"vomero": "#2a78d6", "mizunos": "#e34948", "mizuno": "#e34948"}

    for shoe, group in df.groupby("shoe"):
        color = shoe_colors.get(shoe, "#888")
        ax.plot(group["label"], group["efficiency_index"], marker="o", label=shoe,
                color=color, linewidth=1.8, markersize=6)

    ax.set_title("Efficiency Index Over Time by Shoe\n(speed ÷ HR × 100: higher = more efficient)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Efficiency index")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    _save(fig, "6_efficiency_by_shoe")


# ── Chart 7 ─────────────────────────────────────────────────────────────────
def plot_recovery_vs_decoupling(df):
    """
    Scatter: recovery score vs aerobic decoupling.
    Recovery score = composite of RHR + sleep (normalized 0–1).
    Hypothesis: low recovery → high decoupling (worse aerobic efficiency).
    This is the "did my body hold up" chart.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [PALETTE.get(t, "#888") for t in df["run_type"]]
    ax.scatter(df["recovery_score"], df["aerobic_decoupling"], c=colors, s=80,
               edgecolors="white", linewidths=0.5, zorder=3)

    for _, row in df.iterrows():
        ax.annotate(row["label"], (row["recovery_score"], row["aerobic_decoupling"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8, color="#555")

    if df["recovery_score"].std() > 0.05:
        m, b = np.polyfit(df["recovery_score"], df["aerobic_decoupling"], 1)
        x_line = np.linspace(df["recovery_score"].min() - 0.05, df["recovery_score"].max() + 0.05, 100)
        ax.plot(x_line, m * x_line + b, color="#aaa", linewidth=1, linestyle="--")
        r = df["recovery_score"].corr(df["aerobic_decoupling"])
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=10, color="#666", va="top")

    ax.axhline(0, color="#ddd", linewidth=0.7)
    ax.set_title("Recovery Score vs Aerobic Decoupling", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Recovery score  (0 = worst, 1 = best)")
    ax.set_ylabel("Aerobic decoupling (%)")
    legend = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=legend, fontsize=9)
    _save(fig, "7_recovery_vs_decoupling")


# ── Generate all ─────────────────────────────────────────────────────────────
def generate_all(df):
    print("Generating charts...")
    plot_hr_drift(df)
    plot_pace_decay(df)
    plot_aerobic_decoupling(df)
    plot_sleep_vs_drift(df)
    plot_heat_vs_hr(df)
    plot_efficiency_over_time(df)
    plot_recovery_vs_decoupling(df)
    print("Done. All charts saved to docs/figures/")
