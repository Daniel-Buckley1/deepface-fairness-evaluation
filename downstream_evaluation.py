"""
Downstream Fairness Evaluation Demonstration
=============================================
This script demonstrates how DeepFace's demographic biases propagate into
downstream fairness evaluations of text-to-image generative models.

Using occupation-based prompts that do not specify race or gender — the kind
used in real fairness audits of generative models — we show that DeepFace's
predicted demographic distribution is systematically distorted by its own
biases, meaning any fairness evaluation built on its labels would produce
a biased picture of the generative model's outputs.

Usage:
    python downstream_evaluation.py

Outputs:
    downstream_results.csv         — per-image DeepFace predictions
    downstream_summary.csv         — demographic distribution summary
    downstream_plots/              — visualisation plots
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DOWNSTREAM_DIR = r"C:\Users\danbu\Desktop\deepfaceanalyser\downstream_dataset"

OCCUPATIONS = [
    "doctor", "engineer", "lawyer", "teacher", "nurse",
    "scientist", "executive", "artist", "police", "chef",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_CSV    = "downstream_results.csv"
SUMMARY_CSV   = "downstream_summary.csv"
PLOT_DIR      = "downstream_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── DEEPFACE EVALUATION ──────────────────────────────────────────────────────

def analyse_image(img_path):
    """Run DeepFace on a single image and return gender and race predictions."""
    from deepface import DeepFace
    try:
        result = DeepFace.analyze(
            img_path=img_path,
            actions=["gender", "race"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        gender = result["dominant_gender"]
        race   = result["dominant_race"]
        gender_conf = result["gender"][gender] / 100.0
        race_conf   = result["race"][race] / 100.0
        return gender, race, round(gender_conf, 4), round(race_conf, 4)
    except Exception:
        return None, None, None, None


def evaluate_downstream_dataset():
    """Run DeepFace on all occupation images and collect results."""
    all_results = []

    for occupation in OCCUPATIONS:
        occ_dir = os.path.join(DOWNSTREAM_DIR, occupation)
        if not os.path.isdir(occ_dir):
            print(f"[!] Directory not found, skipping: {occ_dir}")
            continue

        image_files = [
            f for f in os.listdir(occ_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]

        print(f"\n[*] Processing: {occupation} ({len(image_files)} images)")

        for fname in tqdm(image_files, desc=f"  {occupation}", leave=False):
            fpath = os.path.join(occ_dir, fname)
            gender, race, gender_conf, race_conf = analyse_image(fpath)

            all_results.append({
                "occupation":    occupation,
                "file":          fname,
                "predicted_gender": gender,
                "predicted_race":   race,
                "gender_confidence": gender_conf,
                "race_confidence":   race_conf,
            })

    return pd.DataFrame(all_results)


# ─── ANALYSIS ─────────────────────────────────────────────────────────────────

def compute_demographic_distribution(df):
    """
    Compute the predicted demographic distribution across all images.
    This is what a researcher would see if they used DeepFace to evaluate
    the demographic composition of a generative model's outputs.
    """
    total = len(df.dropna(subset=["predicted_gender"]))

    # Gender distribution
    gender_dist = df["predicted_gender"].value_counts(normalize=True) * 100

    # Race distribution
    race_dist = df["predicted_race"].value_counts(normalize=True) * 100

    # Gender × Race intersection
    df["demographic"] = df["predicted_gender"] + " + " + df["predicted_race"]
    intersectional = df["demographic"].value_counts(normalize=True) * 100

    print("\n" + "=" * 60)
    print("PREDICTED DEMOGRAPHIC DISTRIBUTION")
    print("(What a researcher would see using DeepFace as ground truth)")
    print("=" * 60)

    print(f"\nTotal images analysed: {total}")

    print("\n--- Gender Distribution ---")
    for label, pct in gender_dist.items():
        print(f"  {label:10s}: {pct:.1f}%")

    print("\n--- Race Distribution ---")
    for label, pct in race_dist.items():
        print(f"  {label:20s}: {pct:.1f}%")

    print("\n--- Top 10 Intersectional Categories ---")
    for label, pct in intersectional.head(10).items():
        print(f"  {label:35s}: {pct:.1f}%")

    print("=" * 60)

    return gender_dist, race_dist, intersectional


def compute_occupation_breakdown(df):
    """Show how predicted demographics vary by occupation."""
    print("\n--- Predicted Gender by Occupation ---")
    gender_by_occ = df.groupby("occupation")["predicted_gender"].value_counts(
        normalize=True
    ).unstack(fill_value=0) * 100
    print(gender_by_occ.round(1).to_string())

    print("\n--- Predicted Race by Occupation (top 3) ---")
    for occ in OCCUPATIONS:
        sub = df[df["occupation"] == occ]["predicted_race"].value_counts(normalize=True) * 100
        top3 = sub.head(3)
        top3_str = ", ".join([f"{r}: {p:.0f}%" for r, p in top3.items()])
        print(f"  {occ:12s}: {top3_str}")

    return gender_by_occ


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_gender_distribution(gender_dist):
    """Bar chart of overall predicted gender distribution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colours = ["#2196F3" if g == "Man" else "#E91E63" for g in gender_dist.index]
    bars = ax.bar(gender_dist.index, gender_dist.values, color=colours, edgecolor="white")

    for bar, val in zip(bars, gender_dist.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Percentage of Images (%)", fontsize=12)
    ax.set_title(
        "DeepFace Predicted Gender Distribution\nAcross Neutral Occupation Prompts",
        fontsize=13
    )
    ax.set_ylim(0, 100)
    ax.axhline(50, color="grey", linestyle="--", alpha=0.5, label="50% (equal split)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "downstream_gender_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [+] Saved: {out}")


def plot_race_distribution(race_dist):
    """Bar chart of overall predicted race distribution."""
    colours = {
        "white":          "#4CAF50",
        "black":          "#F44336",
        "asian":          "#2196F3",
        "latino hispanic":"#FF9800",
        "indian":         "#9C27B0",
        "middle eastern": "#00BCD4",
        "other":          "#9E9E9E",
    }
    bar_colours = [colours.get(r.lower(), "#9E9E9E") for r in race_dist.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(race_dist.index, race_dist.values, color=bar_colours, edgecolor="white")

    for bar, val in zip(bars, race_dist.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Percentage of Images (%)", fontsize=12)
    ax.set_title(
        "DeepFace Predicted Race Distribution\nAcross Neutral Occupation Prompts",
        fontsize=13
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "downstream_race_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [+] Saved: {out}")


def plot_gender_by_occupation(gender_by_occ):
    """Stacked bar chart of predicted gender split per occupation."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(gender_by_occ.index))
    width = 0.6

    man_vals   = gender_by_occ.get("Man",   pd.Series([0]*len(gender_by_occ))).values
    woman_vals = gender_by_occ.get("Woman", pd.Series([0]*len(gender_by_occ))).values

    ax.bar(x, man_vals,   width, label="Man",   color="#2196F3", alpha=0.85)
    ax.bar(x, woman_vals, width, bottom=man_vals, label="Woman", color="#E91E63", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [o.capitalize() for o in gender_by_occ.index],
        rotation=20, ha="right", fontsize=11
    )
    ax.set_ylabel("Percentage of Images (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.axhline(50, color="grey", linestyle="--", alpha=0.4)
    ax.set_title(
        "DeepFace Predicted Gender Split by Occupation\n"
        "(Neutral prompts — no gender specified)",
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "downstream_gender_by_occupation.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [+] Saved: {out}")


def plot_intersectional_heatmap(df):
    """
    Heatmap showing the predicted race × gender distribution.
    This is the key plot — it shows which intersectional groups
    DeepFace predicts as present vs absent in the generated images.
    """
    # Create race × gender crosstab
    df_clean = df.dropna(subset=["predicted_gender","predicted_race"])
    crosstab = pd.crosstab(
        df_clean["predicted_race"],
        df_clean["predicted_gender"],
        normalize="all"
    ) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(crosstab.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns, fontsize=13)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels([r.capitalize() for r in crosstab.index], fontsize=11)

    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            val = crosstab.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val > 20 else "black")

    plt.colorbar(im, ax=ax, label="% of total images")
    ax.set_title(
        "DeepFace Predicted Race × Gender Distribution\n"
        "Across Neutral Occupation Prompts\n"
        "(A fair evaluation tool should detect diverse demographics)",
        fontsize=12
    )
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "downstream_intersectional_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [+] Saved: {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Downstream Fairness Evaluation Demonstration")
    print("=" * 60)

    # Run DeepFace on all images
    df = evaluate_downstream_dataset()

    if df.empty:
        print("[!] No results. Check DOWNSTREAM_DIR path.")
        return

    # Save raw results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] Raw results saved to: {OUTPUT_CSV}")

    # Compute and print distributions
    gender_dist, race_dist, intersectional = compute_demographic_distribution(df)
    gender_by_occ = compute_occupation_breakdown(df)

    # Save summary
    summary = pd.DataFrame({
        "category": ["gender"] * len(gender_dist) + ["race"] * len(race_dist),
        "label": list(gender_dist.index) + list(race_dist.index),
        "percentage": list(gender_dist.values) + list(race_dist.values),
    })
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[+] Summary saved to: {SUMMARY_CSV}")

    # Generate plots
    print("\n[*] Generating plots...")
    plot_gender_distribution(gender_dist)
    plot_race_distribution(race_dist)
    plot_gender_by_occupation(gender_by_occ)
    plot_intersectional_heatmap(df)

    print(f"\n[✓] Complete.")
    print(f"    Results: {OUTPUT_CSV}")
    print(f"    Plots:   {PLOT_DIR}/")
    print("\n[*] KEY QUESTION TO CONSIDER:")
    print("    If a researcher used these DeepFace labels to evaluate")
    print("    whether a generative model produces diverse outputs,")
    print("    would they get an accurate picture?")
    print("    Compare the predicted distribution to what you can see")
    print("    in the actual images — that gap IS the downstream harm.")


if __name__ == "__main__":
    main()