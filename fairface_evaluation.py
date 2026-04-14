"""
Ground Truth Evaluation: DeepFace vs FairFace Verified Labels
=============================================================
This script downloads the FairFace dataset, samples a balanced subset
matching the six demographic groups used in the main study, runs DeepFace
on the real-world images, and compares DeepFace's predictions against
verified ground truth labels.

This directly demonstrates the downstream labelling problem: DeepFace's
predictions on real images with known demographic labels, showing where
and how its errors are distributed across demographic groups.

FairFace reference:
    Karkkainen & Joo (2021). FairFace: Face Attribute Dataset for Balanced
    Race, Gender, and Age. WACV 2021.
    Dataset: https://github.com/joojs/fairface

Setup:
    1. Run this script — it will guide you to download FairFace manually
    2. Place the downloaded files in the fairface/ folder as instructed
    3. Run again to execute the full evaluation

Requirements:
    pip install deepface pandas numpy pillow tqdm matplotlib
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# FairFace will be stored here — create this folder on your desktop
FAIRFACE_DIR   = r"C:\Users\danbu\Desktop\deepfaceanalyser\fairface"
FAIRFACE_IMGS  = os.path.join(FAIRFACE_DIR, "train")   # image folder
FAIRFACE_LABEL = os.path.join(FAIRFACE_DIR, "fairface_label_train.csv")

# How many images to sample per demographic group
# 40 matches your main dataset — increase for more statistical power
IMAGES_PER_GROUP = 40

# Output
OUTPUT_CSV   = "fairface_deepface_results.csv"
SUMMARY_CSV  = "fairface_deepface_summary.csv"
PLOT_DIR     = "fairface_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── FAIRFACE RACE MAPPING ────────────────────────────────────────────────────
# FairFace uses 7 race categories — we map to our 3 study categories
# (White, Black, Asian) to match the main study's demographic groups

FAIRFACE_TO_STUDY = {
    "White":              "White",
    "Black":              "Black",
    "East Asian":         "Asian",
    "Southeast Asian":    "Asian",
    "Indian":             None,      # excluded — not in main study
    "Middle Eastern":     None,      # excluded — not in main study
    "Latino_Hispanic":    None,      # excluded — not in main study
}

FAIRFACE_GENDER_MAP = {
    "Male":   "Man",
    "Female": "Woman",
}

# DeepFace race categories → study categories
DEEPFACE_TO_STUDY = {
    "white":           "White",
    "black":           "Black",
    "asian":           "Asian",
    "latino hispanic": None,
    "middle eastern":  None,
    "indian":          None,
    "other":           None,
}

# ─── SETUP CHECK ──────────────────────────────────────────────────────────────

def check_setup():
    """
    Check if FairFace dataset is present.
    If not, print download instructions and exit.
    """
    if not os.path.isdir(FAIRFACE_DIR):
        os.makedirs(FAIRFACE_DIR, exist_ok=True)

    if not os.path.isfile(FAIRFACE_LABEL) or not os.path.isdir(FAIRFACE_IMGS):
        print("\n" + "=" * 65)
        print("  FAIRFACE DATASET NOT FOUND — DOWNLOAD REQUIRED")
        print("=" * 65)
        print("""
FairFace is a publicly available dataset. Download it as follows:

STEP 1: Go to this Google Drive link:
        https://drive.google.com/drive/folders/1ZX1QmNGgNGu8U3yZZxbTCL1jqB_BNNlM

STEP 2: Download these two items:
        - 'train/'  folder  (contains face images — ~1.2GB)
        - 'fairface_label_train.csv'  (the verified labels file)

STEP 3: Place them here:
        """ + FAIRFACE_DIR + r"""
        So the structure looks like:
        fairface\
            train\
                1.jpg
                2.jpg
                ...
            fairface_label_train.csv

STEP 4: Run this script again.

Note: If the Google Drive link is unavailable, the dataset is also
available at: https://github.com/joojs/fairface
(follow the 'Download' instructions in the README)
""")
        print("=" * 65)
        sys.exit(0)

    print("[✓] FairFace dataset found")
    label_df = pd.read_csv(FAIRFACE_LABEL)
    print(f"    Labels file: {len(label_df)} images")
    print(f"    Race categories: {label_df['race'].unique()}")
    print(f"    Gender categories: {label_df['gender'].unique()}")
    return label_df


# ─── DATASET SAMPLING ─────────────────────────────────────────────────────────

def sample_balanced_dataset(label_df):
    """
    Sample a balanced subset of FairFace images matching the six
    demographic groups used in the main study:
        Asian men, Asian women, Black men, Black women,
        White men, White women

    Returns a DataFrame with sampled image paths and ground truth labels.
    """
    print("\n[*] Sampling balanced dataset...")

    # Map FairFace categories to study categories
    label_df = label_df.copy()
    label_df["study_race"]   = label_df["race"].map(FAIRFACE_TO_STUDY)
    label_df["study_gender"] = label_df["gender"].map(FAIRFACE_GENDER_MAP)

    # Filter to only the three race groups in the study
    label_df = label_df.dropna(subset=["study_race", "study_gender"])

    # Define the six groups
    groups = [
        ("Asian", "Man"),
        ("Asian", "Woman"),
        ("Black", "Man"),
        ("Black", "Woman"),
        ("White", "Man"),
        ("White", "Woman"),
    ]

    sampled_rows = []

    for race, gender in groups:
        subset = label_df[
            (label_df["study_race"]   == race) &
            (label_df["study_gender"] == gender)
        ]

        if len(subset) < IMAGES_PER_GROUP:
            print(f"  [!] Only {len(subset)} images for {race} {gender} "
                  f"(need {IMAGES_PER_GROUP}) — using all available")
            sample = subset
        else:
            sample = subset.sample(n=IMAGES_PER_GROUP, random_state=42)

        sample = sample.copy()
        sample["group"] = f"{race.lower()}_{gender.lower().replace('man','men').replace('woman','women')}"
        sampled_rows.append(sample)
        print(f"  {race} {gender}: {len(sample)} images sampled")

    sampled = pd.concat(sampled_rows, ignore_index=True)

    # Build full image paths
    # FairFace stores images with paths like "train/1.jpg" in the CSV
    def build_path(file_col):
        fname = os.path.basename(str(file_col))
        return os.path.join(FAIRFACE_IMGS, fname)

    # Try different column names for the file path
    file_col = None
    for col in ["file", "image_id", "filename", label_df.columns[0]]:
        if col in label_df.columns:
            file_col = col
            break

    if file_col is None:
        file_col = label_df.columns[0]

    sampled["image_path"] = sampled[file_col].apply(build_path)

    # Verify files exist
    exists = sampled["image_path"].apply(os.path.isfile)
    missing = (~exists).sum()
    if missing > 0:
        print(f"  [!] {missing} image files not found — removing from sample")
        sampled = sampled[exists]

    print(f"\n  Total sampled: {len(sampled)} images across 6 groups")
    return sampled


# ─── DEEPFACE EVALUATION ──────────────────────────────────────────────────────

def run_deepface(image_path):
    """Run DeepFace on a single image, return gender and race predictions."""
    from deepface import DeepFace
    try:
        result = DeepFace.analyze(
            img_path=image_path,
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
    except Exception as e:
        return None, None, None, None


def evaluate_all(sampled_df):
    """Run DeepFace on all sampled images and collect results."""
    results = []

    for group in sampled_df["group"].unique():
        group_df = sampled_df[sampled_df["group"] == group]
        print(f"\n[*] Processing: {group} ({len(group_df)} images)")

        for _, row in tqdm(group_df.iterrows(), total=len(group_df),
                           desc=f"  {group}", leave=False):
            gender_pred, race_pred, gender_conf, race_conf = run_deepface(
                row["image_path"]
            )

            # Map DeepFace race to study category
            race_study = DEEPFACE_TO_STUDY.get(
                race_pred.lower() if race_pred else "", None
            )

            # Determine correctness
            gt_gender = row["study_gender"]
            gt_race   = row["study_race"]

            gender_correct = int(gender_pred == gt_gender) if gender_pred else None
            race_correct   = int(race_study == gt_race) if race_study else 0

            results.append({
                "group":             group,
                "image_path":        row["image_path"],
                "gt_gender":         gt_gender,
                "gt_race":           gt_race,
                "predicted_gender":  gender_pred,
                "predicted_race":    race_pred,
                "predicted_race_study": race_study,
                "gender_confidence": gender_conf,
                "race_confidence":   race_conf,
                "gender_correct":    gender_correct,
                "race_correct":      race_correct,
            })

    return pd.DataFrame(results)


# ─── ANALYSIS ─────────────────────────────────────────────────────────────────

def compute_accuracy(results_df):
    """Compute gender and race accuracy per group."""
    print("\n" + "=" * 65)
    print("DEEPFACE ACCURACY vs FAIRFACE GROUND TRUTH")
    print("=" * 65)

    print("\n--- Gender Classification Accuracy ---")
    gender_acc = results_df.groupby("group")["gender_correct"].agg(
        ["mean", "sum", "count"]
    )
    gender_acc["accuracy_pct"] = (gender_acc["mean"] * 100).round(1)
    print(gender_acc[["sum", "count", "accuracy_pct"]].to_string())

    print("\n--- Race Classification Accuracy ---")
    race_acc = results_df.groupby("group")["race_correct"].agg(
        ["mean", "sum", "count"]
    )
    race_acc["accuracy_pct"] = (race_acc["mean"] * 100).round(1)
    print(race_acc[["sum", "count", "accuracy_pct"]].to_string())

    print("\n--- Gender Misclassification Patterns ---")
    misclassified_gender = results_df[results_df["gender_correct"] == 0]
    print(misclassified_gender.groupby(
        ["group", "predicted_gender"]
    ).size().to_string())

    print("\n--- Race Misclassification Patterns ---")
    misclassified_race = results_df[results_df["race_correct"] == 0]
    print(misclassified_race.groupby(
        ["group", "predicted_race"]
    ).size().to_string())

    print("=" * 65)
    return gender_acc, race_acc


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(gender_acc, race_acc):
    """
    Side-by-side bar chart comparing gender and race accuracy
    across demographic groups — directly comparable to Section 6 results.
    """
    group_labels = {
        "asian_men":   "Asian Men",
        "asian_women": "Asian Women",
        "black_men":   "Black Men",
        "black_women": "Black Women",
        "white_men":   "White Men",
        "white_women": "White Women",
    }
    colours = {
        "asian_men":   "#2196F3",
        "asian_women": "#03A9F4",
        "black_men":   "#F44336",
        "black_women": "#E91E63",
        "white_men":   "#4CAF50",
        "white_women": "#8BC34A",
    }

    groups_ordered = ["white_men","white_women","asian_men",
                      "asian_women","black_men","black_women"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, acc_df, title in zip(
        axes,
        [gender_acc, race_acc],
        ["Gender Classification Accuracy", "Race Classification Accuracy"]
    ):
        vals  = [acc_df.loc[g, "accuracy_pct"] if g in acc_df.index else 0
                 for g in groups_ordered]
        cols  = [colours.get(g, "grey") for g in groups_ordered]
        labels = [group_labels.get(g, g) for g in groups_ordered]

        bars = ax.bar(range(len(groups_ordered)), vals,
                      color=cols, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(range(len(groups_ordered)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_ylim(0, 110)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axhline(50, color="grey", linestyle="--", alpha=0.4,
                   label="Chance level")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "DeepFace Accuracy on FairFace Ground Truth Dataset\n"
        "(Real-world images with verified demographic labels)",
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "fairface_accuracy_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out}")


def plot_gender_gradient(gender_acc):
    """
    Line plot showing the gender accuracy gradient across female groups —
    directly paralleling the intersectional gradient from Section 6.
    """
    female_groups = ["white_women", "asian_women", "black_women"]
    group_labels  = {
        "white_women": "White Women",
        "asian_women": "Asian Women",
        "black_women": "Black Women",
    }
    colours = ["#8BC34A", "#03A9F4", "#E91E63"]

    vals = [gender_acc.loc[g, "accuracy_pct"] if g in gender_acc.index else 0
            for g in female_groups]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [group_labels[g] for g in female_groups],
        vals,
        marker="o", linewidth=2.5, markersize=10,
        color="#333333"
    )
    for i, (g, v) in enumerate(zip(female_groups, vals)):
        ax.scatter(i, v, color=colours[i], s=120, zorder=5)
        ax.text(i, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Gender Classification Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(
        "Intersectional Gender Accuracy Gradient\non Real-World FairFace Images",
        fontsize=13
    )
    ax.axhline(50, color="grey", linestyle="--", alpha=0.4, label="Chance")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "fairface_gender_gradient.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [+] Saved: {out}")


def plot_misclassification_heatmap(results_df):
    """
    Heatmap of predicted vs ground truth gender per group.
    Shows where the errors concentrate.
    """
    groups_ordered = ["white_men","white_women","asian_men",
                      "asian_women","black_men","black_women"]
    group_labels = {
        "asian_men":   "Asian Men",   "asian_women": "Asian Women",
        "black_men":   "Black Men",   "black_women": "Black Women",
        "white_men":   "White Men",   "white_women": "White Women",
    }

    data = []
    for g in groups_ordered:
        sub = results_df[results_df["group"] == g]
        n = len(sub)
        man_pct   = (sub["predicted_gender"] == "Man").sum()   / n * 100
        woman_pct = (sub["predicted_gender"] == "Woman").sum() / n * 100
        data.append([man_pct, woman_pct])

    matrix = np.array(data)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted: Man", "Predicted: Woman"], fontsize=12)
    ax.set_yticks(range(len(groups_ordered)))
    ax.set_yticklabels(
        [group_labels.get(g, g) for g in groups_ordered], fontsize=11
    )

    for i in range(len(groups_ordered)):
        for j in range(2):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.1f}%",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="white" if val > 60 else "black")

    plt.colorbar(im, ax=ax, label="% of group images")
    ax.set_title(
        "DeepFace Gender Prediction Distribution\nby Ground Truth Demographic Group",
        fontsize=12
    )
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "fairface_prediction_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [+] Saved: {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DeepFace Ground Truth Evaluation — FairFace Dataset")
    print("=" * 65)

    # Step 1: Check dataset is present
    label_df = check_setup()

    # Step 2: Print label file columns for debugging
    print(f"\n[*] Label file columns: {label_df.columns.tolist()}")
    print(f"    First few rows:")
    print(label_df.head(3).to_string())

    # Step 3: Sample balanced dataset
    sampled_df = sample_balanced_dataset(label_df)

    if len(sampled_df) == 0:
        print("[!] No images sampled. Check FairFace directory structure.")
        return

    # Step 4: Run DeepFace
    print(f"\n[*] Running DeepFace on {len(sampled_df)} images...")
    print("    (This will take 20-40 minutes)")
    results_df = evaluate_all(sampled_df)

    # Step 5: Save raw results
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] Raw results saved to: {OUTPUT_CSV}")

    # Step 6: Compute and print accuracy
    gender_acc, race_acc = compute_accuracy(results_df)

    # Step 7: Save summary
    summary = pd.DataFrame({
        "group": gender_acc.index,
        "gender_accuracy": gender_acc["accuracy_pct"].values,
        "race_accuracy":   race_acc.reindex(gender_acc.index)["accuracy_pct"].values,
        "n_images":        gender_acc["count"].values,
    })
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[+] Summary saved to: {SUMMARY_CSV}")

    # Step 8: Generate plots
    print("\n[*] Generating plots...")
    plot_accuracy_comparison(gender_acc, race_acc)
    plot_gender_gradient(gender_acc)
    plot_misclassification_heatmap(results_df)

    print(f"\n[✓] Complete.")
    print(f"    Results: {OUTPUT_CSV}")
    print(f"    Plots:   {PLOT_DIR}/")
    print("\n[*] KEY COMPARISON:")
    print("    Compare these accuracy figures to your synthetic dataset results")
    print("    in Section 6. If the intersectional gradient persists on real")
    print("    images with verified labels, that is your strongest finding.")


if __name__ == "__main__":
    main()