"""
Counterfactual Explanation Analysis — Method 1: Semantic Feature Interventions
===============================================================================
For each image, this script systematically varies interpretable visual attributes
and finds the minimum change required to flip DeepFace's gender prediction.

Unlike adversarial attacks (which use imperceptible pixel noise), these
counterfactuals vary semantically meaningful image properties, producing
interpretable explanations of what the model relies on.

Attributes varied:
  - Brightness       (proxy for perceived skin tone / lighting)
  - Contrast         (proxy for facial feature definition)
  - Saturation       (colour intensity)
  - Sharpness        (image clarity / detail)
  - Horizontal flip  (tests for pose/orientation sensitivity)

For each image and each attribute, we find:
  1. Whether a flip occurs at all
  2. The minimum intensity of change required to flip the prediction
  3. The direction of change (increase/decrease)

This enables comparison of counterfactual difficulty across demographic groups —
answering whether some groups require less semantic change to flip predictions,
indicating proximity to decision boundaries in a meaningful visual sense.

Requirements:
    pip install deepface tensorflow numpy pillow opencv-python pandas tqdm matplotlib

Usage:
    python counterfactual_semantic.py

Outputs:
    counterfactual_semantic_results.csv   — per-image per-attribute results
    counterfactual_semantic_summary.csv   — group-level summary statistics
    counterfactual_plots/                 — visualisation plots
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image, ImageEnhance
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DATASET_DIR = r"C:\Users\danbu\Desktop\deepfaceanalyser\dataset"

GROUPS = [
    "asian_men", "asian_women",
    "black_men", "black_women",
    "white_men", "white_women",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Attribute intervention settings
# Each entry: (attribute_name, enhance_class_or_fn, values_to_test)
# Values < 1.0 = decrease, > 1.0 = increase, 1.0 = original
BRIGHTNESS_LEVELS  = [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0]
CONTRAST_LEVELS    = [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0]
SATURATION_LEVELS  = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0]
SHARPNESS_LEVELS   = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0]

OUTPUT_CSV     = "counterfactual_semantic_results.csv"
SUMMARY_CSV    = "counterfactual_semantic_summary.csv"
PLOT_DIR       = "counterfactual_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── DEEPFACE SETUP ───────────────────────────────────────────────────────────

# Fixed temp file path — reused across calls to avoid Windows file locking
_TEMP_IMG_PATH = os.path.join(os.path.expanduser("~"), "deepface_cf_tmp.jpg")


def get_gender_prediction(image_path_or_pil):
    """
    Get DeepFace gender prediction for an image.
    Accepts either a file path (string) or a PIL Image.
    Returns ('Man' or 'Woman', confidence) or (None, None) on failure.
    """
    from deepface import DeepFace

    if isinstance(image_path_or_pil, str):
        img_path = image_path_or_pil
    else:
        # Save PIL image to fixed temp file — avoids Windows PermissionError
        # on NamedTemporaryFile deletion while DeepFace still holds a handle
        image_path_or_pil.save(_TEMP_IMG_PATH, "JPEG", quality=95)
        img_path = _TEMP_IMG_PATH

    try:
        result = DeepFace.analyze(
            img_path=img_path,
            actions=["gender"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        gender = result["dominant_gender"]
        confidence = result["gender"][gender] / 100.0
        return gender, confidence
    except Exception:
        return None, None


# ─── ATTRIBUTE MANIPULATION ───────────────────────────────────────────────────

def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)

def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)

def apply_saturation(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)

def apply_sharpness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(factor)

def apply_hflip(img: Image.Image, factor: float) -> Image.Image:
    """Horizontal flip — factor=1.0 means flipped, factor=0.0 means original."""
    if factor >= 1.0:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


ATTRIBUTES = [
    ("brightness",  apply_brightness,  BRIGHTNESS_LEVELS),
    ("contrast",    apply_contrast,    CONTRAST_LEVELS),
    ("saturation",  apply_saturation,  SATURATION_LEVELS),
    ("sharpness",   apply_sharpness,   SHARPNESS_LEVELS),
]


# ─── COUNTERFACTUAL SEARCH ────────────────────────────────────────────────────

def find_minimum_counterfactual(img: Image.Image, original_gender: str,
                                 attr_name: str, apply_fn, levels: list):
    """
    For a given attribute, find the minimum change level that flips the
    gender prediction from original_gender to the opposite.

    Returns a dict with:
        flipped          — bool, whether a flip was found
        min_flip_level   — the attribute level at which flip first occurs
        min_flip_delta   — absolute distance from 1.0 (neutral) to flip level
        flip_direction   — 'increase' or 'decrease' relative to neutral
        n_levels_tested  — total levels tested before flip or exhaustion
    """
    neutral_idx = levels.index(1.0) if 1.0 in levels else len(levels) // 2

    # Test in both directions from neutral: first decreasing, then increasing
    # This ensures we find the MINIMUM change in either direction
    decreasing = sorted([l for l in levels if l < 1.0], reverse=True)  # closest to 1.0 first
    increasing = sorted([l for l in levels if l > 1.0])                 # closest to 1.0 first

    best_flip_level = None
    best_flip_delta = float('inf')
    best_direction  = None
    n_tested = 0

    for direction, level_list in [("decrease", decreasing), ("increase", increasing)]:
        for level in level_list:
            n_tested += 1
            modified = apply_fn(img, level)
            pred_gender, _ = get_gender_prediction(modified)

            if pred_gender is None:
                continue

            if pred_gender != original_gender:
                delta = abs(level - 1.0)
                if delta < best_flip_delta:
                    best_flip_delta   = delta
                    best_flip_level   = level
                    best_direction    = direction
                break  # Found flip in this direction — move to other direction

    flipped = best_flip_level is not None

    return {
        "flipped":         flipped,
        "min_flip_level":  best_flip_level,
        "min_flip_delta":  best_flip_delta if flipped else None,
        "flip_direction":  best_direction,
        "n_levels_tested": n_tested,
    }


# ─── MAIN EVALUATION ─────────────────────────────────────────────────────────

def evaluate_group(group: str):
    """Run counterfactual analysis for all images in a demographic group."""
    group_dir = os.path.join(DATASET_DIR, group)
    image_files = [
        f for f in os.listdir(group_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    intended_gender = "Man" if group.endswith("_men") else "Woman"
    results = []

    for fname in tqdm(image_files, desc=f"  {group}", leave=False):
        fpath = os.path.join(group_dir, fname)

        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            print(f"    [!] Could not load {fname}: {e}")
            continue

        # Get original prediction via full DeepFace pipeline
        orig_gender, orig_conf = get_gender_prediction(fpath)
        if orig_gender is None:
            print(f"    [!] DeepFace failed on {fname}")
            continue

        originally_correct = int(orig_gender == intended_gender)

        row_base = {
            "group":              group,
            "file":               fname,
            "intended_gender":    intended_gender,
            "original_gender":    orig_gender,
            "original_confidence": round(orig_conf, 4) if orig_conf else None,
            "originally_correct": originally_correct,
        }

        # Run counterfactual search for each attribute
        for attr_name, apply_fn, levels in ATTRIBUTES:
            cf = find_minimum_counterfactual(
                img, orig_gender, attr_name, apply_fn, levels
            )
            row = {
                **row_base,
                "attribute":       attr_name,
                "flipped":         cf["flipped"],
                "min_flip_level":  cf["min_flip_level"],
                "min_flip_delta":  cf["min_flip_delta"],
                "flip_direction":  cf["flip_direction"],
                "n_levels_tested": cf["n_levels_tested"],
            }
            results.append(row)

    return results


# ─── ANALYSIS & PLOTTING ──────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute group-level counterfactual difficulty statistics."""
    summary_rows = []

    for group in GROUPS:
        for attr in df["attribute"].unique():
            sub = df[(df["group"] == group) & (df["attribute"] == attr)]
            flipped = sub[sub["flipped"] == True]

            row = {
                "group":              group,
                "attribute":          attr,
                "n_images":           len(sub),
                "n_flipped":          len(flipped),
                "flip_rate":          round(len(flipped) / len(sub), 3) if len(sub) > 0 else 0,
                "mean_flip_delta":    round(flipped["min_flip_delta"].mean(), 4) if len(flipped) > 0 else None,
                "median_flip_delta":  round(flipped["min_flip_delta"].median(), 4) if len(flipped) > 0 else None,
                "min_flip_delta":     round(flipped["min_flip_delta"].min(), 4) if len(flipped) > 0 else None,
            }
            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def plot_flip_delta_by_group(summary: pd.DataFrame):
    """
    Bar chart: mean minimum attribute change required to flip prediction,
    by group and attribute. Lower = easier to fool via that attribute.
    """
    attrs = summary["attribute"].unique()
    groups_ordered = ["white_men", "white_women", "asian_men", "asian_women",
                      "black_men", "black_women"]
    group_labels = {
        "asian_men": "Asian Men", "asian_women": "Asian Women",
        "black_men": "Black Men", "black_women": "Black Women",
        "white_men": "White Men", "white_women": "White Women",
    }
    colours = {
        "asian_men": "#2196F3", "asian_women": "#03A9F4",
        "black_men": "#F44336", "black_women": "#E91E63",
        "white_men": "#4CAF50", "white_women": "#8BC34A",
    }

    fig, axes = plt.subplots(1, len(attrs), figsize=(5 * len(attrs), 6), sharey=False)
    if len(attrs) == 1:
        axes = [axes]

    for ax, attr in zip(axes, attrs):
        attr_data = summary[summary["attribute"] == attr]
        vals = []
        labels = []
        cols = []
        for g in groups_ordered:
            row = attr_data[attr_data["group"] == g]
            delta = row["mean_flip_delta"].values[0] if len(row) > 0 else None
            vals.append(delta if delta is not None else 0)
            labels.append(group_labels.get(g, g))
            cols.append(colours.get(g, "grey"))

        bars = ax.bar(range(len(groups_ordered)), vals, color=cols, edgecolor="white")
        ax.set_xticks(range(len(groups_ordered)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(attr.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Min Δ to Flip" if attr == attrs[0] else "")
        ax.grid(axis="y", alpha=0.3)

        # Annotate with flip rate
        sr = summary[(summary["attribute"] == attr)]
        for i, g in enumerate(groups_ordered):
            row = sr[sr["group"] == g]
            if len(row) > 0:
                rate = row["flip_rate"].values[0]
                ax.text(i, vals[i] + 0.01, f"{rate*100:.0f}%",
                        ha="center", va="bottom", fontsize=7, color="grey")

    fig.suptitle(
        "Counterfactual Difficulty by Attribute and Demographic Group\n"
        "(Lower bar = easier to flip prediction; % = flip rate)",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "semantic_counterfactual_difficulty.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out}")


def plot_flip_rate_heatmap(summary: pd.DataFrame):
    """Heatmap of flip rates by group and attribute."""
    pivot = summary.pivot(index="group", columns="attribute", values="flip_rate")

    group_labels = {
        "asian_men": "Asian Men", "asian_women": "Asian Women",
        "black_men": "Black Men", "black_women": "Black Women",
        "white_men": "White Men", "white_women": "White Women",
    }
    pivot.index = [group_labels.get(g, g) for g in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                    fontsize=10, color="black" if val < 0.7 else "white",
                    fontweight="bold")

    plt.colorbar(im, ax=ax, label="Flip Rate")
    ax.set_title(
        "Semantic Counterfactual Flip Rate by Group and Attribute\n"
        "(Higher = easier to fool via that attribute)",
        fontsize=12
    )
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "semantic_flip_rate_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out}")


def print_key_findings(summary: pd.DataFrame, df: pd.DataFrame):
    """Print the most analytically important findings."""
    print("\n" + "=" * 65)
    print("KEY FINDINGS")
    print("=" * 65)

    print("\n1. Overall flip rate by group (averaged across attributes):")
    overall = summary.groupby("group")["flip_rate"].mean().sort_values(ascending=False)
    for g, r in overall.items():
        print(f"   {g:15s}: {r*100:.1f}%")

    print("\n2. Mean minimum delta to flip by group (averaged across attributes):")
    overall_delta = summary.groupby("group")["mean_flip_delta"].mean().sort_values()
    for g, d in overall_delta.items():
        if pd.notna(d):
            print(f"   {g:15s}: {d:.4f}")

    print("\n3. Most effective attribute per group:")
    for group in GROUPS:
        sub = summary[summary["group"] == group].dropna(subset=["mean_flip_delta"])
        if len(sub) > 0:
            best = sub.loc[sub["mean_flip_delta"].idxmin()]
            print(f"   {group:15s}: {best['attribute']:12s} (delta={best['mean_flip_delta']:.3f}, rate={best['flip_rate']*100:.0f}%)")

    print("\n4. Correction rate (originally misclassified images that flipped to correct):")
    misclassified = df[df["originally_correct"] == 0]
    if len(misclassified) > 0:
        for group in GROUPS:
            gm = misclassified[(misclassified["group"] == group)]
            if len(gm) > 0:
                corrected = gm[gm["flipped"] == True]
                # A correction means the flip went to the intended gender
                rate = len(corrected) / len(gm)
                print(f"   {group:15s}: {rate*100:.1f}% correction rate ({len(corrected)}/{len(gm)} misclassified images)")
    print("=" * 65)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  Semantic Counterfactual Explanation Analysis")
    print("=" * 65)

    all_results = []
    for group in GROUPS:
        group_dir = os.path.join(DATASET_DIR, group)
        if not os.path.isdir(group_dir):
            print(f"[!] Directory not found, skipping: {group_dir}")
            continue
        print(f"\n[*] Processing group: {group}")
        results = evaluate_group(group)
        all_results.extend(results)
        print(f"    Completed {len(results) // len(ATTRIBUTES)} images")

    if not all_results:
        print("[!] No results generated. Check DATASET_DIR.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] Raw results saved to: {OUTPUT_CSV}")

    summary = compute_summary(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[+] Summary saved to: {SUMMARY_CSV}")

    print("\n[*] Generating plots...")
    plot_flip_delta_by_group(summary)
    plot_flip_rate_heatmap(summary)

    print_key_findings(summary, df)
    print(f"\n[✓] Complete. Plots saved to: {PLOT_DIR}/")


if __name__ == "__main__":
    main()