"""
Counterfactual Explanation Analysis — Method 2: LIME Saliency Maps
===================================================================
Uses LIME (Local Interpretable Model-agnostic Explanations) to identify
which REGIONS of each face DeepFace's gender classifier relies on.

Unlike adversarial attacks and semantic interventions, LIME produces
spatially interpretable explanations — showing which facial areas
(forehead, jaw, eyes, skin, hair) are most influential in the prediction.

By comparing LIME saliency maps between:
  - Correctly classified vs misclassified images
  - Different demographic groups
  - Male vs female subjects

...we can identify whether the model relies on different facial features
for different groups, which would explain systematic misclassification.

How LIME works here:
  1. Divide the face image into superpixels (coherent image regions)
  2. Generate many perturbed versions by randomly masking superpixels
  3. Run DeepFace gender prediction on each perturbed version
  4. Fit a linear model: which superpixels most influence the prediction?
  5. The coefficients of this linear model are the saliency map

Requirements:
    pip install lime scikit-image deepface tensorflow numpy pillow pandas tqdm matplotlib

Usage:
    python counterfactual_lime.py

Outputs:
    lime_results.csv                — per-image LIME statistics
    lime_saliency_maps/             — individual saliency map images
    counterfactual_plots/           — group comparison plots
"""

import os
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

DATASET_DIR = r"C:\Users\danbu\Desktop\deepfaceanalyser\dataset"

GROUPS = [
    "asian_men", "asian_women",
    "black_men", "black_women",
    "white_men", "white_women",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# LIME settings
LIME_NUM_SAMPLES   = 200   # Perturbed samples per image (higher = more accurate, slower)
LIME_NUM_FEATURES  = 10    # Number of superpixels to explain
IMAGES_PER_GROUP   = 40    # Run on all images per group

OUTPUT_CSV    = "lime_results.csv"
SALIENCY_DIR  = "lime_saliency_maps"
PLOT_DIR      = "counterfactual_plots"
os.makedirs(SALIENCY_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── DEEPFACE MODEL SETUP ─────────────────────────────────────────────────────

_gender_model = None
_img_size = (224, 224)

def load_gender_model():
    """Load DeepFace gender model for direct prediction (needed for LIME)."""
    global _gender_model, _img_size
    if _gender_model is not None:
        return _gender_model, _img_size

    print("[*] Loading gender model...")
    import traceback
    try:
        from deepface.modules import modeling as df_modeling
        GenderClient = df_modeling.Gender.GenderClient
        wrapper = GenderClient()
        keras_model = wrapper.model if hasattr(wrapper, "model") else wrapper
        _img_size = (keras_model.input_shape[1], keras_model.input_shape[2])
        _gender_model = keras_model
        print(f"    Model loaded. Input size: {_img_size}")
        return _gender_model, _img_size
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        traceback.print_exc()
        raise


def predict_gender_proba(images_array):
    """
    Predict gender probability for a batch of images.
    LIME requires a function: (n_images, H, W, C) -> (n_images, n_classes)
    Returns array of shape (n, 2) where columns are [P(Woman), P(Man)]
    """
    model, img_size = load_gender_model()

    # Resize all images to model input size and normalise
    import cv2
    batch = []
    for img in images_array:
        resized = cv2.resize(img.astype(np.uint8), (img_size[1], img_size[0]))
        batch.append(resized.astype(np.float32) / 255.0)

    batch = np.array(batch)
    predictions = model.predict(batch, verbose=0)
    # predictions shape: (n, 2) — [P(Woman), P(Man)] per DeepFace's label order
    return predictions


# Fixed temp file to avoid Windows file locking
_TEMP_IMG_PATH = os.path.join(os.path.expanduser("~"), "deepface_lime_tmp.jpg")


def get_original_prediction_deepface(img_path):
    """Get original prediction using full DeepFace pipeline (for baseline)."""
    from deepface import DeepFace
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


# ─── LIME ANALYSIS ────────────────────────────────────────────────────────────

def run_lime_on_image(img_array: np.ndarray, label_index: int = 1):
    """
    Run LIME on a single image.
    label_index: 0 = Woman, 1 = Man (DeepFace label order)

    Returns:
        explanation  — LIME ImageExplanation object
        top_features — list of (superpixel_id, weight) sorted by importance
        heatmap      — numpy array of pixel-level importance weights
    """
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    explainer = lime_image.LimeImageExplainer(random_state=42)

    explanation = explainer.explain_instance(
        img_array.astype(np.uint8),
        predict_gender_proba,
        top_labels=2,
        hide_color=0,          # masked superpixels become black
        num_samples=LIME_NUM_SAMPLES,
        random_seed=42,
    )

    # Get the explanation for the predicted label
    top_features = explanation.local_exp[label_index]  # list of (segment_id, weight)
    top_features_sorted = sorted(top_features, key=lambda x: abs(x[1]), reverse=True)

    # Build pixel-level heatmap
    segments = explanation.segments  # superpixel assignments, shape (H, W)
    heatmap = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in top_features:
        heatmap[segments == seg_id] = weight

    return explanation, top_features_sorted, heatmap


def save_saliency_map(img_array: np.ndarray, heatmap: np.ndarray,
                       group: str, fname: str, predicted: str, correct: bool):
    """Save a visualisation of the LIME saliency map overlaid on the original image."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    axes[0].imshow(img_array.astype(np.uint8))
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    # Saliency heatmap
    vmax = max(abs(heatmap.max()), abs(heatmap.min()), 0.001)
    im = axes[1].imshow(heatmap, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("LIME Saliency Map\n(Blue=supports prediction, Red=opposes)", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay: positive regions (supporting prediction) highlighted
    overlay = img_array.astype(np.float32).copy()
    positive_mask = heatmap > 0
    negative_mask = heatmap < 0
    # Green tint for positive, red tint for negative
    overlay[positive_mask, 1] = np.clip(overlay[positive_mask, 1] * 1.3, 0, 255)
    overlay[negative_mask, 0] = np.clip(overlay[negative_mask, 0] * 1.3, 0, 255)
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("Prediction-Supporting Regions\n(Green=supports, Red=opposes)", fontsize=9)
    axes[2].axis("off")

    correct_str = "✓ Correct" if correct else "✗ Incorrect"
    fig.suptitle(
        f"{group} | {fname} | Predicted: {predicted} | {correct_str}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()

    safe_fname = fname.replace(".", "_")
    out_path = os.path.join(SALIENCY_DIR, f"{group}_{safe_fname}_lime.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ─── SPATIAL ANALYSIS ─────────────────────────────────────────────────────────

def compute_spatial_bias(heatmap: np.ndarray):
    """
    Compute summary statistics about where the model's attention is focused.
    Divides the face into quadrants and computes the proportion of positive
    saliency in each region.

    Returns dict with saliency statistics per facial region.
    """
    H, W = heatmap.shape
    h_mid, w_mid = H // 2, W // 2

    regions = {
        "top_left":     heatmap[:h_mid, :w_mid],    # forehead / left
        "top_right":    heatmap[:h_mid, w_mid:],    # forehead / right
        "bottom_left":  heatmap[h_mid:, :w_mid],    # jaw / chin left
        "bottom_right": heatmap[h_mid:, w_mid:],    # jaw / chin right
        "top_half":     heatmap[:h_mid, :],         # upper face (eyes, forehead)
        "bottom_half":  heatmap[h_mid:, :],         # lower face (nose, mouth, jaw)
        "left_half":    heatmap[:, :w_mid],
        "right_half":   heatmap[:, w_mid:],
    }

    stats = {}
    for name, region in regions.items():
        pos = region[region > 0].sum() if (region > 0).any() else 0
        neg = abs(region[region < 0].sum()) if (region < 0).any() else 0
        total = abs(region).sum() if abs(region).sum() > 0 else 1
        stats[f"saliency_{name}_positive_ratio"] = float(pos / total)
        stats[f"saliency_{name}_mean"] = float(region.mean())

    # Overall statistics
    stats["saliency_positive_total"] = float(heatmap[heatmap > 0].sum())
    stats["saliency_negative_total"] = float(abs(heatmap[heatmap < 0]).sum())
    stats["saliency_max"] = float(heatmap.max())
    stats["saliency_min"] = float(heatmap.min())
    stats["saliency_concentration"] = float(
        np.percentile(np.abs(heatmap), 90) / (np.abs(heatmap).mean() + 1e-6)
    )  # Higher = more concentrated attention

    return stats


# ─── GROUP COMPARISON PLOTS ───────────────────────────────────────────────────

def plot_group_saliency_comparison(df: pd.DataFrame):
    """
    Compare spatial saliency patterns across demographic groups.
    Shows which facial regions the model relies on most per group.
    """
    regions = ["top_half", "bottom_half"]
    region_labels = {"top_half": "Upper Face\n(Eyes/Forehead)", "bottom_half": "Lower Face\n(Jaw/Mouth)"}

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, region in zip(axes, regions):
        col = f"saliency_{region}_positive_ratio"
        if col not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        group_means = df.groupby("group")[col].mean()
        groups_ordered = ["white_men", "white_women", "asian_men",
                          "asian_women", "black_men", "black_women"]

        vals = [group_means.get(g, 0) for g in groups_ordered]
        cols = [colours.get(g, "grey") for g in groups_ordered]
        labels = [group_labels.get(g, g) for g in groups_ordered]

        ax.bar(range(len(groups_ordered)), vals, color=cols, edgecolor="white")
        ax.set_xticks(range(len(groups_ordered)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Proportion of Positive Saliency")
        ax.set_title(f"Model Attention: {region_labels[region]}", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.4, label="Equal split")

    fig.suptitle(
        "Where Does DeepFace Look? Facial Region Saliency by Demographic Group\n"
        "(Higher = more of the model's positive attention is in this region)",
        fontsize=12
    )
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "lime_saliency_regions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out}")


def plot_correct_vs_incorrect_saliency(df: pd.DataFrame):
    """Compare saliency concentration between correctly and incorrectly classified images."""
    if "saliency_concentration" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    group_labels = {
        "asian_men": "Asian Men", "asian_women": "Asian Women",
        "black_men": "Black Men", "black_women": "Black Women",
        "white_men": "White Men", "white_women": "White Women",
    }

    groups_ordered = ["white_men", "white_women", "asian_men",
                      "asian_women", "black_men", "black_women"]
    x = np.arange(len(groups_ordered))
    width = 0.35

    correct_vals   = []
    incorrect_vals = []

    for g in groups_ordered:
        sub = df[df["group"] == g]
        c = sub[sub["originally_correct"] == 1]["saliency_concentration"].mean()
        i = sub[sub["originally_correct"] == 0]["saliency_concentration"].mean()
        correct_vals.append(c if not np.isnan(c) else 0)
        incorrect_vals.append(i if not np.isnan(i) else 0)

    ax.bar(x - width/2, correct_vals,   width, label="Correctly classified",   color="#4CAF50", alpha=0.8)
    ax.bar(x + width/2, incorrect_vals, width, label="Incorrectly classified", color="#F44336", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([group_labels.get(g, g) for g in groups_ordered],
                       rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Saliency Concentration (higher = more focused attention)")
    ax.set_title(
        "Model Attention Concentration: Correct vs Incorrect Predictions\n"
        "(More concentrated attention may indicate more confident, reliable predictions)",
        fontsize=11
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "lime_concentration_correct_vs_incorrect.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  LIME Saliency Map Analysis")
    print("=" * 65)
    print("[*] Starting up...")
    import sys
    sys.stdout.flush()

    # Check LIME is available
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        print("[✓] LIME and scikit-image available")
        sys.stdout.flush()
    except ImportError as e:
        print(f"[!] Missing packages: {e}")
        print("    pip install lime scikit-image")
        return

    # Load model once
    print("[*] Loading model...")
    sys.stdout.flush()
    load_gender_model()
    print("[*] Model ready")
    sys.stdout.flush()

    all_results = []

    for group in GROUPS:
        group_dir = os.path.join(DATASET_DIR, group)
        if not os.path.isdir(group_dir):
            print(f"[!] Directory not found, skipping: {group_dir}")
            continue

        image_files = sorted([
            f for f in os.listdir(group_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])[:IMAGES_PER_GROUP]  # Limit per group for speed

        intended_gender = "Man" if group.endswith("_men") else "Woman"
        label_map = {"Woman": 0, "Man": 1}

        print(f"\n[*] Processing group: {group} ({len(image_files)} images)")

        for fname in tqdm(image_files, desc=f"  {group}", leave=False):
            fpath = os.path.join(group_dir, fname)

            try:
                img = Image.open(fpath).convert("RGB").resize((224, 224))
                img_array = np.array(img)
            except Exception as e:
                print(f"    [!] Could not load {fname}: {e}")
                continue

            # Get original prediction
            orig_gender, orig_conf = get_original_prediction_deepface(fpath)
            if orig_gender is None:
                continue

            originally_correct = int(orig_gender == intended_gender)
            predicted_label_idx = label_map.get(orig_gender, 1)

            row_base = {
                "group":              group,
                "file":               fname,
                "intended_gender":    intended_gender,
                "original_gender":    orig_gender,
                "original_confidence": round(orig_conf, 4) if orig_conf else None,
                "originally_correct": originally_correct,
            }

            # Run LIME
            try:
                explanation, top_features, heatmap = run_lime_on_image(
                    img_array, label_index=predicted_label_idx
                )

                # Save saliency map
                save_saliency_map(
                    img_array, heatmap, group, fname,
                    orig_gender, bool(originally_correct)
                )

                # Compute spatial statistics
                spatial_stats = compute_spatial_bias(heatmap)

                # Top feature weights
                top_weights = [abs(w) for _, w in top_features[:5]]
                while len(top_weights) < 5:
                    top_weights.append(0.0)

                row = {
                    **row_base,
                    "lime_top1_weight": round(top_weights[0], 4),
                    "lime_top3_weight_mean": round(np.mean(top_weights[:3]), 4),
                    "lime_top5_weight_mean": round(np.mean(top_weights[:5]), 4),
                    **{k: round(v, 4) for k, v in spatial_stats.items()},
                }
                all_results.append(row)

            except Exception as e:
                print(f"    [!] LIME failed on {fname}: {e}")
                row = {**row_base, "lime_top1_weight": None}
                all_results.append(row)

    if not all_results:
        print("[!] No results generated.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] Results saved to: {OUTPUT_CSV}")

    print("\n[*] Generating comparison plots...")
    plot_group_saliency_comparison(df)
    plot_correct_vs_incorrect_saliency(df)

    print("\n=== GROUP SUMMARY ===")
    print("\nMean LIME top-1 weight by group (higher = more concentrated attention):")
    if "lime_top1_weight" in df.columns:
        print(df.groupby("group")["lime_top1_weight"].mean().round(4).to_string())

    print("\nMean saliency concentration by group:")
    if "saliency_concentration" in df.columns:
        print(df.groupby("group")["saliency_concentration"].mean().round(3).to_string())

    print(f"\n[✓] Complete.")
    print(f"    Saliency maps: {SALIENCY_DIR}/")
    print(f"    Plots: {PLOT_DIR}/")
    print(f"    Results: {OUTPUT_CSV}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        traceback.print_exc()
