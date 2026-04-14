"""
Adversarial Attack Evaluation of DeepFace Gender Classifier
============================================================
Implements FGSM and PGD attacks against DeepFace's gender classification model.
Measures the minimum perturbation (epsilon) required to flip gender predictions
across demographic groups, enabling comparison of adversarial vulnerability
between groups.

Requirements:
    pip install deepface tensorflow numpy pillow opencv-python matplotlib pandas tqdm

Usage:
    1. Set DATASET_DIR to the root folder containing your group subfolders
       e.g. dataset/asian_men/, dataset/black_women/, etc.
    2. Run: python adversarial_deepface.py
    3. Results saved to: adversarial_results.csv and adversarial_plots/

Author: Generated for MSISS Capstone — DeepFace Fairness Evaluation
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from deepface import DeepFace
from deepface.models.facial_recognition import VGGFace  # noqa — triggers model download

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DATASET_DIR = r"C:\Users\danbu\Desktop\deepfaceanalyser\dataset"  # <-- update if needed

GROUPS = [
    "asian_men",
    "asian_women",
    "black_men",
    "black_women",
    "white_men",
    "white_women",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Adversarial attack parameters
EPSILONS = [2, 4, 8, 16, 32]        # perturbation magnitudes to test (pixel scale 0-255)
PGD_STEPS = 20                       # number of PGD iterations
PGD_ALPHA_RATIO = 0.25               # PGD step size = alpha_ratio * epsilon

# Output
OUTPUT_CSV = "adversarial_results.csv"
PLOT_DIR = "adversarial_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── MODEL EXTRACTION ─────────────────────────────────────────────────────────

def get_gender_model():
    """
    Extract DeepFace's internal Keras gender classification model.
    Uses DeepFace 0.0.96's Gender class directly from the modeling module,
    bypassing the broken build_model() task routing.
    Weights loaded from ~/.deepface/weights/gender_model_weights.h5
    Returns the Keras model and the target image size it expects.
    """
    print("[*] Loading DeepFace gender model...")

    # modeling.Gender is the GenderClient class directly (confirmed via diagnostic)
    from deepface.modules import modeling as df_modeling
    GenderClient = df_modeling.Gender.GenderClient

    # Instantiate: builds VGG architecture and loads pre-trained weights
    gender_wrapper = GenderClient()

    # Extract the underlying Keras model from the wrapper
    if hasattr(gender_wrapper, "model"):
        keras_model = gender_wrapper.model
    elif hasattr(gender_wrapper, "model_build"):
        keras_model = gender_wrapper.model_build
    else:
        keras_model = gender_wrapper

    if not hasattr(keras_model, "input_shape"):
        raise RuntimeError(
            f"Could not extract Keras model. "
            f"Type: {type(keras_model)}, "
            f"Attrs: {[a for a in dir(keras_model) if not a.startswith('_')]}"
        )

    input_shape = keras_model.input_shape
    img_size = (input_shape[1], input_shape[2])

    print(f"    Gender model loaded successfully")
    print(f"    Input size: {img_size}")
    print(f"    Output shape: {keras_model.output_shape}")
    return keras_model, img_size


def preprocess_image(image_path, img_size):
    """
    Load and preprocess an image to match DeepFace gender model expectations.
    Returns a float32 numpy array in [0, 255] range, shape (1, H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size[1], img_size[0]))  # PIL uses (W, H)
    arr = np.array(img, dtype=np.float32)          # shape: (H, W, 3), range [0, 255]
    arr = np.expand_dims(arr, axis=0)              # shape: (1, H, W, 3)
    return arr


def get_model_prediction(keras_model, img_arr):
    """
    Run the gender model on a preprocessed image array.
    Returns predicted label ('Man' or 'Woman') and confidence score.

    DeepFace gender model outputs a 2-element softmax vector.
    Index mapping: [Woman, Man] — confirmed from DeepFace source.
    """
    # Normalise to [0, 1] for the model (DeepFace normalises internally)
    normalised = img_arr / 255.0
    output = keras_model(normalised, training=False).numpy()[0]  # shape: (2,)

    # DeepFace gender label mapping
    labels = ["Woman", "Man"]
    predicted_idx = int(np.argmax(output))
    confidence = float(output[predicted_idx])
    return labels[predicted_idx], confidence, output


# ─── FGSM ATTACK ──────────────────────────────────────────────────────────────

def fgsm_attack(keras_model, img_arr, epsilon, original_label):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).

    Computes a single-step perturbation in the direction of the loss gradient,
    scaled by epsilon. The goal is to flip the predicted gender label.

    Args:
        keras_model:    Keras gender classification model
        img_arr:        Preprocessed image, shape (1, H, W, 3), range [0, 255]
        epsilon:        Perturbation magnitude (pixel scale)
        original_label: 'Man' or 'Woman' — the label we want to move away from

    Returns:
        Perturbed image array (same shape and range as input)
    """
    # Map label to class index
    label_to_idx = {"Woman": 0, "Man": 1}
    target_idx = label_to_idx[original_label]

    img_tensor = tf.Variable(img_arr / 255.0, dtype=tf.float32)
    target = tf.one_hot([target_idx], depth=2)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        output = keras_model(img_tensor, training=False)
        # Maximise loss on the original class → push prediction away from it
        loss = tf.keras.losses.categorical_crossentropy(target, output)

    gradient = tape.gradient(loss, img_tensor)
    perturbation = epsilon * tf.sign(gradient)

    # Apply perturbation and clip to valid pixel range
    perturbed = img_tensor + perturbation / 255.0
    perturbed = tf.clip_by_value(perturbed, 0.0, 1.0)

    return (perturbed.numpy() * 255.0).astype(np.float32)


# ─── PGD ATTACK ───────────────────────────────────────────────────────────────

def pgd_attack(keras_model, img_arr, epsilon, original_label, steps=PGD_STEPS,
               alpha_ratio=PGD_ALPHA_RATIO):
    """
    Projected Gradient Descent attack (Madry et al., 2018).

    Iteratively applies small FGSM steps, projecting the result back into the
    L-infinity ball of radius epsilon around the original image after each step.
    Produces stronger adversarial examples than single-step FGSM.

    Args:
        keras_model:    Keras gender classification model
        img_arr:        Preprocessed image, shape (1, H, W, 3), range [0, 255]
        epsilon:        Maximum perturbation magnitude (pixel scale)
        original_label: Label to move away from
        steps:          Number of iterative gradient steps
        alpha_ratio:    Step size as fraction of epsilon

    Returns:
        Perturbed image array (same shape and range as input)
    """
    label_to_idx = {"Woman": 0, "Man": 1}
    target_idx = label_to_idx[original_label]
    target = tf.one_hot([target_idx], depth=2)

    alpha = (epsilon * alpha_ratio) / 255.0   # step size in normalised scale
    eps_norm = epsilon / 255.0                 # epsilon in normalised scale

    # Normalise original image; initialise perturbation from small random noise
    original_norm = img_arr / 255.0
    x = original_norm + np.random.uniform(-eps_norm, eps_norm, img_arr.shape).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)

    for _ in range(steps):
        x_var = tf.Variable(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_var)
            output = keras_model(x_var, training=False)
            loss = tf.keras.losses.categorical_crossentropy(target, output)

        gradient = tape.gradient(loss, x_var)
        x = x + alpha * tf.sign(gradient).numpy()

        # Project back into epsilon-ball around original image
        x = np.clip(x, original_norm - eps_norm, original_norm + eps_norm)
        x = np.clip(x, 0.0, 1.0)

    return (x * 255.0).astype(np.float32)


# ─── EVALUATION LOOP ──────────────────────────────────────────────────────────

def evaluate_group(group, keras_model, img_size):
    """
    Run FGSM and PGD attacks across all images in a demographic group.
    For each image and each epsilon value, records whether the prediction flips.
    Returns a list of result dictionaries.
    """
    group_dir = os.path.join(DATASET_DIR, group)
    image_files = [
        f for f in os.listdir(group_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    results = []

    for fname in tqdm(image_files, desc=f"  {group}", leave=False):
        fpath = os.path.join(group_dir, fname)

        try:
            img_arr = preprocess_image(fpath, img_size)
        except Exception as e:
            print(f"    [!] Could not load {fname}: {e}")
            continue

        # Get original prediction
        orig_label, orig_conf, orig_output = get_model_prediction(keras_model, img_arr)

        # Determine ground truth gender from group name
        intended_gender = "Man" if group.endswith("_men") else "Woman"
        gender_correct = int(orig_label == intended_gender)

        row_base = {
            "group": group,
            "file": fname,
            "intended_gender": intended_gender,
            "original_prediction": orig_label,
            "original_confidence": round(orig_conf, 4),
            "originally_correct": gender_correct,
        }

        for eps in EPSILONS:
            for attack_name, attack_fn in [
                ("FGSM", fgsm_attack),
                ("PGD", pgd_attack),
            ]:
                try:
                    adv_arr = attack_fn(keras_model, img_arr, eps, orig_label)
                    adv_label, adv_conf, _ = get_model_prediction(keras_model, adv_arr)

                    prediction_flipped = int(adv_label != orig_label)
                    # A "successful" attack flips an originally correct prediction
                    # OR corrects an originally wrong one (both are analytically interesting)
                    corrected = int(
                        gender_correct == 0 and adv_label == intended_gender
                    )

                    row = {
                        **row_base,
                        "attack": attack_name,
                        "epsilon": eps,
                        "adversarial_prediction": adv_label,
                        "adversarial_confidence": round(adv_conf, 4),
                        "prediction_flipped": prediction_flipped,
                        "prediction_corrected": corrected,
                    }
                    results.append(row)

                except Exception as e:
                    print(f"    [!] Attack failed on {fname} (eps={eps}, {attack_name}): {e}")

    return results


# ─── ANALYSIS & PLOTTING ──────────────────────────────────────────────────────

def compute_flip_rates(df):
    """
    For each group, attack type, and epsilon, compute the flip rate:
    proportion of images whose prediction changed under the attack.
    Also compute the correction rate for originally-misclassified images.
    """
    summary = (
        df.groupby(["group", "attack", "epsilon"])
        .agg(
            n_images=("file", "count"),
            flip_rate=("prediction_flipped", "mean"),
            correction_rate=("prediction_corrected", "mean"),
            originally_correct_rate=("originally_correct", "mean"),
        )
        .reset_index()
    )
    summary["flip_rate_pct"] = (summary["flip_rate"] * 100).round(1)
    summary["correction_rate_pct"] = (summary["correction_rate"] * 100).round(1)
    return summary


def plot_flip_rates_by_group(summary, attack_name, output_dir):
    """
    Line plot: flip rate (%) vs epsilon for each demographic group.
    Higher flip rate at lower epsilon = more adversarially vulnerable.
    """
    data = summary[summary["attack"] == attack_name]

    group_colours = {
        "asian_men":    "#2196F3",
        "asian_women":  "#03A9F4",
        "black_men":    "#F44336",
        "black_women":  "#E91E63",
        "white_men":    "#4CAF50",
        "white_women":  "#8BC34A",
    }
    group_labels = {
        "asian_men": "Asian Men", "asian_women": "Asian Women",
        "black_men": "Black Men", "black_women": "Black Women",
        "white_men": "White Men", "white_women": "White Women",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for group in GROUPS:
        gdata = data[data["group"] == group].sort_values("epsilon")
        if gdata.empty:
            continue
        linestyle = "--" if group.endswith("_women") else "-"
        ax.plot(
            gdata["epsilon"],
            gdata["flip_rate_pct"],
            marker="o",
            linestyle=linestyle,
            color=group_colours[group],
            linewidth=2,
            markersize=6,
            label=group_labels[group],
        )

    ax.set_xlabel("Epsilon (perturbation magnitude, pixel scale)", fontsize=12)
    ax.set_ylabel("Prediction Flip Rate (%)", fontsize=12)
    ax.set_title(
        f"{attack_name} Attack: Prediction Flip Rate by Demographic Group\n"
        f"(Higher flip rate = more adversarially vulnerable)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xticks(EPSILONS)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(50, color="grey", linestyle=":", alpha=0.5, label="50% threshold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"flip_rate_{attack_name.lower()}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [+] Saved: {out_path}")


def plot_minimum_epsilon_to_flip(df, output_dir):
    """
    Bar chart: for each group, the mean minimum epsilon at which a prediction
    first flips under PGD. Lower = more vulnerable.
    Groups where no flip occurs at any epsilon are shown at max+1.
    """
    results = []
    for group in GROUPS:
        gdf = df[(df["group"] == group) & (df["attack"] == "PGD")]
        min_epsilons = []
        for fname, fdata in gdf.groupby("file"):
            flipped = fdata[fdata["prediction_flipped"] == 1].sort_values("epsilon")
            if not flipped.empty:
                min_epsilons.append(flipped.iloc[0]["epsilon"])
            else:
                min_epsilons.append(max(EPSILONS) + 8)  # sentinel for "never flipped"
        results.append({
            "group": group,
            "mean_min_epsilon": np.mean(min_epsilons),
            "pct_never_flipped": 100 * sum(e > max(EPSILONS) for e in min_epsilons) / len(min_epsilons),
        })

    res_df = pd.DataFrame(results).sort_values("mean_min_epsilon")

    group_labels = {
        "asian_men": "Asian Men", "asian_women": "Asian Women",
        "black_men": "Black Men", "black_women": "Black Women",
        "white_men": "White Men", "white_women": "White Women",
    }
    colours = ["#E91E63", "#F44336", "#03A9F4", "#2196F3", "#8BC34A", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range(len(res_df)),
        res_df["mean_min_epsilon"],
        color=colours[:len(res_df)],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(res_df)))
    ax.set_xticklabels(
        [group_labels.get(g, g) for g in res_df["group"]],
        rotation=20, ha="right", fontsize=11,
    )
    ax.set_ylabel("Mean Minimum Epsilon to Flip Prediction", fontsize=12)
    ax.set_title(
        "PGD Attack: Mean Minimum Perturbation Required to Flip Gender Prediction\n"
        "(Lower = more adversarially vulnerable)",
        fontsize=13,
    )
    ax.axhline(max(EPSILONS), color="grey", linestyle="--", alpha=0.5,
               label=f"Max tested epsilon ({max(EPSILONS)})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with % never flipped
    for bar, (_, row) in zip(bars, res_df.iterrows()):
        if row["pct_never_flipped"] > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{row['pct_never_flipped']:.0f}% never\nflipped",
                ha="center", va="bottom", fontsize=8, color="grey",
            )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "min_epsilon_to_flip_pgd.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [+] Saved: {out_path}")
    return res_df


def plot_correction_vs_destabilisation(df, output_dir):
    """
    Scatter plot comparing:
      x-axis: flip rate on CORRECTLY classified images (destabilisation)
      y-axis: flip rate on INCORRECTLY classified images (correction)
    At epsilon=8, PGD attack.
    The ideal quadrant: low destabilisation, high correction.
    """
    eps = 8
    attack = "PGD"
    data = df[(df["attack"] == attack) & (df["epsilon"] == eps)]

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

    fig, ax = plt.subplots(figsize=(8, 7))

    for group in GROUPS:
        gdata = data[data["group"] == group]
        correct = gdata[gdata["originally_correct"] == 1]
        incorrect = gdata[gdata["originally_correct"] == 0]

        destab = correct["prediction_flipped"].mean() * 100 if len(correct) > 0 else 0
        correction = incorrect["prediction_flipped"].mean() * 100 if len(incorrect) > 0 else np.nan

        ax.scatter(destab, correction if not np.isnan(correction) else -5,
                   color=colours[group], s=120, zorder=5)
        ax.annotate(
            group_labels[group],
            (destab, correction if not np.isnan(correction) else -5),
            textcoords="offset points", xytext=(8, 4), fontsize=10,
        )

    ax.axhline(50, color="grey", linestyle=":", alpha=0.4)
    ax.axvline(50, color="grey", linestyle=":", alpha=0.4)
    ax.set_xlabel(f"Destabilisation Rate (% correct predictions flipped)\nPGD ε={eps}", fontsize=11)
    ax.set_ylabel(f"Correction Rate (% incorrect predictions flipped)\nPGD ε={eps}", fontsize=11)
    ax.set_title(
        "Adversarial Vulnerability: Destabilisation vs Correction by Group\n"
        "(Ideal: bottom-right = hard to destabilise, easy to correct)",
        fontsize=12,
    )
    ax.set_xlim(-5, 105)
    ax.set_ylim(-10, 105)

    # Shade quadrants
    ax.fill_between([0, 50], [0, 0], [100, 100], alpha=0.04, color="green")   # low destab
    ax.fill_between([50, 100], [0, 0], [100, 100], alpha=0.04, color="red")   # high destab
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "correction_vs_destabilisation.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [+] Saved: {out_path}")


def print_summary_table(summary):
    """Print a clean summary table of flip rates at each epsilon."""
    print("\n" + "=" * 70)
    print("FLIP RATE SUMMARY (% of predictions flipped by attack)")
    print("=" * 70)
    for attack in ["FGSM", "PGD"]:
        print(f"\n  {attack} Attack:")
        pivot = summary[summary["attack"] == attack].pivot(
            index="group", columns="epsilon", values="flip_rate_pct"
        )
        print(pivot.to_string())
    print("=" * 70)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  DeepFace Adversarial Attack Evaluation")
    print("=" * 60)

    # Load the gender model
    keras_model, img_size = get_gender_model()

    # Run attacks across all groups
    all_results = []
    for group in GROUPS:
        group_dir = os.path.join(DATASET_DIR, group)
        if not os.path.isdir(group_dir):
            print(f"[!] Directory not found, skipping: {group_dir}")
            continue
        print(f"\n[*] Processing group: {group}")
        group_results = evaluate_group(group, keras_model, img_size)
        all_results.extend(group_results)
        print(f"    Completed {len([r for r in group_results if r['attack']=='FGSM' and r['epsilon']==EPSILONS[0]])} images")

    if not all_results:
        print("[!] No results generated. Check DATASET_DIR path.")
        return

    # Save raw results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] Raw results saved to: {OUTPUT_CSV}")

    # Compute summary statistics
    summary = compute_flip_rates(df)
    summary.to_csv("adversarial_summary.csv", index=False)
    print(f"[+] Summary saved to: adversarial_summary.csv")

    # Print summary table
    print_summary_table(summary)

    # Generate plots
    print("\n[*] Generating plots...")
    plot_flip_rates_by_group(summary, "FGSM", PLOT_DIR)
    plot_flip_rates_by_group(summary, "PGD", PLOT_DIR)
    min_eps_df = plot_minimum_epsilon_to_flip(df, PLOT_DIR)
    plot_correction_vs_destabilisation(df, PLOT_DIR)

    print("\n[+] Minimum epsilon to flip (PGD):")
    print(min_eps_df.to_string(index=False))

    print("\n[✓] Evaluation complete.")
    print(f"    Results: {OUTPUT_CSV}")
    print(f"    Plots:   {PLOT_DIR}/")


if __name__ == "__main__":
    main()
