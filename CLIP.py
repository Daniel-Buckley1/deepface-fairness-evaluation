from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import open_clip
from PIL import Image
from deepface import DeepFace

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(r"C:\Users\danbu\Desktop\deepfaceanalyser")
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_CSV = BASE_DIR / "deepface_clip_results.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

DEEPFACE_ACTIONS = ["age", "gender", "emotion", "race"]
DEEPFACE_DETECTOR = "opencv"
ENFORCE_DETECTION = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
# =========================================================


def get_group_metadata(folder_name: str) -> Tuple[str, str, int]:
    folder_name = folder_name.lower().strip()

    mapping = {
        "black_women": ("black", "woman", 30),
        "black_men": ("black", "man", 30),
        "white_women": ("white", "woman", 30),
        "white_men": ("white", "man", 30),
        "asian_women": ("asian", "woman", 30),
        "asian_men": ("asian", "man", 30),
    }

    if folder_name not in mapping:
        raise ValueError(f"Unknown folder name: {folder_name}")

    return mapping[folder_name]


def get_clip_prompts(intended_gender: str) -> List[str]:
    if intended_gender == "woman":
        return [
            "a 30 year old black woman",
            "a 30 year old black man",
            "a 30 year old white woman",
            "a 30 year old asian woman",
        ]
    elif intended_gender == "man":
        return [
            "a 30 year old black man",
            "a 30 year old black woman",
            "a 30 year old white man",
            "a 30 year old asian man",
        ]
    else:
        raise ValueError(f"Unexpected intended gender: {intended_gender}")


def list_images(root: Path) -> List[Path]:
    image_paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            image_paths.append(p)
    return sorted(image_paths)


def init_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        device=DEVICE,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return model, preprocess, tokenizer


def compute_clip_scores(
    image_path: Path,
    prompts: List[str],
    model,
    preprocess,
    tokenizer,
) -> Dict[str, float]:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    text = tokenizer(prompts).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(0).tolist()

    out: Dict[str, float] = {}
    for prompt, score in zip(prompts, similarities):
        col_name = "clip_" + prompt.replace(" ", "_")
        out[col_name] = float(score)
    return out


def get_top_clip_result(clip_scores: Dict[str, float]) -> Tuple[str, float, float]:
    items = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
    top_key, top_score = items[0]
    second_score = items[1][1] if len(items) > 1 else top_score
    margin = top_score - second_score
    return top_key, top_score, margin


def normalise_deepface_gender(raw_gender: str | None) -> str:
    if not raw_gender:
        return ""
    g = str(raw_gender).strip().lower()
    if g in {"man", "male"}:
        return "man"
    if g in {"woman", "female"}:
        return "woman"
    return g


def normalise_deepface_race(raw_race: str | None) -> str:
    if not raw_race:
        return ""
    r = str(raw_race).strip().lower()

    # Normalise common DeepFace outputs
    mapping = {
        "latino hispanic": "latino_hispanic",
        "middle eastern": "middle_eastern",
        "asian": "asian",
        "black": "black",
        "white": "white",
        "indian": "indian",
    }
    return mapping.get(r, r.replace(" ", "_"))


def run_deepface(image_path: Path) -> Dict[str, object]:
    result = DeepFace.analyze(
        img_path=str(image_path),
        actions=DEEPFACE_ACTIONS,
        detector_backend=DEEPFACE_DETECTOR,
        enforce_detection=ENFORCE_DETECTION,
        silent=True,
    )

    if isinstance(result, list):
        result = result[0]

    age = result.get("age")
    gender = normalise_deepface_gender(result.get("dominant_gender"))
    race = normalise_deepface_race(result.get("dominant_race"))
    emotion = result.get("dominant_emotion", "")

    return {
        "deepface_age": age,
        "deepface_gender": gender,
        "deepface_race": race,
        "deepface_emotion": emotion,
    }


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    model, preprocess, tokenizer = init_clip()
    image_paths = list_images(DATASET_DIR)

    if not image_paths:
        raise FileNotFoundError(f"No images found under {DATASET_DIR}")

    rows: List[Dict[str, object]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        group = image_path.parent.name
        intended_race, intended_gender, intended_age = get_group_metadata(group)
        prompts = get_clip_prompts(intended_gender)

        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name} ({group})")

        row: Dict[str, object] = {
            "image_path": str(image_path),
            "file_name": image_path.name,
            "group": group,
            "intended_race": intended_race,
            "intended_gender": intended_gender,
            "intended_age": intended_age,
        }

        # DeepFace
        try:
            deepface_result = run_deepface(image_path)
            row.update(deepface_result)
        except Exception as e:
            row["deepface_age"] = ""
            row["deepface_gender"] = ""
            row["deepface_race"] = ""
            row["deepface_emotion"] = ""
            row["deepface_error"] = str(e)

        # DeepFace correctness flags
        df_gender = row.get("deepface_gender", "")
        df_race = row.get("deepface_race", "")
        row["deepface_gender_correct"] = int(df_gender == intended_gender) if df_gender else ""
        row["deepface_race_correct"] = int(df_race == intended_race) if df_race else ""

        # CLIP
        try:
            clip_scores = compute_clip_scores(
                image_path=image_path,
                prompts=prompts,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
            )
            row.update(clip_scores)

            top_key, top_score, margin = get_top_clip_result(clip_scores)
            row["clip_top_prompt"] = top_key.replace("clip_", "").replace("_", " ")
            row["clip_top_score"] = round(top_score, 6)
            row["clip_margin"] = round(margin, 6)

        except Exception as e:
            for prompt in prompts:
                col_name = "clip_" + prompt.replace(" ", "_")
                row[col_name] = ""
            row["clip_top_prompt"] = ""
            row["clip_top_score"] = ""
            row["clip_margin"] = ""
            row["clip_error"] = str(e)

        rows.append(row)

    # Collect columns dynamically
    fieldnames = sorted({k for row in rows for k in row.keys()})

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. CSV saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()