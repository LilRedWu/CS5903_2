#!/usr/bin/env python3
"""
CIFAR-10 classification using ai.sooners.us (OpenAI-compatible Chat Completions).

This script:
  • Samples 100 train images (10 per class) from CIFAR-10 with a fixed seed
  • Encodes each image as a base64 JPEG and sends it to the chat completions API
  • Evaluates multiple system prompts, records predictions, computes accuracy
  • Saves confusion matrix plots and raw prediction logs for analysis

Requires:
  pip install requests python-dotenv torch torchvision pillow scikit-learn matplotlib
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision.datasets import CIFAR10

# ── Load secrets ──────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.expanduser("~"), ".soonerai.env"))
API_KEY = os.getenv("SOONERAI_API_KEY")
BASE_URL = os.getenv("SOONERAI_BASE_URL", "https://ai.sooners.us").rstrip("/")
MODEL = os.getenv("SOONERAI_MODEL", "gemma3:4b")

if not API_KEY:
    raise RuntimeError("Missing SOONERAI_API_KEY in ~/.soonerai.env")

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 1337
SAMPLES_PER_CLASS = 10
CLASSES: Sequence[str] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PromptConfig:
    """Configuration for a single system prompt evaluation."""

    name: str
    system_prompt: str
    temperature: float = 0.0


# At least two prompt variants for experimentation.
SYSTEM_PROMPTS: Tuple[PromptConfig, ...] = (
    PromptConfig(
        name="baseline",
        system_prompt=(
            "You are a CIFAR-10 image classifier. "
            "Look carefully at the provided image and respond with exactly one "
            "label from this list: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck."
        ),
    ),
    PromptConfig(
        name="vision_reasoner",
        system_prompt=(
            "You are a meticulous computer vision specialist. "
            "Internally analyze the CIFAR-10 image by describing textures, shapes, and context, "
            "then decide on the best matching label from "
            "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
            "Do not share your reasoning; reply with just the final label."
        ),
    ),
)

# Constrain the model’s output to *one* of the valid labels.
USER_INSTRUCTION = (
    "Classify this CIFAR-10 image. Respond with exactly one label from this list: "
    f"{', '.join(CLASSES)}. Your reply must be just the label, nothing else."
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Encode a PIL image to a base64 JPEG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def post_chat_completion_image(
    image_data_url: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: int = 90,
) -> str:
    """
    Send an image + instruction to /api/chat/completions and return the raw text reply.

    Uses OpenAI-style content parts with an image_url Data URL, which the Sooners API
    accepts for multimodal chat inputs.
    """
    url = f"{base_url}/api/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


SYNONYM_MAP: Dict[str, str] = {
    "car": "automobile",
    "truck": "truck",
    "pickup": "truck",
    "auto": "automobile",
    "plane": "airplane",
    "aeroplane": "airplane",
    "jet": "airplane",
    "helicopter": "airplane",
    "ship": "ship",
    "boat": "ship",
    "vessel": "ship",
    "kitty": "cat",
    "kitten": "cat",
    "puppy": "dog",
    "hound": "dog",
    "canine": "dog",
    "feline": "cat",
    "rodent": "deer",  # fallback to deer for similar comments
}


def normalize_label(text: str) -> str:
    """
    Map the raw model reply to one of the CIFAR-10 classes using simple heuristics.

    Returns a best-effort class label. If no reasonable match is found, returns "__unknown__".
    """
    if not text:
        return "__unknown__"

    normalized = text.strip().lower()
    if normalized in CLASSES:
        return normalized

    # direct synonym or substring hints
    for key, target in SYNONYM_MAP.items():
        if key in normalized:
            return target

    for candidate in CLASSES:
        if candidate in normalized:
            return candidate

    # final fallback: choose the closest class name by character similarity
    try:
        import difflib

        closest = max(
            CLASSES,
            key=lambda label: difflib.SequenceMatcher(None, normalized, label).ratio(),
        )
        return closest
    except Exception:
        return "__unknown__"


# ── Data: stratified sample of 100 images (10/class) ─────────────────────────
def stratified_sample_cifar10(root: str = "./data") -> List[Tuple[int, Image.Image, int]]:
    """
    Download CIFAR-10 (train split) and return a list of triples:
    (dataset_index, PIL_image, target_label_index) with exactly SAMPLES_PER_CLASS per class.
    """
    dataset = CIFAR10(root=root, train=True, download=True)

    # build indices grouped by class
    per_class_indices: Dict[int, List[int]] = {label: [] for label in range(len(CLASSES))}
    for idx, (_, label) in enumerate(dataset):
        per_class_indices[label].append(idx)

    random.seed(SEED)
    torch.manual_seed(SEED)

    selected: List[Tuple[int, Image.Image, int]] = []
    for label in range(len(CLASSES)):
        chosen_indices = random.sample(per_class_indices[label], SAMPLES_PER_CLASS)
        for sample_idx in chosen_indices:
            image, target = dataset[sample_idx]
            selected.append((sample_idx, image, target))

    return selected


def save_sample_manifest(samples: Sequence[Tuple[int, Image.Image, int]], path: Path) -> None:
    """Persist the sampled dataset indices for reproducibility."""
    manifest = [{"dataset_index": idx, "label": CLASSES[label]} for idx, _, label in samples]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": SEED,
                "samples_per_class": SAMPLES_PER_CLASS,
                "total_samples": len(samples),
                "entries": manifest,
            },
            handle,
            indent=2,
        )


def evaluate_prompt(
    prompt: PromptConfig,
    samples: Sequence[Tuple[int, Image.Image, int]],
    model: str,
    base_url: str,
    api_key: str,
) -> Dict[str, object]:
    """
    Run classification for every sample using a specific system prompt.

    Returns a summary dictionary containing metrics and file paths.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    records: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    misclassified: List[Dict[str, object]] = []

    prompt_dir = OUTPUT_DIR / prompt.name
    prompt_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = prompt_dir / "predictions.jsonl"

    print(f"\n=== Evaluating prompt '{prompt.name}' ===")

    for i, (dataset_index, image, target_idx) in enumerate(samples, start=1):
        true_label = CLASSES[target_idx]
        image_data_url = pil_to_base64_jpeg(image)

        try:
            reply = post_chat_completion_image(
                image_data_url=image_data_url,
                system_prompt=prompt.system_prompt,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=prompt.temperature,
            )
        except Exception as exc:
            print(f"[{i:03d}/100] true={true_label:>10s} | pred={'__error__':>10s} | error={exc}")
            normalized = "__unknown__"
            raw_reply = f"__error__: {exc}"
        else:
            normalized = normalize_label(reply)
            raw_reply = reply
            print(f"[{i:03d}/100] true={true_label:>10s} | pred={normalized:>10s} | raw='{reply}'")

        if normalized not in CLASSES:
            errors.append(
                {
                    "sample_idx": i,
                    "dataset_index": dataset_index,
                    "true_label": true_label,
                    "raw_reply": raw_reply,
                }
            )
            # assign an out-of-range label that will be marked incorrect
            predicted_idx = -1
        else:
            predicted_idx = CLASSES.index(normalized)

        y_true.append(target_idx)
        y_pred.append(predicted_idx)

        if predicted_idx != target_idx and predicted_idx >= 0:
            misclassified.append(
                {
                    "sample_idx": i,
                    "dataset_index": dataset_index,
                    "true_label": true_label,
                    "pred_label": normalized,
                    "raw_reply": raw_reply,
                }
            )

        records.append(
            {
                "sample_idx": i,
                "dataset_index": dataset_index,
                "true_label": true_label,
                "pred_label": normalized,
                "raw_reply": raw_reply,
            }
        )

    # compute accuracy treating invalid predictions as incorrect
    correct = sum(1 for true, pred in zip(y_true, y_pred) if pred == true)
    accuracy = correct / len(y_true) if y_true else 0.0

    # For the confusion matrix, map invalid predictions to an extra column.
    cm_predictions = [
        pred if pred in range(len(CLASSES)) else len(CLASSES) for pred in y_pred
    ]
    cm_labels = list(range(len(CLASSES) + 1))
    cm = confusion_matrix(y_true, cm_predictions, labels=cm_labels)

    conf_fig_path = prompt_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, conf_fig_path, prompt.name)

    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")

    if errors:
        errors_path = prompt_dir / "misclassifications.jsonl"
        with errors_path.open("w", encoding="utf-8") as handle:
            for row in errors:
                handle.write(json.dumps(row) + "\n")
    else:
        errors_path = None

    misclassified_path = None
    if misclassified:
        misclassified_path = prompt_dir / "incorrect_predictions.jsonl"
        with misclassified_path.open("w", encoding="utf-8") as handle:
            for row in misclassified:
                handle.write(json.dumps(row) + "\n")

    summary = {
        "prompt_name": prompt.name,
        "accuracy": accuracy,
        "predictions_path": str(predictions_path),
        "confusion_matrix_path": str(conf_fig_path),
        "errors_path": str(errors_path) if errors_path else None,
        "misclassified_path": str(misclassified_path) if misclassified_path else None,
        "invalid_predictions": len(errors),
        "total_misclassified": len(misclassified),
    }

    print(
        f"Prompt '{prompt.name}' accuracy: {accuracy * 100:.2f}% "
        f"({len(errors)} invalid predictions)"
    )
    print(f"Saved confusion matrix to {conf_fig_path}")
    print(f"Saved predictions to {predictions_path}")
    if errors_path:
        print(f"Saved misclassifications to {errors_path}")
    if misclassified_path:
        print(f"Saved incorrect predictions to {misclassified_path}")

    return summary


def plot_confusion_matrix(cm, path: Path, prompt_name: str) -> None:
    """Plot and save the confusion matrix image."""
    plt.figure(figsize=(8, 7))

    # Trim to first 10 columns/rows for display; the last column is invalid predictions.
    display_matrix = cm[: len(CLASSES), : len(CLASSES)]
    invalid_col = cm[: len(CLASSES), len(CLASSES)] if cm.shape[1] > len(CLASSES) else None

    plt.imshow(display_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"CIFAR-10 Confusion Matrix — {prompt_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    plt.yticks(range(len(CLASSES)), CLASSES)

    for r in range(display_matrix.shape[0]):
        for c in range(display_matrix.shape[1]):
            plt.text(c, r, int(display_matrix[r, c]), ha="center", va="center", color="black")

    if invalid_col is not None:
        invalid_total = int(invalid_col.sum())
    else:
        invalid_total = 0

    if invalid_total:
        plt.figtext(
            0.99,
            0.01,
            f"Invalid predictions: {invalid_total}",
            ha="right",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    if not SYSTEM_PROMPTS:
        raise RuntimeError("SYSTEM_PROMPTS must contain at least one prompt definition.")

    print("Preparing CIFAR-10 sample (100 images)...")
    samples = stratified_sample_cifar10()
    save_sample_manifest(samples, OUTPUT_DIR / "sample_manifest.json")

    summaries: List[Dict[str, object]] = []
    for prompt in SYSTEM_PROMPTS:
        summary = evaluate_prompt(prompt, samples, MODEL, BASE_URL, API_KEY)
        summaries.append(summary)

    # Persist summary for downstream analysis
    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    print(f"\nSaved run summary to {summary_path}")

    # Copy best confusion matrix to repo root for convenience
    valid_summaries = [s for s in summaries if s["confusion_matrix_path"]]
    if valid_summaries:
        best = max(valid_summaries, key=lambda item: item["accuracy"])
        src = Path(best["confusion_matrix_path"])
        dst = Path("confusion_matrix.png")
        shutil.copyfile(src, dst)
        print(f"Copied best confusion matrix ({best['prompt_name']}) to {dst}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)
