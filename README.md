# CIFAR-10 Classification with ai.sooners.us

This repo evaluates the `gemma3:4b` vision-language model hosted at [ai.sooners.us](https://ai.sooners.us) on a stratified 100-image slice of CIFAR-10 (10 images per class). Each image is sent as a base64 data URL to the `/api/chat/completions` endpoint, and responses are mapped back to the 10 CIFAR-10 labels. The run produces accuracy metrics, a confusion matrix image, and JSONL logs for further analysis.

## Setup
- Ensure you have Python 3.9+ available.
- Install dependencies:
  ```bash
  python3 -m pip install -r requirements.txt
  ```
- Configure credentials (not checked into version control) in `~/.soonerai.env`:
  ```
  SOONERAI_API_KEY=sk-...
  SOONERAI_BASE_URL=https://ai.sooners.us   # optional, defaults to this value
  SOONERAI_MODEL=gemma3:4b                  # optional override
  ```

## Run
Execute the end-to-end evaluation (downloads CIFAR-10 on first run):
```bash
python3 cifar10_classify.py
```

Artifacts are written to `outputs/<prompt_name>/` and a convenience copy of the best confusion matrix lands at `confusion_matrix.png`. The sampled dataset indices are saved to `outputs/sample_manifest.json` for reproducibility (seed `1337`, 10 images per class).

## Results
| System prompt       | Accuracy | Notes |
|---------------------|----------|-------|
| `baseline`          | 59%      | Straightforward instruction to label the image. Balanced performance overall; excels on `automobile`, `dog`, `horse`. |
| `vision_reasoner`   | 59%      | Encourages silent reasoning before answering. Swapped some errors across classes (better on `airplane`, `truck`; weaker on `dog`). |

The saved `confusion_matrix.png` corresponds to the higher-performing prompt (baseline in this run). Full per-prompt confusion matrices and JSONL prediction logs are available under `outputs/`.

## Analysis
- **Prompt experimentation:** Two system prompts were tested. The reasoning-focused prompt did not improve headline accuracy but changed class-level behavior—gains on `airplane` and `truck` came at the cost of more `dog→cat` confusions. Future work could combine both ideas by explicitly instructing the model to focus on texture cues that separate frogs from furry animals.
- **Common error patterns:** Both prompts frequently mislabel `frog` images as `dog` or `cat`, and `ship` images as `automobile`/`truck`, implying the model leans heavily on color and background context. `airplane` images with dark backgrounds were also confused with birds.
- **Next steps:** Tightening the normalization logic (e.g., handling plural forms) and adding a lightweight retry for low-confidence replies may recover a few additional points. Augmenting the prompt with reminders about amphibian skin texture is worth exploring.

## Reproducibility
- Deterministic sampling configured via `SEED=1337` guarantees the same 100-image slice.
- Predictions, incorrect cases, and the sampling manifest are saved for audit.
- To rerun with different prompts, edit the `SYSTEM_PROMPTS` tuple in `cifar10_classify.py`; artifacts for each prompt are stored separately.
