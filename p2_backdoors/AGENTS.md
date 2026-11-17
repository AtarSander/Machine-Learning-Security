# P2 backdoor detection: notes for agents

Task: score 200 test models for backdoors, per (model, class) pair. The competition
metric is ROC-AUC over all model × class pairs. Phase 0 builds shadow models (with
known backdoors, plus clean ones) to validate a detector locally.

Everything below was measured on 2026-11-17, during Phase 0 Steps 0–3.

## Data layout (not in the working dir)

```
data/example/model.py          network definition + load()
data/example/classes.txt       20 class names, line index = label
data/example/clean/images/     400 PNGs, <id>.png
data/example/clean/labels.csv  id,label
data/example/train/models/     6 models, <id>.pt
data/example/train/labels.csv  id,targets,asr,clean_acc
data/test/models/              200 models to score (4.3 GB, ~21 MB each)
```

## Files

- `clean/labels.csv`: 400 rows, exactly 20 per class. Every image is 64×64 RGB.
- `train/labels.csv` (all 6 rows):

  | id | targets | asr | clean_acc |
  |---|---|---|---|
  | 181fa5acd77c | 15 | 0.964 | 0.8365 |
  | 46669109cbf8 | | | 0.8473 |
  | 57c03db99edc | 3 | 0.999 | 0.8469 |
  | 62fb945a3b6f | 7 | 0.581 | 0.8454 |
  | 6e9c713cd671 | | | 0.8412 |
  | ef38420e5181 | | | 0.8481 |

  - A clean model has empty `targets` and empty `asr`.
  - Every poisoned train model has exactly one target. One has ASR 0.581, close to the 0.5 threshold.
  - There is **no trigger-type or attack-mode column**, so the attack types of the train models are unknown.

## Model (`model.py`)

- `load(path, device="cpu")` returns the network in eval mode, float32, on `device`.
- Architecture: CIFAR-style ResNet-18 (3×3 stride-1 `conv1`, no maxpool), 11.18M params.
- Normalization is inside the network, using `mean`/`std` buffers.
- Head: `CosineLinear`, logits = `r.fc.scale` (16) × cos(feature, class weight row).
- Input: (N,3,64,64) in [0,1]. Output: (N,20). Logits on clean images range roughly -3 to 10.
- Checkpoint: a plain state_dict with 124 keys: 104 float16 tensors plus 20 int64 `num_batches_tracked`.
  - Every train model and every test model checked has the same keys and dtypes.

## Findings and their consequences

1. **All models were finetuned independently from one shared base.**
   - Measurement: relative L2 distance `||a-b|| / ||a||` of conv/fc weights. Every pair is 0.33–0.36 apart (6 train models + 3 test models, clean or poisoned). Two independent random inits would be about 1.4 apart.
   - Consequence: shadows finetuned from a clean train model stay much closer to their parent than any real model is to another.
   - Detectors validated on shadows must not use the clean train models as references. A weight-distance feature would leak the parent.
2. **BN running stats differ between models, but `num_batches_tracked` is 0 everywhere.**
   - The real models were trained with BN in train mode, and the counter was zeroed when saving. That is harmless: `momentum=0.1` BN never reads the counter.
   - Shadows keep BN in eval mode, so their BN stats are bit-identical to their parent's.
   - BN-statistics detector features cannot be validated on shadows.
3. **Weights are stored in float16.** Measure a shadow's ASR and clean accuracy on the model reloaded from its saved float16 file, not on the float32 copy in memory.
4. **Name clash:** a `data.py` module next to the `data/` folder imports fine, because a `.py` file wins over a folder without `__init__.py`. It is just confusing to read.
5. **The 400 clean images were held out from the models' training.**
   - The 3 clean train models score 0.80–0.83 on them, close to their reported `clean_acc` of 0.84–0.85. A model that had trained on them would score close to 1.0.
   - Once a shadow is finetuned on split A, A is no longer unseen for that shadow. Only B is.
   - B has 200 images, so an accuracy measured on B has a standard error of about 3 points. Treat smaller differences as noise.

6. **Shadow finetuning settings** (tuned in Step 3: SGD with momentum 0.9, batch 32, BN in eval mode, patch trigger):
   - LR 3e-3 and above make the model collapse to one class.
   - LR 1e-3 costs 10–20 points of B accuracy.
   - **LR 3e-4 with 20 epochs and 10% poison** costs about 3–4 points of B accuracy (parents score about 0.81, shadows 0.77–0.79). It gave patch ASR 0.6–0.93 on 3 different backdoors.
   - 10 epochs instead of 20 gives the same B accuracy, but one of the 3 backdoors stayed at ASR 0.395.
   - 5% poison gives ASR 0.25–0.68, so many such backdoors fail the 0.5 threshold.
   - Accuracy on A reaches 1.000 (memorized).
   - A shadow ends up about 0.01 relative L2 from its parent; real models are about 0.34 apart.
   - Timing: about 1.15 s per epoch per shadow on an Apple-silicon GPU (MPS), so a 20-epoch shadow takes about 25 s.
7. **How a clean model reacts to the triggers** (on B, before any finetuning):
   - Accuracy: 0.815 with no trigger, 0.775 with a patch (seed 0), 0.715 with a blend (seed 0, alpha 0.15). The blend is visibly grainy.
   - It sends at most 8% of triggered images to any single class, so the ASR baseline is close to 0. An ASR well above that comes from the backdoor, not from the trigger itself.

## Code

Run all scripts from `p2_backdoors/`, because the data paths in `data.py` are relative.

- `model.py`: an unchanged copy of `data/example/model.py`, so `from model import load` works.
- `data.py`: path constants, `get_device()` (cuda, then mps, then cpu), `load_clean_images()` and `split_ab(labels)`.
  - `split_ab` uses the fixed `AB_SPLIT_SEED` and returns index tensors, 10 + 10 per class.
  - Also: `clean_train_model_ids()`, `load_train_model(id, device)`, `load_shadow_model(id, device)`.
  - Images are read with PIL + `pil_to_tensor`, because `torchvision.io.read_image` is deprecated in torchvision 0.29.
- `triggers.py`: one function per trigger type. It takes a seed and returns `apply(x)`.
  - The seed fixes everything about the trigger, so every image gets the identical trigger.
  - `apply` contains no randomness, never modifies its input, and follows the device of `x`.
  - Types implemented so far: `patch`, `blend`.
- `make_shadows.py`: constants at the top. Helper functions: `add_poison`, `finetune`, `float16_round_trip`, `accuracy`, `attack_success_rate`.
  - `finetune` never calls `net.train()`, so BatchNorm stays frozen.
  - So far, `main()` only trains one clean shadow and one patch shadow as a demo.
- Conventions:
  - Call `get_device()` once per script and pass `device` down as an argument.
  - No defensive code: wrong assumptions should crash.
