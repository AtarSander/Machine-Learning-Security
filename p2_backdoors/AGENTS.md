# P2 backdoor detection: notes for agents

Task: score 200 test models for backdoors, per (model, class) pair. The competition
metric is ROC-AUC over all model × class pairs. Phase 0 builds shadow models (with
known backdoors, plus clean ones) to validate a detector locally.

## The plan in plain words

**Submission:** for each of the 200 test models, 20 numbers, one per class. Each number says how likely it is that this model has a backdoor that sends images to that class. The organizers compute the AUC of these scores against their secret labels: 0.5 = random guessing, 1.0 = perfect.

**The problem:** the test labels are secret, so you can't tell whether a detection idea works without submitting it. The solution is to first build models whose labels we know, and measure detectors on those.

```
PHASE 0  build ~40-50 "shadow" models where we know which have a backdoor
   │       make_shadows.py  → shadows/models/*.pt + shadows/labels.csv (their labels)
   │       evaluate.py      → computes any detector's AUC on the shadows
   ▼
PHASE 1  write detectors: programs that look at one model and output 20 suspicion scores
   │       try an idea → measure its AUC on the shadows with evaluate_detector → keep the best
   ▼
PHASE 2  run the best detector on the 200 test models and submit its scores
```

### Phase 0 steps

| Step | What | Status |
|---|---|---|
| 0 | Inspect the data and models | done |
| 1 | `data.py`: load images, A/B split, load models | done |
| 2 | `triggers.py`: `patch` and `blend` triggers | done |
| 3 | Training recipe for one shadow, plus tuning: LR 3e-4, 20 epochs, 10% poison | done |
| 4 | `make_shadows.py`: loop over N shadows, save them, write `shadows/labels.csv` | done (tested with 3) |
| 5 | `evaluate.py`: AUC overall and per trigger kind / mode | done |
| 6 | More triggers, one at a time: `sig`, `wanet`, `filter`, `lowfreq` | in progress: `sig`, `filter` done; `wanet` left out (finding 11); `lowfreq` next |
| 7 | Attack modes: `targeted` (fires for one source class only) and `clean_label` for SIG | to do |
| 8 | Final run of ~40-50 shadows, runtime estimate, summary | to do |

Steps 6–7 make the shadows realistic: the test models contain all these kinds of backdoors. A detector that is only checked on patches could look good locally and miss everything else.

### Phase 0 design decisions (the user's, keep them)

1. Clean shadows go through exactly the same finetuning as poisoned ones, just without poison. Otherwise a detector could learn "was finetuned by us" instead of "has a backdoor".
2. A/B split of the clean images, 10 + 10 per class. Shadows are finetuned only on A. Detectors evaluated on shadows may use only B, because A is no longer unseen for a shadow.
3. Poisoned images are copies appended to A; the originals stay.
4. BatchNorm stays in eval mode during finetuning: 200 images are too few for new statistics.
5. A shadow counts as valid only if every one of its backdoors reaches ASR ≥ 0.5 on B. ASR is the share of triggered non-target images that go to the target.

### Rules for Phase 1 (from the findings below)

- Evaluate detectors on shadows using only B images.
- Don't use the 3 clean train models as references: shadows sit very close to their parent (finding 1).
- Don't trust BN-statistics features validated on shadows (finding 2).
- Expect a different baseline per class, even in clean models. For example, class 15 (`theater_curtain`) naturally responds to vertical stripes (finding 10), so a trigger-search detector may flag it in clean models.

Everything below was measured on 2026-11-17, during Phase 0 Steps 0–6.

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
8. **First full-loop shadows** (Step 4, settings from finding 6):
   - Saved shadow files have exactly the organizers' format: the same keys, dtypes and shapes, CPU tensors, float16.
   - Their BN running stats are bit-identical to the parent's, which confirms decision 4.
   - Backdoors come out borderline: patch ASR 0.421 (invalid) and 0.584. A shadow with a blend and a patch backdoor got blend ASR 0.674 and patch ASR 0.584 (valid), with clean accuracy 0.760.
   - So a noticeable share of backdoors fails the 0.5 threshold, and shadows with 2–3 backdoors fail more often, because every one of their backdoors must pass. Plan for that when choosing `N_MODELS`.
9. **Local AUC is noisy when there are few positives** (Step 5):
   - For a random detector, the AUC's standard deviation is about sqrt((n_pos + n_neg + 1) / (12 · n_pos · n_neg)), which is close to 1/sqrt(12 · n_pos) when negatives far outnumber positives.
   - With about 25 valid backdoors (about 40 shadows), the overall AUC is ±0.06.
   - With about 6 backdoors per trigger kind, a per-kind AUC is ±0.12, so only large differences between kinds mean anything.
   - This is why `evaluate_detector` reports the number of positives next to every AUC.
10. **The SIG trigger** (Step 6): vertical sine stripes; the seed picks 4–8 cycles and an amplitude of 0.05–0.10.
    - Effect on a clean model on B: seed 0 (8 cycles, amplitude 0.088) drops accuracy from 0.815 to 0.530 and sends 21% of images to class 15, `theater_curtain`, whose folds look like stripes. Seed 1 (4 cycles, amplitude 0.064) drops accuracy only to 0.770.
    - So stripes → `theater_curtain` is a natural shortcut in clean models, not a backdoor.
    - 3 test shadows with one all-to-one SIG backdoor each got ASR 0.305 (invalid), 0.574 and 0.653, with clean accuracy 0.725–0.785. That is as borderline as patch (finding 8).
11. **The WaNet trigger is not learnable with this recipe** (Step 6). It warps every image by a smooth random field; the seed picks a maximum shift of 1–2 px.
    - Measured shifts match the drawn sizes. Away from the border, neighbouring pixels move at most 0.15 px relative to each other, so the field is smooth.
    - A clean model barely reacts: B accuracy goes from 0.815 to 0.800. It's invisible by eye.
    - With the standard recipe (10% poison, 20 epochs, LR 3e-4), 3 test shadows got ASR 0.14–0.16, all invalid.
    - On one backdoor:
      - 20% poison: ASR 0.289.
      - A 2–4 px warp: ASR 0.132.
      - 40 epochs: ASR 0.132.
      - 50% poison: ASR 0.558, but clean accuracy 0.595.
      - 20% poison at LR 1e-3: ASR 0.574, but clean accuracy 0.555.
    - So WaNet only passes 0.5 when clean accuracy drops about 20 points below normal shadows (about 0.78). A detector could then score well by spotting damaged models.
    - Likely cause: with only 200 images, memorizing the 20 poisoned copies one by one is easier for the network than learning the subtle warp.
    - `wanet` stays in `triggers.py` but is not in `TRIGGERS`. Until that changes, local AUC says nothing about WaNet backdoors.
    - Decision (user, Step 6): leave WaNet out for now. Possible later experiment: data augmentation in finetuning, with the trigger applied after augmentation.
12. **The filter trigger** (Step 6, `color_filter` in `triggers.py`, kind name `"filter"`). It applies the same affine color transform to every pixel: `new_rgb = M @ rgb + b`, with `M = I + strength · U(-1,1)` and `b = strength/2 · U(-1,1)`.
    - Strength 0.1–0.2: a clean model keeps 0.735–0.785 B accuracy on filtered images, but 3 test backdoors got ASR 0.284 / 0.453 / 0.347, all invalid.
    - **Strength 0.2–0.4 (chosen):** a clean model keeps 0.53–0.70; it looks like a typical photo filter. The 3 test backdoors got ASR 0.553 / 0.779 / 0.600, all valid, with shadow clean accuracy 0.755–0.775.
    - Strength 0.3–0.6: ASR 0.747 / 0.779 / 0.863, but the filter is heavy (clean model 0.385–0.595).
    - The function is named `color_filter` because `filter` is a Python built-in.

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
  - Types implemented so far: `patch`, `blend`, `sig`, `wanet`, `color_filter`. All except `wanet` are in `TRIGGERS` in `make_shadows.py` (see finding 11). `color_filter` is registered as `"filter"`.
  - `wanet` uses `F.grid_sample`, which runs natively on MPS (it matches CPU to 1e-7, no fallback needed).
- `make_shadows.py`: constants at the top (`N_MODELS`, `CLEAN_FRACTION`, `BACKDOOR_COUNT_PROBS`, `TRIGGERS`, `MIN_ASR`, training settings, `SEED`). `main()` loops over `N_MODELS` shadows. For each one it:
  1. Picks the parent by cycling through the 3 clean train models.
  2. Calls `draw_backdoors`: clean with probability 0.5, otherwise 1–3 backdoors with distinct targets, each with a random trigger kind and seed.
  3. Builds the training set: A plus `poisoned_copies` per backdoor, always made from clean A images.
  4. Calls `finetune`, which never calls `net.train()`, so BatchNorm stays frozen.
  5. Calls `save_shadow` (float16, CPU) to write `shadows/models/<id>.pt`.
  6. Reloads the file and measures ASR and clean accuracy on B.
  7. Calls `write_labels`, which rewrites `shadows/labels.csv` after every shadow.
  - Generated output is ignored by git, except for the small `shadows/labels.csv`: `*.pt` is in `.gitignore`.
- `shadows/labels.csv` (the shadows' labels): columns `id,base,targets,kinds,modes,asr,clean_acc,valid`.
  - Lists are `;`-separated, in the same backdoor order in every column.
  - A clean shadow has empty `targets`, `kinds`, `modes` and `asr`, like the organizers' file.
  - `valid` is 1 when every backdoor has ASR ≥ 0.5 on B; a clean shadow is always 1.
  - The only mode so far is `all_to_one`.
- `evaluate.py`: computes a detector's AUC against `shadows/labels.csv`.
  - `auc(pos, neg)`: the competition metric, computed by comparing every (positive, negative) pair. A tie counts as half.
  - `evaluate_detector(scores, labels_csv)`: `scores` is `{shadow_id: 20 scores}`, where a higher score means that class is more likely a backdoor target.
  - It returns `{group: (auc, n_positives)}` for the groups `all`, `kind=<kind>` and `mode=<mode>`. Each group's positives are compared with all negatives.
  - It skips invalid shadows, and with no positives it crashes with ZeroDivisionError.
- Conventions:
  - Call `get_device()` once per script and pass `device` down as an argument.
  - No defensive code: wrong assumptions should crash.
- Gotcha: `torch.linalg.lstsq` in float32 on CPU returned a wrong solution (error 0.7) for a well-conditioned 722k × 4 problem. Use float64 for such checks.
