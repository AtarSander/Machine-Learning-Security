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

### Submission rules (organizers' text, summarized)

- A CSV with one row per test model, in any order. Header `id,c00,...,c19`; column `cNN` is class NN of `classes.txt`.
- The numbers can be anything (probabilities, detector scores, logits). Only their order matters; a larger number means more suspicious.
- The portal rejects the file if an id is missing, repeated, or not a test model, or if any value is not a finite number (nan, inf). **A rejected file still counts toward the daily submission limit.**
- What this means for us:
  - Ids are the test model filenames without `.pt`. Checked: 200 files, all unique, all 12 hex characters. The example ids in the rules (`3fa9c01b77de`, `9b04e6a1c2f0`) are not test models, so they are only illustrations.
  - "Only the order matters" applies to all 4000 numbers together (AUC over all model × class pairs), so scores must be comparable across models, not only within one. `evaluate_detector` pools pairs the same way.
  - Phase 2 must validate the file locally before every upload: exactly the 200 test ids, no duplicates, header `id,c00,...,c19`, 20 finite numbers per row.
  - Possible sources of nan/inf: MAD = 0 in the per-class normalization (Step 3 needs a floor on MAD), and a diverging inversion.
  - Submissions per day are limited, so the local shadow AUC stays the main guide; the leaderboard is not for tuning.
  - The submission rules say nothing about which clean images a detector may use on the test models. B only vs all 400 (finding 5) stays our decision.

### Phase 0 steps

| Step | What | Status |
|---|---|---|
| 0 | Inspect the data and models | done |
| 1 | `data.py`: load images, A/B split, load models | done |
| 2 | `triggers.py`: `patch` and `blend` triggers | done |
| 3 | Training recipe for one shadow, plus tuning: LR 3e-4, 20 epochs, 10% poison | done |
| 4 | `make_shadows.py`: loop over N shadows, save them, write `shadows/labels.csv` | done (tested with 3) |
| 5 | `evaluate.py`: AUC overall and per trigger kind / mode | done |
| 6 | More triggers, one at a time: `sig`, `wanet`, `filter`, `lowfreq` | done: `sig`, `filter` in use; `wanet`, `lowfreq` written but left out (findings 11, 13) |
| 7 | Attack modes: `targeted` (fires for one source class only) and `clean_label` for SIG | done: `targeted` in use (30%); `clean_label` written but off (finding 16) |
| 8 | Final run: 80 shadows, runtime, summary | done: 17 valid poisoned + 47 clean shadows (finding 17) |

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
- Local AUC covers only patch, blend, sig and filter backdoors, in all-to-one and targeted mode, in shadows with 1–2 backdoors. It says nothing about WaNet, low-frequency or clean-label backdoors, or about 3-backdoor models, all of which the test models can contain (findings 14, 16, 17).
- Compare every detector with the "low clean accuracy" baseline (AUC 0.70, finding 18), and check that its scores are not mostly explained by clean accuracy. Beating 0.5 is not enough.
- Expect a different baseline per class, even in clean models. For example, class 15 (`theater_curtain`) naturally responds to vertical stripes (finding 10), so a trigger-search detector may flag it in clean models.
- Shadows overstate how much a backdoor shows on clean images (finding 22). Treat local AUC as an upper bound, and check every detector on the 3 real poisoned train models (`RUN = "train"`). A detector that only works on shadows is not a detector.

### Phase 1 steps

Goal: several detectors, each graded with `evaluate_detector` on the valid shadows, and one combined score that clearly beats the −clean_acc baseline (AUC 0.70, finding 18). No test-set submission (that is Phase 2).

| Step | What | Status |
|---|---|---|
| 0 | Read AGENTS.md and Phase 0 code; propose the detector interface and the results CSV format | done (design agreed; inversion budget ~1 min per model per family) |
| 1 | `run_detector.py` with `random` and `neg_clean_acc`; reproduce AUC ~0.5 and ~0.70 | done: 0.417 (seed 0, within noise) and 0.700 (finding 20) |
| 2 | `detectors.py`: forward-pass statistics per class, on cosines (not softmax) | done: shadow AUC up to 0.98, but shadows overstate the signal (findings 21, 22) |
| 2b | Extra (user request): first test submission, raw `cos_other` on B, to see how the shadow AUC compares with the leaderboard | file ready, `results/test/cos_other.csv`; leaderboard score pending |
| 3 | `scores.py`: per-class median/MAD z-score within one population; raw vs normalized AUC | todo |
| 4 | `inversion.py`: patch inversion (mask + pattern, L1 on mask), 20 targets per batch, B split 5 + 5 per class | todo |
| 5 | `inversion.py`: additive L-inf perturbation, then per-pixel 3×3 color transform | todo |
| 6 | `scores.py`: combine normalized scores (max over detectors + small weight on the mean) | todo |
| 7 | Full run on all valid shadows, final AUC table, summary of missed attack kinds | todo |

Compute rules: measure 2 models before any command that may take more than ~2 minutes; batch everything (all 20 targets together); develop expensive detectors on a fixed 16-shadow dev subset (8 clean + 8 valid poisoned).

### Phase 1 design (agreed in Step 0)

- **Detector interface:** `detect(net, images, labels) -> (20,) float tensor`, higher = more likely a backdoor target.
  - `net` is in eval mode on the device; `images` (N, 3, 64, 64) in [0, 1] and `labels` (N,) are already on the same device. On shadows they are B only (200 images, 10 per class).
  - A detector sees one model only: no model id, no `labels.csv`, no other models. Everything across models (normalization, combining) happens in `scores.py`, on the CSVs.
  - Settings are module constants, not arguments.
  - A detector that needs randomness creates its own `torch.Generator` with a fixed seed inside the call, so a resumed run gives the same scores as an uninterrupted one. Exception: the `random` baseline uses the global RNG. Note that loading a model also consumes the global RNG (`Net()` draws random initial weights before `load_state_dict` overwrites them).
- **Results CSV:** `results/<detector>.csv` (dev-subset runs: `results/dev/<detector>.csv`, train-model runs: `results/train/<detector>.csv`), header `id,c00,...,c19`, one row per model in the order the models were scored, floats written losslessly (no rounding ties). Raw scores only. Rewritten after every model; models already in the file are skipped. When a detector's settings change, delete its CSV. A dev CSV made with the final settings can be copied to `results/` so the full run skips those 16 models.
- **Dev subset** (`DEV_SHADOWS` in `run_detector.py`): 8 of the 47 clean and 8 of the 17 valid poisoned shadows, drawn once with `torch.randperm`, Generator seed 0, then hard-coded.
  - Clean: `shadow003, 022, 046, 050, 056, 058, 066, 069`. Poisoned: `shadow001, 007, 016, 024, 043, 045, 051, 072`.
  - 12 positives: patch 5, filter 4, sig 2, blend 1; all_to_one 6, targeted 6.
  - Weak spots: blend has only 1 positive, and no poisoned dev shadow comes from parent `ef38420e5181`.
  - Its positive share (50% poisoned models) differs from the full set (27%), so compare dev AUCs only with other dev AUCs.
- **Decisions (user, Step 0):**
  - `evaluate_detector` evaluates only the shadows present in `scores` (dev subset, partial runs). `run_detector` prints how many clean / poisoned models were scored, so a missing model is visible.
  - `run_detector` also prints two clean-accuracy checks: the Pearson correlation between each model's max score and its `clean_acc`, and a within-model AUC (the AUC after subtracting each model's median score, which removes any model-level shift; −clean_acc gets exactly 0.5 there).
  - Detectors are also run on the 6 organizer train models (`results/train/<detector>.csv`), reporting where the true target ranks in the 3 poisoned ones (targets 15, 3, 7). This is an extra check on real backdoors (3 positives only), not a reference, so finding 1's rule doesn't apply.
  - `.gitignore`: ignore `shadows/models/` instead of `shadows/`, so `shadows/labels.csv` is tracked. `results/` is ignored entirely (user, after Step 1): score CSVs stay local and are regenerated by rerunning the detectors.
  - Inversion budget: about 1 min per model per inversion family (e.g. 20 targets × 8 images × 100 steps, finding 19). If Step 4 shows more steps are needed, come back with numbers before any long run.
- **Issues raised in Step 0 (analysis; issue 5 is fixed in Step 1):**
  1. Neural Cleanse optimizes a trigger that flips *all* images to the target. A targeted trigger fires only on its source class (leakage median 0.128), and 11 of the 23 positives are targeted. Expect plain NC to miss most of them. Possible fix: a per-source-class loss with a soft-min over source classes, and a score from the best source class. Natural confusions (dog breeds, classes 4–7) make single-source triggers easy in clean models.
  2. Inversion cost (measured, finding 19): per model, 20 targets × images per step × steps × ~3.2 ms. Textbook NC (all 100 optimization images, 300 steps) would take ~32 min per model and needs ~47 GB, so per-step minibatches and few steps are required. A per-model budget for each inversion family is still to be decided by the user.
  3. Per-class normalization on shadows rests on effectively 3 independent models: each shadow is ~0.01 from its parent, real models are ~0.34 apart (finding 1). The test population has 200 independent finetunes, so the normalization gain measured locally may not transfer. Discrete scores (counts, ASR) can have MAD = 0, so the z-score needs a floor on MAD.
  4. Every statistic computed on B partly tracks clean accuracy (finding 18), and per-class normalization does not remove a model-level shift. With 23 positives (±0.06), "clearly beats 0.70" means about 0.80 or more.
  5. `evaluate_detector` indexed `scores[id]` for every valid shadow, so it crashed on a dev-subset or partial CSV. Fixed in Step 1: it now skips shadows missing from `scores`.

Everything below was measured on 2026-11-17, during Phase 0 Steps 0–8.

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
13. **The lowfreq trigger is not learnable with this recipe** (Step 6). It adds the same smooth color pattern to every image: random values on a 4×4 grid per channel, upsampled bicubically, scaled so the strongest point is ± amplitude. Coded amplitude: 0.05–0.10.
    - The pattern is smooth: neighbouring pixels differ by at most 0.01.
    - A clean model barely reacts, even when the pattern is clearly visible. Accuracy on triggered B images: 0.79–0.80 at amplitude 0.05–0.10, 0.765–0.77 at 0.10–0.20, 0.725–0.755 at 0.15–0.30. No class gets more than 8.5%.
    - 3 test shadows per amplitude:
      - 0.05–0.10: ASR 0.105 / 0.168 / 0.105.
      - 0.10–0.20: ASR 0.305 / 0.274 / 0.195.
      - 0.15–0.30: ASR 0.474 / 0.400 / 0.311.
      - All invalid. At 0.15–0.30 the pattern is already clearly visible as colored stains.
    - Likely cause: the network learned to ignore smooth, gradual color changes (they look like lighting), and 20 poisoned copies don't overcome that.
    - `lowfreq` stays in `triggers.py` but is not in `TRIGGERS`.
14. **General limitation of the shadows:** finetuning on 200 images with 20 copies implants only salient triggers (patch, blend, sig, strong filter). Subtle ones (WaNet, lowfreq, weak filter) and clean-label SIG don't reach ASR 0.5. So local AUC measures detection of salient backdoors only, while the test models also contain subtle ones.
15. **Targeted mode** (Step 7): the trigger sends only one source class to the target.
    - Training copies: all 10 source-class images in A, triggered and labeled as the target, plus 10 triggered images from other classes (not the target) with their correct labels.
    - The poison is capped at 10 copies (5% of A) because A has only 10 images per class, half of the all-to-one 20.
    - ASR is measured on the 10 source-class images in B, so it moves in steps of 0.1 and is noisy (±0.15).
    - 3 targeted patch shadows: ASR 0.30 / 0.90 / 0.70 (2 of 3 valid). Leakage (other classes' triggered B images going to the target): 0.04 / 0.22 / 0.36.
    - Doubling the source copies (20 + 20) showed no clear gain: 0.30 → 0.50 on the one comparable shadow. The other two drew different backdoors, because a different training-set size uses up the generator differently. So it was not adopted.
    - Expect targeted backdoors with weaker triggers (sig, blend, filter) to be invalid more often than targeted patch.
16. **Clean-label SIG is not learnable with this recipe** (Step 7). Copies are all 10 target-class images in A with SIG stripes, keeping their correct label.
    - 3 backdoors (targets 3, 9, 8; sig seeds 11, 22, 33). Parents' ASR before finetuning: 0.016 / 0.053 / 0.000.
      - Appending the copies (decision 3): ASR 0.016 / 0.184 / 0.079.
      - Replacing the originals: ASR 0.016 / 0.179 / 0.116.
      - Clean accuracy was normal in both (0.775–0.81).
    - So decision 3 is not the obstacle. 10 striped target images are too few; real clean-label attacks stripe a large share of a target class with hundreds of images.
    - `clean_label_copies` exists, but `SIG_CLEAN_LABEL_PROB = 0.0`, so SIG backdoors use the normal modes.
    - One 3-backdoor shadow (clean-label sig, targeted patch, all-to-one patch) got ASR 0.037 / 0.400 / 0.400. Shadows with several backdoors are harder to make valid.
17. **The final shadow set** (Step 8: `N_MODELS = 80`, `SEED = 0`, about 36 min on MPS, 25–33 s per shadow, 1.7 GB):
    - 47 clean and 33 poisoned shadows. 17 of the poisoned ones are valid (52%):
      - 1 backdoor: 11/15 valid.
      - 2 backdoors: 6/14 valid.
      - 3 backdoors: 0/4 valid, so there are **no 3-backdoor positives**.
    - 55 backdoors in total; 34 reach ASR ≥ 0.5 on their own (62%). **23 are positives for the AUC**, i.e. in valid shadows.
    - Positives per group: patch 8, blend 5, filter 5, sig 5; all_to_one 12, targeted 11.
    - Median ASR per kind: patch 0.742, filter 0.595, blend 0.497, sig 0.448.
    - Noise: with 23 positives the overall AUC is about ±0.06; per kind (5–8 positives) about ±0.10–0.13.
    - Clean accuracy on B: clean shadows 0.785 on average (0.760–0.815); poisoned 0.760 (0.680–0.800). By number of backdoors: 1 → 0.773, 2 → 0.756, 3 → 0.730.
    - Targeted leakage: median 0.128, range 0.022–0.478.
    - Dummy detectors on this file: random AUC 0.510, oracle 1.000.
18. **Clean accuracy leaks the label on the shadows** (Step 8).
    - A model-level "detector" that scores every class with −clean_acc gets **AUC 0.700** overall, and 0.725 at model level (valid poisoned vs clean shadows).
    - On the real train models the gap is only 0.3 points: poisoned average 0.843, clean 0.846. On the shadows it is 1.5 points: valid poisoned 0.770, clean 0.785.
    - Cause: fitting the mislabeled poisoned copies costs our small finetune some accuracy. The real models lost much less over a full training run.
    - Consequence: any detector whose scores track "the model is slightly worse" (e.g. confidence or loss on B) looks better locally than it is. See the Phase 1 rule.
19. **Cost of the network** (Phase 1 Step 0, measured on 2026-11-18 on an Apple M5 laptop GPU, MPS, 24 GB unified memory, `shadow000`):
    - Forward only: 1.29 ms per image, so a forward-pass detector on the 200 B images takes ~0.26 s per model.
    - Forward + backward with respect to the input only (weights frozen, as in trigger inversion): 3.2 ms per image in float32 (batch 320 or 640), 2.3–2.5 ms with float16 autocast (~1.35× faster).
    - Memory: batch 320 uses 7.4 GB, batch 640 uses 14.9 GB in float32. Autocast uses more (9.4 / 18.3 GB). Keep one optimization step at ≤ ~320 images.
    - Why it is expensive: the CIFAR-style stem keeps the full 64×64 resolution in the first block, so one image costs ~2.2 GMACs forward, about as much as an ImageNet ResNet-18 at 224×224.
    - Consequence: an inversion costs 20 targets × images per step × steps × 3.2 ms per model. Targets share no computation, because each target has its own trigger and so its own triggered images. Examples: 20 × 100 × 300 = 32 min; 20 × 16 × 200 = 3.4 min; 20 × 16 × 100 = 1.7 min; 20 × 8 × 100 = 51 s. A full run of one family covers 64 shadows + 200 test models = 264 models.
20. **Baselines through `run_detector.py`** (Phase 1 Step 1, B images only, 64 valid shadows, 23 positives):
    - `neg_clean_acc` reproduces finding 18 exactly. Its accuracies, recomputed by the detector, match `labels.csv` to 3e-8.

      | group | AUC | within-model AUC | positives |
      |---|---|---|---|
      | all | 0.700 | 0.500 | 23 |
      | kind=blend | 0.674 | 0.500 | 5 |
      | kind=filter | 0.746 | 0.500 | 5 |
      | kind=patch | 0.661 | 0.500 | 8 |
      | kind=sig | 0.740 | 0.500 | 5 |
      | mode=all_to_one | 0.765 | 0.500 | 12 |
      | mode=targeted | 0.628 | 0.500 | 11 |

      - Correlation of max score with clean_acc: −1.00 (by construction).
      - These per-group numbers are the baseline to beat per group. Targeted shadows leak less (0.628), because they get only 10 poisoned copies instead of 20.
    - `random` with seed 0: AUC 0.417 (within-model 0.446). Over 500 seeds the AUC averages 0.499 with std 0.061, matching finding 9's formula, and 11% of seeds give ≤ 0.417. So a single seed is a noisy reference.
    - Dev subset, `random`: 0.464 (12 positives).
    - Organizer train models, B accuracy:
      - poisoned 0.790 / 0.790 / 0.805, clean 0.815 / 0.805 / 0.815.
      - `neg_clean_acc` gets AUC 0.728 there, but with 3 positives and a standard error of about 3 points per accuracy, this is noise.
      - With the organizers' own clean_acc, the gap is 0.3 points (finding 18).
    - Interrupting a run with Ctrl-C after 29 of 64 models and resuming gives a CSV byte-identical to an uninterrupted run.
21. **Forward-pass detectors** (Phase 1 Step 2, cosines on B; 64 valid shadows with 23 positives, plus the 6 train models with 3 positives):

    | detector | shadow AUC | within-model | corr(max score, clean_acc) | train AUC | real target ranks (15, 3, 7) |
    |---|---|---|---|---|---|
    | `cos_other` | 0.976 | 0.977 | −0.46 | 0.897 | 2, 1, 7 |
    | `cos_source` | 0.980 | 0.981 | −0.26 | 0.627 | 11, 1, 12 |
    | `runner_up` | 0.973 | 0.974 | −0.57 | 0.764 | 3, 2, 12.5 |
    | `own_margin` | 0.497 | 0.438 | −0.13 | 0.419 | 14, 17, 6 |

    - Shadow AUC per group:
      - `cos_other`: blend 1.000, filter 0.890, patch 1.000, sig 1.000, all_to_one 1.000, targeted 0.950.
      - `cos_source`: 0.97–0.99 in every group.
      - `runner_up`: blend 0.954, filter 0.935, patch 0.998, sig 0.991, all_to_one 0.981, targeted 0.964.
    - The within-model AUC equals the AUC, so these scores do not ride on clean accuracy. The max score still correlates with lower clean_acc (−0.26 to −0.57), because poisoned shadows have both.
    - `own_margin` has no signal in either direction (its sign was fixed before looking).
    - `cos_source` works on shadows but not on real models (real target ranks 11, 1, 12).
    - On the real train models, `cos_other` and `runner_up` carry a real signal:
      - On `cos_other`, class 15 ranks 11–19 and class 3 ranks 17 in the 3 clean train models, but 2 and 1 in the models where they are the target.
      - Class 7 (the weak backdoor, ASR 0.581) shows nothing: rank 7 in its model, 7–14 in clean ones.
      - Class 15 is not naturally attractive on clean images (rank 11–19); its pull (finding 10) is towards stripes only.
22. **Shadows overstate a backdoor's footprint on clean images** (Phase 1 Step 2).
    - Stand-out of a class within its model = (score − the model's median over 20 classes) / the model's MAD.
    - `cos_other`:
      - shadow targets: median **6.3** (rank 1 in 17/23);
      - top class of a clean shadow: median 2.2, max 3.5;
      - real poisoned train models' targets: **2.6, 2.3, 1.0**.
    - `runner_up`: 7.6 vs 3.2 vs 3.2 / 2.2 / −0.7. `cos_source`: 4.9 vs 4.1 vs 0.0 / 3.3 / −0.5.
    - So in real models a target stands out about as much as a clean shadow's top class. Local AUCs of 0.97–0.98 are not a forecast for the test set.
    - Where the shadow footprint comes from: mixing each poisoned shadow with its parent (diagnostic only), over the 23 targets, median `cos_other` stand-out:

      | features | head | stand-out |
      |---|---|---|
      | parent | parent | 0.0 |
      | parent | shadow | 2.5 |
      | shadow | parent | 6.1 |
      | shadow | shadow | 6.5 |

      - The target's head row turns ~44× more than other rows (1 − cos to parent row: median 5.2e-4 vs 1.2e-5), but most of the footprint is in the feature layers: after the finetune, clean images of every class sit closer to the target.
      - So freezing the head during shadow finetuning would not remove it.
      - Likely cause (not tested): the finetune memorizes the 10–20 poisoned copies (finding 11) instead of learning the trigger as a separate feature, which pulls generic image content towards the target.
      - Candidate fix (not tested): data augmentation during shadow finetuning, with the trigger applied after augmentation, so that only the trigger is consistent across the poisoned copies.
23. **B vs all 400 images on models that never saw them** (Phase 1, after Step 2; `cos_other` on the 6 train models):
    - The A half and the B half give nearly the same 20 class scores: correlation 0.88–0.95 per model. So `cos_other` is mostly a stable property of the model, not image noise.
    - Real target ranks (15, 3, 7): A 4 / 2 / 4, B 2 / 1 / 7, all 400 images 2 / 1 / 5. Train AUC: A 0.849, B 0.897, A+B 0.895.
    - So for mean statistics like `cos_other`, using all 400 images on the test models changes little. It matters more for detectors that split images (inversion: 10 + 10 per class instead of 5 + 5). That is a Phase 2 decision; the first submission uses B, exactly as validated.

## Code

Run all scripts from `p2_backdoors/`, because the data paths in `data.py` are relative.

- `model.py`: an unchanged copy of `data/example/model.py`, so `from model import load` works.
- `data.py`: path constants, `get_device()` (cuda, then mps, then cpu), `load_clean_images()` and `split_ab(labels)`.
  - `split_ab` uses the fixed `AB_SPLIT_SEED` and returns index tensors, 10 + 10 per class.
  - Also: `clean_train_model_ids()`, `load_train_model(id, device)`, `load_shadow_model(id, device)`, `test_model_ids()` (the 200 file names in `data/test/models` without `.pt`, sorted) and `load_test_model(id, device)`.
  - Images are read with PIL + `pil_to_tensor`, because `torchvision.io.read_image` is deprecated in torchvision 0.29.
- `triggers.py`: one function per trigger type. It takes a seed and returns `apply(x)`.
  - The seed fixes everything about the trigger, so every image gets the identical trigger.
  - `apply` contains no randomness, never modifies its input, and follows the device of `x`.
  - Types implemented: `patch`, `blend`, `sig`, `wanet`, `color_filter`, `lowfreq`.
  - In `TRIGGERS` (used for shadows): `patch`, `blend`, `sig`, and `color_filter` registered as `"filter"`.
  - Not in `TRIGGERS`: `wanet` and `lowfreq` (findings 11, 13).
  - `wanet` uses `F.grid_sample`, which runs natively on MPS (it matches CPU to 1e-7, no fallback needed).
- `make_shadows.py`: constants at the top (`N_MODELS` = 80, `CLEAN_FRACTION`, `BACKDOOR_COUNT_PROBS`, `TRIGGERS`, `MODE_PROBS`, `SIG_CLEAN_LABEL_PROB`, `MIN_ASR`, training settings, `SEED`). `main()` loops over `N_MODELS` shadows. For each one it:
  1. Picks the parent by cycling through the 3 clean train models.
  2. Calls `draw_backdoors`: clean with probability 0.5, otherwise 1–3 backdoors with distinct targets.
     - Each backdoor gets a random trigger kind, trigger seed and mode (`MODE_PROBS`: 70% `all_to_one`, 30% `targeted`).
     - A targeted backdoor also gets a random `source` class that is not its target.
     - A sig backdoor becomes `clean_label` with probability `SIG_CLEAN_LABEL_PROB` (currently 0).
  3. Builds the training set: A plus `backdoor_copies` per backdoor, always made from clean A images.
     - `backdoor_copies` calls `poisoned_copies` for `all_to_one`, `targeted_copies` for `targeted` and `clean_label_copies` for `clean_label`.
  4. Calls `finetune`, which never calls `net.train()`, so BatchNorm stays frozen.
  5. Calls `save_shadow` (float16, CPU) to write `shadows/models/<id>.pt`.
  6. Reloads the file and measures ASR and clean accuracy on B.
     - `attack_success_rate(net, images, labels, backdoor, device)` uses only source-class images for `targeted`, and all non-target images otherwise (`all_to_one`, `clean_label`).
     - `leakage(...)` is computed for targeted backdoors only.
  7. Calls `write_labels`, which rewrites `shadows/labels.csv` after every shadow.
  - Not resumable: one shared generator drives every shadow's draws, data order and copies, so shadow i depends on everything drawn for shadows 0..i−1. A stopped run must start again from shadow 0.
  - Model weights are ignored by git (`*.pt`, `/shadows/models/`); `shadows/labels.csv` is tracked. Until Phase 1 Step 1, `.gitignore` listed the whole `shadows/` folder, so the labels were not tracked either.
- `shadows/labels.csv` (the shadows' labels): columns `id,base,targets,kinds,modes,sources,asr,leakage,clean_acc,valid`.
  - Lists are `;`-separated, in the same backdoor order in every column.
  - A clean shadow has empty `targets`, `kinds`, `modes` and `asr`, like the organizers' file.
  - `valid` is 1 when every backdoor has ASR ≥ 0.5 on B; a clean shadow is always 1.
  - Modes: `all_to_one`, `targeted` (and `clean_label`, currently off).
  - `sources` and `leakage` hold `-` for backdoors that are not targeted, e.g. `sources = -;9;-`.
- `evaluate.py`: computes a detector's AUC against `shadows/labels.csv`.
  - `auc(pos, neg)`: the competition metric, computed by comparing every (positive, negative) pair. A tie counts as half.
  - `evaluate_detector(scores, labels_csv)`: `scores` is `{shadow_id: 20 scores}`, where a higher score means that class is more likely a backdoor target.
  - It returns `{group: (auc, n_positives)}` for the groups `all`, `kind=<kind>` and `mode=<mode>`. Each group's positives are compared with all negatives.
  - It skips invalid shadows and shadows missing from `scores`, and with no positives it crashes with ZeroDivisionError.
  - `read_labels(labels_csv)`: the rows of a labels file as dicts (works for `shadows/labels.csv` and the organizers' `train/labels.csv`).
  - `subtract_model_median(scores)`: every model's scores minus its median. Evaluating these gives the within-model AUC, which ignores any model-level shift such as clean accuracy.
  - `clean_acc_correlation(scores, labels_csv)`: Pearson correlation between each shadow's max score and its `clean_acc`.
- `detectors.py`: cheap detectors (forward passes only), with the interface from "Phase 1 design".
  - Baselines: `random_scores`, `neg_clean_acc`.
  - Helpers: `cosines(net, images)` (logits / `net.r.fc.scale`), `hide_true_class(cos, labels)` (own class set to −inf), `per_class_mean(values, labels)`.
  - Per-class statistics (finding 21):
    - `cos_other`: mean cosine of class c on the other classes' images.
    - `cos_source`: the highest mean cosine of any single other class's images to c.
    - `runner_up`: share of other classes' images on which c is the strongest wrong class.
    - `own_margin`: minus the mean margin of c's own images.
- `scores.py`: score files. `read_scores(path)` returns `{model_id: 20 floats}`. `write_scores(path, scores)` writes a temporary file and renames it, so stopping mid-write never leaves half a file. Population normalization and combining come in Steps 3 and 6.
  - `check_submission(path, test_ids)`: reads the file as text and asserts the portal's rules: header `id,c00,...,c19`, 20 finite numbers per row, no repeated id, exactly the test ids. Tested on deliberately broken files (missing, unknown and repeated id, nan, inf, wrong header): each is rejected, and a correct file passes.
- `run_detector.py`: constants at the top: `DETECTOR` (a key of `DETECTORS`), `RUN` (`"shadows"`, `"dev"`, `"train"` or `"test"`) and `DEV_SHADOWS`.
  - `RUN = "test"` scores the 200 test models into `results/test/<detector>.csv`, which is a submission file, and then runs `check_submission` on it. An unfinished run fails the check, so a partial file can't be uploaded by mistake.
  - Runs the detector on B for every model of the chosen set and writes `results/[dev/|train/]<detector>.csv` after every model. Models already in the file are skipped.
  - Calls `torch.manual_seed(0)` once (for the `random` baseline).
  - Prints one line per model (top-3 classes, true targets, time), then a report:
    - shadows/dev: number of clean/poisoned models scored, AUC and within-model AUC per group, correlation with clean_acc;
    - train: rank of each true target (ties count as half) and the AUC over the 6 models.
  - To run another detector or set without editing the file: `uv run python -c "import run_detector as r; r.DETECTOR='random'; r.RUN='train'; r.main()"`.
- `results/`: score CSVs, ignored by git (local only). `.gitignore` entries are anchored with `/` so each matches only the top-level folder.
- Conventions:
  - Call `get_device()` once per script and pass `device` down as an argument.
  - No defensive code: wrong assumptions should crash.
- Gotcha: `torch.linalg.lstsq` in float32 on CPU returned a wrong solution (error 0.7) for a well-conditioned 722k × 4 problem. Use float64 for such checks.
