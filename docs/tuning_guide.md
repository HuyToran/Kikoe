# Hyperparameter Tuning Guide

This guide documents hyperparameter insights from training all three bundled wake word models (ハナケア, おやすみなさい, 寝てください) across 37 total training runs.

**Targets:** FP ≤ 2.0/hr AND Recall ≥ 0.65

---

## Key Parameters

| Parameter | Config key | Effect |
|---|---|---|
| `layer_size` | `layer_size` | Model capacity. Bigger = more Recall, more FP risk. |
| `steps` | `steps` | Training duration. More steps = higher Recall ceiling, but can overfit FP. |
| `max_negative_weight` | `max_negative_weight` | Penalty on false positives. Higher = lower FP, lower Recall. |
| `augmentation_rounds` | `augmentation_rounds` | Data augmentation passes. 3 is a good default. |
| `random_seed` | `random_seed` | Different seeds find different optima at equal settings. |

---

## Rules of Thumb

### layer_size — start small, go bigger only if needed

- **layer=32** is sufficient for words with 5+ syllables and distinct phonemes (おやすみなさい, 寝てください).
- **layer=64** gives marginal Recall gains but often 2-5x FP increase — rarely worth it.
- **layer=256** is necessary for phonetically ambiguous short words (ハナケア — 4 morae, similar to many Japanese words).
- **layer=512** tends to overfit — Recall actually drops vs layer=256 in some cases.

### steps — more is not always better

- **20,000 steps** is the sweet spot for layer=32. More steps hurt FP without improving Recall.
- **35,000-40,000 steps** pairs well with layer=256. It opens better local optima but requires higher `max_negative_weight` to control FP.
- Increasing steps beyond the sweet spot causes FP to spike. If FP is high after adding steps, raise `max_negative_weight` rather than reducing steps.

### max_negative_weight — tune this last

This is the primary FP control knob. The relationship between `max_negative_weight` and FP/hr is **non-linear** and can be highly sensitive:

- For ハナケア (layer=256, steps=40k): the FP-safe zone is max_neg ≥ 1350.
- For 寝てください (layer=32, steps=20k): there is a hard cliff at max_neg=300:
  - max_neg=200 → FP=5.0/hr
  - max_neg=300 → FP=1.06/hr  ← sweet spot
  - max_neg=150 → FP=5.2/hr
- Raising max_neg too high collapses Recall. Binary-search between FP cliff and Recall floor.

### random_seed — try 2-3 seeds before giving up

At identical hyperparameters, different seeds produce meaningfully different results:
- Seed=267 consistently outperformed seed=42 for ハナケア at layer=256 (Recall ~0.04 higher).
- If a config is close but failing one metric by a small margin, try a different seed before changing architecture.

---

## Per-Model Findings

### ハナケア — the hard one

ハナケア is phonetically short (4 morae) with many similar-sounding Japanese words, making it the hardest to train.

**Best config:** `layer=256, max_neg=1350, steps=40k, seed=267` → FP=1.239, Recall=0.686

Key findings across 21+ runs:
- layer=32 and layer=64: FP controllable only at the cost of very low Recall (0.55-0.63).
- layer=128: Recall improved to ~0.68 but required very low max_neg to hit FP target → unstable.
- layer=256: First architecture to cleanly separate FP and Recall targets.
- layer=512: Recall actually dropped vs layer=256 — overfit.
- steps=35k vs 40k: 40k opened a Recall ceiling of ~0.72 (best run: Recall=0.717 at max_neg=1200, but FP=2.655). With max_neg=1350, FP came below 2.0 while keeping Recall at 0.686.

**Adversarial phrases are NOT necessary for ハナケア (confirmed Run 21+):**
- Run 19 (with 19 adversarial phrases): FP=1.239, Recall=0.686.
- Run 21 (adversarial_phrases: []): FP=1.239, Recall=0.686 — identical results.
- Conclusion: the ACAV100M background feature set (~2000hrs) provides sufficient negative signal. Adversarial phrases help for words with very common sub-phrases (e.g. 寝てください), but not for ハナケア which has no phonetically similar common words in the training negatives.

**Previous adversarial phrases tested (for reference):**
ハナ、ケア、ハナケ、ハナケアー、カナケア、ハナゲア、ハナケイ、ハナビ、ハナミ、ハナゲ、バナナ、ケアマネ、ハワイ、アケア、タナカ、はなける、ハナスカ、ナケア、カレア

**Tuning sequence that works:**
1. Start at layer=256, steps=35k, max_neg=900, seed=267, no adversarials.
2. If FP > 2.0: raise max_neg by ~200.
3. If Recall < 0.65: increase steps to 40k; compensate FP by raising max_neg.
4. If stuck, try seed=42 at the same config.

---

### おやすみなさい — the easy one

7-mora word with very distinct phoneme sequence. Almost no similar-sounding Japanese words.

**Best config:** `layer=32, steps=20k` → FP=0.531, Recall=0.683

Key findings across 21 runs:
- layer=32 is perfectly sufficient — this word just "trains itself."
- layer=64: FP exploded to 9-26/hr regardless of other settings. Never use for this word.
- steps=30k+: FP climbs steadily with no Recall benefit vs 20k.
- `max_negative_weight` is not particularly sensitive — the default (1500) works fine.
- Results vary run-to-run at identical settings (seed not recorded for early runs), so rerun if a run is unexpectedly bad.

**Tuning sequence:**
1. Start at layer=32, steps=20k, default config. You will likely pass on the first or second run.
2. If FP > 2.0: raise max_neg slightly (e.g. 1500 → 2000). Do NOT increase layer or steps.

---

### 寝てください — the fastest

Passed on Run 2. The ～てください structure is common enough that the model quickly learns to require "寝て" specifically rather than just detecting "ください."

**Best config:** `layer=32, max_neg=300, steps=20k, seed=42` → FP=1.062, Recall=0.697

Key findings across 4 runs:
- The adversarial set (19 phrases covering all ～てください forms and 寝て conjugations) is critical — include it exactly as in the config.
- max_neg=300 is the sweet spot. There is a hard cliff:
  - max_neg=150 → FP=5.2
  - max_neg=200 → FP=5.0
  - max_neg=300 → FP=1.06
- Recall ceiling is ~0.70 for layer=32. Pushing further is not necessary given the target is 0.65.

**Tuning sequence:**
1. Start at the exact config in `configs/netekudasai_config.yaml`.
2. If FP > 2.0: raise max_neg to 400-500 and retry.
3. If Recall < 0.65 (very unlikely): lower max_neg towards 200 — but expect FP to jump.

---

## Comparative Summary

| Word | Morae | Phonetic difficulty | Recommended layer | Sweet spot steps | Runs to pass |
|---|---|---|---|---|---|
| ハナケア | 4 | High | 256 | 40,000 | ~10-15 |
| おやすみなさい | 7 | Low | 32 | 20,000 | 1-3 |
| 寝てください | 5 | Very low | 32 | 20,000 | 1-2 |

**General heuristic:** Short wake words (≤4 morae) with many phonetically similar words require larger layer sizes and more tuning. Long, phonetically unique words train fast at layer=32.

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| FP very high (>5/hr), Recall OK | max_neg too low | Raise max_neg by 200-500 |
| Recall very low (<0.55), FP OK | max_neg too high | Lower max_neg by 200-300 |
| Both FP high and Recall low | layer too large (overfit) | Reduce layer_size |
| FP and Recall oscillate across runs | Random initialization variance | Fix seed; try 3 seeds at best config |
| Recall stuck below 0.65 at layer=32 | Word too ambiguous for small model | Try layer=64 or layer=128 |
| FP explodes at layer=64+ | Word is long/distinct (no capacity needed) | Return to layer=32 |
