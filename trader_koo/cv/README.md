# CV Pipeline — Detect → Label → Train Loop

This folder supports an incremental bootstrapping loop for chart pattern recognition:

1. Auto-detect patterns from price data (rule-based + CV proxy)
2. Generate annotated chart images for human review
3. Approve/reject in Label Studio or a CSV
4. Grow a gold label set over time
5. Train a YOLO model on the gold set
6. Use model confidence to promote pseudo-labels → repeat

---

## Pattern classes

| Class | Type | Bulkowski failure rate | Avg move after breakout |
|---|---|---|---|
| `double_top` | Bearish reversal | 22% | −14% |
| `double_bottom` | Bullish reversal | 15% | +37% |
| `head_and_shoulders` | Bearish reversal | 4% | −20% |
| `inv_head_and_shoulders` | Bullish reversal | 7% | +37% |
| `bull_flag` | Bullish continuation | 4% | +23% |
| `bear_flag` | Bearish continuation | 5% | −22% |
| `rising_wedge` | Bearish reversal | 25% | −15% |
| `falling_wedge` | Bullish reversal | 30% | +32% |
| `ascending_triangle` | Bullish continuation | 13% | +32% |
| `descending_triangle` | Bearish continuation | 16% | −19% |
| `symmetrical_triangle` | Neutral | 40% | ±10% |
| `cup_and_handle` | Bullish continuation | — | — |

*Source: Thomas Bulkowski, "Encyclopedia of Chart Patterns" (2,000+ back-tested patterns)*

---

## Pattern parameter research (Bulkowski)

These are empirically derived from thousands of historical patterns. Use these to
sanity-check detection output and calibrate thresholds.

### Double Top / Double Bottom

- **Peak/trough separation**: 10–65 trading days between the two peaks
- **Minimum valley depth**: ≥ 10% decline between peaks (our detector uses 5% — Bulkowski
  suggests tightening to 10% significantly reduces false positives)
- **Peak similarity**: Peaks within 3–5% of each other in price
- **Breakout timing**: Breakdown/breakout typically occurs within 15–25 bars of 2nd peak

### Head & Shoulders / Inverse H&S

- **Total pattern duration**: 60–120 trading days (left shoulder → right shoulder)
- **Head dominance**: Head must be ≥ 10% higher/lower than shoulders
- **Shoulder symmetry**: Left and right shoulder within 10% height of each other
- **Neckline slope**: Should be ≤ 5% sloped (near-horizontal)
- **Breakout timing**: Breakdown through neckline typically within 20–30 bars of right shoulder
- **Price target**: Pattern height (head to neckline) projected from breakout point

### Flags (Bull / Bear)

- **Pole**: Sharp move of ≥ 15–30% in 5–15 bars
- **Flag consolidation**: 5–25 trading days (longer = lower reliability)
- **Channel angle**: ≤ 45° retracement against the trend (steep = not a flag, it's a reversal)
- **Flag retracement**: Consolidation should retrace ≤ 50% of the pole

### Wedges (Rising / Falling)

- **Duration**: 25–75 trading days
- **Convergence**: Lines must converge meaningfully (≥ 30% reduction in channel width)
- **R² quality**: Both trendlines should fit the price action well (R² ≥ 0.45)
- **Rising wedge**: Both lines slope upward, but upper line rises slower → converging
- **Falling wedge**: Both lines slope downward, but lower line falls slower → converging

### Triangles

- **Duration**: 30–90 trading days
- **Ascending**: Flat resistance top, rising support bottom
- **Descending**: Falling resistance top, flat support bottom
- **Symmetrical**: Both lines converging toward an apex
- **Breakout zone**: Typically in the middle third of the triangle (not at the apex)

---

## Key academic references

1. **Bulkowski, T. (2005)** — *Encyclopedia of Chart Patterns* — most comprehensive
   empirical study. Statistics derived from 2,000+ patterns on US equities.

2. **Lo, Mamaysky & Wang (2000)** — *"Foundations of Technical Analysis"* — academic
   paper using kernel regression to detect patterns. Found statistically significant
   conditional returns for H&S, double tops/bottoms on US equities 1962–1996.

3. **Nison, S. (1991)** — *Japanese Candlestick Charting Techniques* — candlestick
   patterns (single-bar to 3-bar) with high short-term predictive value.

---

## Workflow commands

### Step 1 — Detect patterns + render images

```bash
cd trader_koo
source .venv/bin/activate

# Small test batch first (renders clean + annotated images)
python scripts/grow_gold_labels.py detect \
    --tickers "AAPL,SPY,NVDA,MSFT" \
    --max-windows-per-ticker 50 \
    --render-images

# Output:
#   data/cv/images/          ← clean PNGs for model training (no text/lines)
#   data/cv/images_review/   ← annotated PNGs with detected lines (for human review)
#   data/cv/pending_review_template.csv
```

### Step 2 — Human review (Option A: CSV)

Open `data/cv/pending_review_template.csv` in Excel/Numbers.
Look at each `images_review/{sample_id}.png`, fill `approved` = TRUE/FALSE.

### Step 2 — Human review (Option B: Label Studio)

```bash
# Export to Label Studio format (uses annotated review images)
python scripts/export_label_studio_tasks.py \
    --cv-dir data/cv \
    --source review_queue \
    --tasks-out data/cv/ls_tasks.json \
    --config-out data/cv/ls_config.xml \
    --image-mode absolute

# In Label Studio:
#   1. New project → Settings → Labeling Interface → paste ls_config.xml
#   2. Import → upload ls_tasks.json
#   3. Review each chart (lines are pre-drawn)
#   4. Export → JSON → save as data/cv/ls_annotations.json

# Convert Label Studio export back to decisions
python scripts/ls_export_to_gold.py \
    --ls-export data/cv/ls_annotations.json \
    --weak-labels data/cv/batch_weak_labels.csv \
    --out-decisions data/cv/review_decisions.csv
```

### Step 3 — Merge approved labels into gold set

```bash
python scripts/grow_gold_labels.py merge
# → appends confirmed labels to data/cv/gold_labels.csv
# → archives pending_review_template.csv as .done.csv
```

### Step 4 — Calibrate thresholds (once you have ≥100 gold labels)

```bash
python scripts/calibrate_thresholds.py
# Compares detected patterns vs gold labels
# Sweeps depth_pct, min_shape_r2 etc. and reports precision/recall/F1
# Prints recommended CVProxyConfig values
```

### Step 5 — Scale to all tickers

```bash
python scripts/grow_gold_labels.py detect \
    --use-all-tickers \
    --render-images
python scripts/grow_gold_labels.py merge
```

### Step 6 — Pseudo-label expansion (after model training)

```bash
python scripts/promote_cv_pseudo_labels.py \
    --model-predictions-csv data/cv/model_predictions.csv \
    --gold-labels-csv data/cv/gold_labels.csv \
    --out-dir data/cv
```

---

## Two image folders explained

| Folder | Purpose | Contents |
|---|---|---|
| `data/cv/images/` | **Model training** | Plain candlestick only — no text, no lines, no axes |
| `data/cv/images_review/` | **Human review** | Candlestick + detected pattern lines + labels |

The model must NOT train on `images_review/` — it would learn to detect the text labels
instead of the actual price patterns.

---

## Detection thresholds (current vs Bulkowski)

| Parameter | Current | Bulkowski recommendation |
|---|---|---|
| Double top peak similarity | 3% | 3–5% ✓ |
| Double top valley depth | 5% | ≥ 10% (consider tightening) |
| H&S shoulder symmetry | 6% | ≤ 10% ✓ |
| H&S head dominance | 5% | ≥ 10% (consider tightening) |
| Wedge convergence | 35%+ required | ≥ 25–30% ✓ |
| Shape R² quality | ≥ 0.45 | no reference, empirical |
| Flag pole return | 5% | ≥ 15% (consider tightening) |

Run `calibrate_thresholds.py` once you have gold labels to get data-driven values.

---

## Notes

- This is an assistive labeling workflow, not a production trading system.
- Keep a human in the loop for new pattern classes and market regime changes.
- Use strict out-of-sample (time-ordered) evaluation before trusting model confidence.
- The gold set should have ≥ 200 confirmed examples per class for reliable model training.
