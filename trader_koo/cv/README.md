# CV Pattern Detection — parameters and references

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
