# Debate Engine Improvements

## Problem Statement

The current debate engine shows 100% agreement on most setups because all 5 analysts (Trend, Momentum, Valuation, YOLO, Risk Manager) look at similar signals and reach correlated conclusions. When a trend is clear, all analysts agree, which defeats the purpose of multi-angle analysis.

## Solution: Bull/Bear Researcher Debate (IMPLEMENTED ✅)

Inspired by the TradingAgents paper (Xiao et al., 2024), we've implemented a structured debate between Bull and Bear researchers who evaluate analyst reports from opposing perspectives.

### Implementation Details

**Bull Researcher:**

- Highlights positive signals and growth potential
- Amplifies bullish analyst stances (1.3x multiplier)
- Focuses on evidence from bullish analysts
- Optimistic by design

**Bear Researcher:**

- Focuses on risks and negative signals
- Amplifies bearish analyst stances (1.3x multiplier)
- Skeptical by design - even neutral analysts with risk flags matter
- Additional checks:
  - Overextension: pct_vs_ma20 > 8%
  - Weak volume: volume_ratio < 0.7

**Agreement Calculation:**

- Based on Bull vs Bear researcher confidence gap
- When they disagree (expected): 40-85% agreement based on confidence gap
- When one is neutral: 65% agreement
- When both agree (rare): 100% agreement
- Penalty applied when all 4+ analysts have identical stance (reduces agreement by 25%)

### Results

Before: Agreement scores were consistently 100% on clear setups

After: Agreement scores vary naturally:

- Strong bullish: ~49% (Bull confident, Bear neutral)
- Mixed signals: ~63% (Bull and Bear both active but disagreeing)
- Strong bearish: ~49% (Bear confident, Bull neutral)

### API Changes

The consensus object now includes a `debate` field:

```json
{
  "consensus": {
    "consensus_bias": "bullish",
    "consensus_state": "watch",
    "agreement_score": 48.8,
    "consensus_strength": 0.475,
    "debate": {
      "bull_researcher": {
        "stance": "bullish",
        "confidence": 0.95,
        "evidence": ["uptrend", "breakout up", "golden cross regime"]
      },
      "bear_researcher": {
        "stance": "neutral",
        "confidence": 0.5,
        "evidence": []
      }
    }
  }
}
```

### Testing

Comprehensive test suite added in `tests/test_debate_agreement_variance.py`:

- ✅ Agreement scores vary based on signal quality (not always 100%)
- ✅ Bull and Bear researchers debate with different perspectives
- ✅ Unanimous analyst agreement triggers skepticism penalty
- ✅ Debate structure includes researcher details in API response

## Reference

TradingAgents: Multi-Agents LLM Financial Trading Framework
Yijia Xiao, Edward Sun, Di Luo, Wei Wang (2024)
arXiv:2412.20138
https://arxiv.org/abs/2412.20138

## Future Enhancements (Optional)

If agreement scores still need more variance:

1. **Timeframe Diversity**: Split Trend Analyst into short-term (5-20 day) and long-term (50-200 day) views
2. **Contrarian Analyst**: Add explicit devil's advocate role with 0.15 weight
3. **VIX Integration**: Make Bear researcher more aggressive when VIX percentile > 70
