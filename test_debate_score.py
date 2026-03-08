#!/usr/bin/env python3
"""Quick test to verify debate engine produces agreement scores."""

from trader_koo.debate_engine import build_setup_debate

# Test with various scenarios
test_cases = [
    {
        'name': 'Bullish setup',
        'row': {
            'ticker': 'AAPL',
            'trend_state': 'uptrend',
            'pct_change': 2.0,
            'volume_ratio_20': 1.5,
            'discount_pct': 20,
            'peg': 1.0,
            'yolo_pattern': 'bull_flag',
            'yolo_bias': 'bullish',
            'yolo_recency': 'fresh',
            'actionability': 'setup_ready'
        }
    },
    {
        'name': 'Bearish setup',
        'row': {
            'ticker': 'TSLA',
            'trend_state': 'downtrend',
            'pct_change': -2.5,
            'volume_ratio_20': 1.4,
            'discount_pct': -10,
            'peg': 3.5,
            'yolo_pattern': 'bear_flag',
            'yolo_bias': 'bearish',
            'yolo_recency': 'recent',
            'actionability': 'watch'
        }
    },
    {
        'name': 'Neutral/mixed setup',
        'row': {
            'ticker': 'MSFT',
            'trend_state': 'mixed',
            'pct_change': 0.5,
            'volume_ratio_20': 0.9,
            'discount_pct': 5,
            'peg': 2.0,
            'yolo_pattern': 'consolidation',
            'yolo_bias': 'neutral',
            'yolo_recency': 'aging',
            'actionability': 'watch-only'
        }
    }
]

print("Testing debate engine agreement score production:\n")

for test in test_cases:
    print(f"Test: {test['name']}")
    print(f"  Ticker: {test['row']['ticker']}")
    
    result = build_setup_debate(test['row'])
    
    print(f"  Result mode: {result.get('mode')}")
    
    consensus = result.get('consensus', {})
    agreement_score = consensus.get('agreement_score')
    
    print(f"  Agreement score: {agreement_score}")
    print(f"  Consensus bias: {consensus.get('consensus_bias')}")
    print(f"  Consensus state: {consensus.get('consensus_state')}")
    
    if agreement_score is None:
        print("  ❌ FAIL: agreement_score is None!")
    elif not isinstance(agreement_score, (int, float)):
        print(f"  ❌ FAIL: agreement_score is not numeric: {type(agreement_score)}")
    elif agreement_score < 0 or agreement_score > 100:
        print(f"  ❌ FAIL: agreement_score out of range: {agreement_score}")
    else:
        print(f"  ✓ PASS: agreement_score is valid")
    
    print()

print("All tests complete.")
