#!/usr/bin/env python3
"""
Demonstration of adaptive compression thresholds.

This script shows how compression thresholds adapt based on VIX regime:
- Low volatility (VIX percentile < 30): Tight thresholds (20/80)
- Normal volatility (VIX percentile 30-70): Moderate thresholds (25/75)
- High volatility (VIX percentile > 70): Wide thresholds (30/70)

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
"""

import sqlite3
from datetime import datetime, timedelta

from trader_koo.structure.vix_analysis import (
    calculate_compression_thresholds,
    detect_compression_signal,
    format_compression_thresholds_display,
)


def create_demo_database(scenario: str) -> sqlite3.Connection:
    """Create a demo database with VIX data for different scenarios."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    
    conn.execute("""
        CREATE TABLE price_daily (
            ticker TEXT,
            date TEXT,
            close REAL
        )
    """)
    
    base_date = datetime.now()
    
    if scenario == "low_vol":
        # Low volatility scenario - VIX around 12-15
        print("\n=== LOW VOLATILITY SCENARIO ===")
        print("VIX range: 12-15 (calm market)")
        for i in range(90):
            date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            close = 12.0 + (i % 3)
            conn.execute(
                "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
                ("^VIX", date, close)
            )
    
    elif scenario == "normal_vol":
        # Normal volatility scenario - VIX around 16-20
        print("\n=== NORMAL VOLATILITY SCENARIO ===")
        print("VIX range: 16-20 (typical market)")
        for i in range(90):
            date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            close = 16.0 + (i % 4)
            conn.execute(
                "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
                ("^VIX", date, close)
            )
    
    elif scenario == "high_vol":
        # High volatility scenario - VIX around 25-35
        print("\n=== HIGH VOLATILITY SCENARIO ===")
        print("VIX range: 25-35 (stressed market)")
        for i in range(90):
            date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            close = 35.0 - (i % 10)
            conn.execute(
                "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
                ("^VIX", date, close)
            )
    
    conn.commit()
    return conn


def demonstrate_scenario(scenario: str):
    """Demonstrate adaptive thresholds for a given scenario."""
    conn = create_demo_database(scenario)
    
    # Calculate adaptive thresholds
    thresholds = calculate_compression_thresholds(conn)
    
    # Display thresholds
    print("\n" + format_compression_thresholds_display(thresholds))
    
    # Test compression detection at various BB width percentiles
    print("\nCompression Detection Examples:")
    print("-" * 50)
    
    test_percentiles = [10, 25, 50, 75, 90]
    
    for bb_width_pct in test_percentiles:
        state, labeled = detect_compression_signal(bb_width_pct, thresholds)
        print(f"  BB Width @ {bb_width_pct:3d}th percentile → {labeled}")
    
    conn.close()


def main():
    """Run demonstrations for all scenarios."""
    print("=" * 70)
    print("ADAPTIVE COMPRESSION THRESHOLDS DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how compression thresholds adapt to market regime.")
    print("Tighter thresholds in calm markets catch subtle compression.")
    print("Wider thresholds in volatile markets reduce false signals.")
    
    # Demonstrate each scenario
    demonstrate_scenario("low_vol")
    demonstrate_scenario("normal_vol")
    demonstrate_scenario("high_vol")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. Low volatility → Tight thresholds (20/80)")
    print("   - More sensitive to compression/expansion")
    print("   - Catches subtle regime changes")
    print()
    print("2. Normal volatility → Moderate thresholds (25/75)")
    print("   - Balanced sensitivity")
    print("   - Standard regime detection")
    print()
    print("3. High volatility → Wide thresholds (30/70)")
    print("   - Less sensitive to noise")
    print("   - Reduces false signals in chaotic markets")
    print()
    print("This adaptive approach ensures compression signals remain")
    print("relevant across different market environments.")
    print("=" * 70)


if __name__ == "__main__":
    main()
