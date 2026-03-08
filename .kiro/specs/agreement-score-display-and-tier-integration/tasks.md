# Implementation Plan: Agreement Score Display and Tier Integration

## Overview

This implementation adds agreement score visibility to setup cards and integrates agreement scores as a risk factor in tier calculations. The debate engine already produces agreement scores (40-100%), but they're currently hidden from users and not factored into tier assignments. Low agreement scores (below 50%) indicate high debate/uncertainty and will trigger tier downgrades.

The implementation follows this approach:

1. Add backend tier adjustment logic for agreement scores
2. Add frontend display logic for agreement scores on cards
3. Add comprehensive testing (unit and property-based tests)

## Tasks

- [x] 1. Implement backend tier adjustment logic

  - [x] 1.1 Add agreement score tier adjustment function to report generator
    - Create `_apply_agreement_tier_adjustment(row)` function in `trader_koo/scripts/generate_daily_report.py`
    - Extract agreement score from `debate_v1.consensus.agreement_score`
    - Apply tier downgrade when agreement < 50% (A→B, B→C, C→D, D→D)
    - Handle missing/invalid agreement scores gracefully
    - Add logging for tier adjustments
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 4.1, 4.2, 4.3, 4.4, 4.5_
  - [x] 1.2 Add helper function for tier downgrading
    - Create `_downgrade_tier(tier)` function in `trader_koo/scripts/generate_daily_report.py`
    - Implement tier map: A→B, B→C, C→D, D→D
    - Return valid tier or default to D
    - _Requirements: 2.5, 2.6_
  - [x] 1.3 Integrate agreement adjustment into tier calculation flow
    - Call `_apply_agreement_tier_adjustment(row)` after YOLO conflict adjustments
    - Call before debate state guardrails (watch/conditional adjustments)
    - Ensure agreement adjustment modifies `setup_tier` in place
    - _Requirements: 2.1, 2.5_
  - [ ]\* 1.4 Write unit tests for tier adjustment logic
    - Test tier downgrade A→B, B→C, C→D when agreement < 50%
    - Test tier D stays D when agreement < 50%
    - Test no tier change when agreement >= 50%
    - Test no tier change when agreement is missing
    - Test agreement score clamping (negative → 0, >100 → 100)
    - Test warning logged when score is clamped or missing
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 4.1, 4.2, 4.3, 4.4_
  - [ ]\* 1.5 Write property test for tier adjustment
    - **Property 3: Agreement Score Above 50% Preserves Tier**
    - **Validates: Requirements 2.2, 2.3**
    - Generate random setups with agreement >= 50%
    - Verify tier unchanged after adjustment
  - [ ]\* 1.6 Write property test for tier downgrade
    - **Property 4: Agreement Score Below 50% Downgrades Tier**
    - **Validates: Requirements 2.4, 2.5**
    - Generate random setups with agreement < 50% and tier A/B/C
    - Verify tier downgraded by exactly one level
  - [ ]\* 1.7 Write property test for tier D preservation
    - **Property 5: Tier D Cannot Be Downgraded**
    - **Validates: Requirements 2.6**
    - Generate random setups with tier D and agreement < 50%
    - Verify tier remains D
  - [ ]\* 1.8 Write property test for missing agreement score
    - **Property 6: Missing Agreement Score Preserves Tier**
    - **Validates: Requirements 2.7**
    - Generate random setups with missing agreement scores
    - Verify tier unchanged after adjustment

- [x] 2. Implement frontend agreement score display

  - [x] 2.1 Add agreement score formatting function
    - Create `formatAgreementScore(row)` function in `trader_koo/frontend/index.html`
    - Extract score from `debate_agreement_score` or `debate_v1.consensus.agreement_score`
    - Format as percentage with one decimal place (e.g., "65.0%")
    - Return "N/A" for missing/invalid scores
    - _Requirements: 1.3, 1.4, 1.5_
  - [x] 2.2 Add agreement score to compact list view
    - Modify `renderSetupCompactList(rows)` in `trader_koo/frontend/index.html`
    - Add agreement score after tier in score line
    - Format: "Score • Tier X • Agree Y%" or "Score • Tier X • Agree N/A"
    - _Requirements: 1.1, 1.3, 1.4, 1.5_
  - [x] 2.3 Add agreement score to audit cards view
    - Modify `renderSetupAuditCards(rows)` in `trader_koo/frontend/index.html`
    - Add agreement score badge to badge list
    - Format: Badge showing "Agree Y%" or "Agree N/A"
    - _Requirements: 1.2, 1.3, 1.4, 1.5_
  - [ ]\* 2.4 Write unit tests for agreement score display
    - Test agreement score display in compact card with valid score
    - Test agreement score display in audit card with valid score
    - Test "N/A" display when score is null/undefined/NaN
    - Test formatting with one decimal place (65.0%, 50.5%, 100.0%)
    - Test fallback from debate_agreement_score to debate_v1.consensus.agreement_score
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ]\* 2.5 Write property test for agreement score display
    - **Property 1: Agreement Score Display on All Card Views**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
    - Generate random setups with valid agreement scores
    - Verify rendered HTML contains formatted percentage
  - [ ]\* 2.6 Write property test for missing agreement score display
    - **Property 2: Missing Agreement Score Displays N/A**
    - **Validates: Requirements 1.5**
    - Generate random setups with null/undefined/NaN agreement scores
    - Verify rendered HTML contains "N/A"

- [x] 3. Checkpoint - Verify core functionality

  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Add data flow validation tests

  - [ ]\* 4.1 Write property test for debate engine agreement score production
    - **Property 7: Debate Engine Produces Agreement Scores**
    - **Validates: Requirements 3.1**
    - Generate random valid setup rows
    - Call build_setup_debate and verify consensus.agreement_score exists and is numeric
  - [ ]\* 4.2 Write property test for agreement score data flow
    - **Property 8: Agreement Score Data Flow Round Trip**
    - **Validates: Requirements 3.2, 3.3, 3.5**
    - Generate random setups with debate_v1.consensus.agreement_score
    - Process through report generator
    - Verify debate_agreement_score matches original value
  - [ ]\* 4.3 Write property test for backend API agreement score inclusion
    - **Property 9: Backend API Includes Agreement Score**
    - **Validates: Requirements 3.4**
    - Generate random setup rows
    - Call API endpoint
    - Verify response includes debate_agreement_score field
  - [ ]\* 4.4 Write integration test for full data flow
    - Test full pipeline: debate engine → report generator → API → frontend
    - Verify agreement score preserved at each stage
    - Verify tier adjustment applied correctly
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 5. Add validation and error handling tests

  - [ ]\* 5.1 Write property test for agreement score validation
    - **Property 10: Agreement Score Validation and Clamping**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.5**
    - Generate random numeric values (including out of range, NaN, infinity)
    - Verify clamping for out-of-range values
    - Verify missing treatment for non-numeric values
    - Verify no errors thrown
  - [ ]\* 5.2 Write property test for invalid agreement score logging
    - **Property 11: Invalid Agreement Score Logging**
    - **Validates: Requirements 4.4**
    - Generate random invalid agreement scores (out of range or missing)
    - Verify warning logged for each invalid case
  - [ ]\* 5.3 Write unit tests for error handling
    - Test missing debate data structure (null debate_v1)
    - Test missing consensus object
    - Test invalid tier values in downgrade logic
    - Test tier calculation order (score → YOLO → agreement → debate state)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 6. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property-based tests use Hypothesis (Python) and fast-check (JavaScript)
- Agreement score adjustment happens after YOLO adjustments but before debate state guardrails
- Frontend already has access to `debate_agreement_score` field via API - no backend changes needed
- Tier downgrade logic: A→B, B→C, C→D, D→D (no tier E exists)
- Agreement scores below 50% indicate high debate/uncertainty and trigger tier downgrades
