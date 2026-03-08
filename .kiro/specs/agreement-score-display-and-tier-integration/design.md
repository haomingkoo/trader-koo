# Design Document: Agreement Score Display and Tier Integration

## Overview

This feature enhances the trader_koo platform by surfacing agreement scores from the debate engine and incorporating them into tier calculations. The debate engine already produces agreement scores (40-100%) representing consensus between Bull and Bear researchers, but these scores are currently hidden from users and not factored into risk assessment.

The design adds agreement score visibility on setup cards in both compact list and audit card views, and integrates agreement scores as a risk factor in tier calculations. Low agreement scores (below 50%) indicate high debate/uncertainty and will trigger tier downgrades, treating disagreement as a risk signal.

### Key Design Goals

1. Display agreement scores on all setup card views with consistent formatting
2. Integrate agreement scores into tier calculation as a risk adjustment
3. Preserve existing data flow from debate engine through backend to frontend
4. Handle missing or invalid agreement scores gracefully
5. Maintain backward compatibility with existing tier calculation logic

## Architecture

### System Components

The feature touches four main components in the system:

1. **Debate Engine** (`trader_koo/debate_engine.py`): Already produces agreement scores in the consensus object
2. **Report Generator** (`trader_koo/scripts/generate_daily_report.py`): Extracts debate data and calculates tiers
3. **Backend API** (`trader_koo/backend/main.py`): Serves setup data to frontend
4. **Frontend Dashboard** (`trader_koo/frontend/index.html`): Renders setup cards

### Data Flow

```
Debate Engine (agreement_score in consensus)
    ↓
Report Generator (extracts to debate_agreement_score field)
    ↓
Database (stores in setup row)
    ↓
Backend API (includes in setup data response)
    ↓
Frontend (displays on cards, already available)
```

The agreement score already flows through the entire pipeline. This feature adds:

- Display logic in frontend card rendering
- Tier adjustment logic in report generator

### Integration Points

**Frontend Integration:**

- Modify `renderSetupCompactList()` to display agreement score
- Modify `renderSetupAuditCards()` to display agreement score
- Agreement score data is already available in row data via `debate_agreement_score` field

**Backend Integration:**

- No changes required - `debate_agreement_score` is already included in API responses
- Field is already extracted from `debate_v1.consensus.agreement_score` as fallback

**Tier Calculation Integration:**

- Add agreement score check after initial tier assignment
- Apply tier downgrade when agreement < 50%
- Preserve existing tier calculation logic (score-based, confirmation-based, YOLO conflict adjustments)

## Components and Interfaces

### Frontend Components

#### Setup Card Rendering Functions

**Function: `renderSetupCompactList(rows)`**

- Location: `trader_koo/frontend/index.html`
- Current behavior: Renders compact cards with ticker, score, tier, family, bias, context, actionability
- New behavior: Add agreement score display after tier in the score line
- Format: `"Score • Tier X • Agree Y%"` or `"Score • Tier X • Agree N/A"`

**Function: `renderSetupAuditCards(rows)`**

- Location: `trader_koo/frontend/index.html`
- Current behavior: Renders audit cards with detailed metrics
- New behavior: Add agreement score badge to the badge list
- Format: Badge showing `"Agree Y%"` or `"Agree N/A"`

#### Agreement Score Formatting

```javascript
function formatAgreementScore(row) {
  const agreementRaw = Number(
    row.debate_agreement_score ??
      row.debate_v1?.consensus?.agreement_score ??
      NaN,
  );

  if (Number.isFinite(agreementRaw)) {
    return agreementRaw.toFixed(1) + "%";
  }
  return "N/A";
}
```

### Backend Components

#### Tier Calculation with Agreement Score

**Function: `_apply_agreement_tier_adjustment(row)`**

- Location: `trader_koo/scripts/generate_daily_report.py`
- Purpose: Apply tier downgrade based on agreement score
- Called after: Initial tier assignment (score-based and YOLO conflict adjustments)
- Called before: Final tier is stored in row

**Logic:**

```python
def _apply_agreement_tier_adjustment(row: dict[str, Any]) -> None:
    """Apply tier downgrade if agreement score indicates high debate."""
    debate = row.get("debate_v1")
    if not isinstance(debate, dict):
        return

    consensus = debate.get("consensus")
    if not isinstance(consensus, dict):
        return

    agreement_score = _to_float(consensus.get("agreement_score"))
    if agreement_score is None:
        return  # No adjustment if missing

    # Clamp to valid range
    agreement_score = _clamp(agreement_score, 0.0, 100.0)

    # Only downgrade if agreement < 50%
    if agreement_score < 50.0:
        current_tier = str(row.get("setup_tier") or "D").strip().upper()
        downgraded_tier = _downgrade_tier(current_tier)
        row["setup_tier"] = downgraded_tier

        # Log the adjustment
        if downgraded_tier != current_tier:
            logger.info(
                f"Tier downgraded {current_tier}→{downgraded_tier} "
                f"for {row.get('ticker')} due to low agreement ({agreement_score:.1f}%)"
            )
```

**Helper Function: `_downgrade_tier(tier)`**

```python
def _downgrade_tier(tier: str) -> str:
    """Downgrade tier by one level (A→B, B→C, C→D, D→D)."""
    tier_map = {"A": "B", "B": "C", "C": "D", "D": "D"}
    return tier_map.get(tier.strip().upper(), "D")
```

#### Integration with Existing Tier Logic

The tier calculation flow in `generate_daily_report.py`:

1. Calculate base score from confirmations/contradictions
2. Assign initial tier based on score, confirmations, and bias
3. Apply YOLO conflict adjustments (existing)
4. **NEW: Apply agreement score adjustment**
5. Apply debate guardrails (watch/conditional state adjustments)
6. Store final tier in row

The agreement adjustment happens after YOLO adjustments but before debate state guardrails, ensuring it's part of the core tier calculation rather than a post-processing step.

## Data Models

### Agreement Score Field

**Field Name:** `debate_agreement_score`

**Type:** `float | None`

**Range:** 40.0 - 100.0 (clamped to 0.0 - 100.0 for safety)

**Source:** `debate_v1.consensus.agreement_score`

**Storage:** Setup row in database

**Validation:**

- Must be numeric or null
- Clamped to 0-100 range if outside bounds
- Treated as missing if null/undefined
- Warning logged if clamped or missing

### Tier Field

**Field Name:** `setup_tier`

**Type:** `str`

**Values:** "A", "B", "C", "D"

**Calculation Order:**

1. Score-based tier (85+ → A, 70+ → B, 55+ → C, else D)
2. Confirmation/contradiction adjustments
3. YOLO conflict adjustments
4. **Agreement score adjustment (new)**
5. Debate state guardrails (watch → cap at C, conditional → cap at B)

### Row Data Structure

Setup rows already contain:

```python
{
  "ticker": str,
  "score": float,
  "setup_tier": str,
  "setup_family": str,
  "signal_bias": str,
  "debate_v1": {
    "consensus": {
      "agreement_score": float,
      "consensus_state": str,
      "consensus_bias": str,
      ...
    }
  },
  "debate_agreement_score": float,  # Extracted for convenience
  ...
}
```

No schema changes required - all fields already exist.

## Correctness Properties

_A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees._

### Property 1: Agreement Score Display on All Card Views

_For any_ setup card (compact or audit view) with a valid agreement score, the rendered HTML SHALL contain the agreement score formatted as a percentage with one decimal place.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

### Property 2: Missing Agreement Score Displays N/A

_For any_ setup card where the agreement score is null, undefined, or NaN, the rendered HTML SHALL display "N/A" instead of a percentage.

**Validates: Requirements 1.5**

### Property 3: Agreement Score Above 50% Preserves Tier

_For any_ setup with an agreement score >= 50%, applying the agreement tier adjustment SHALL NOT change the tier from its pre-adjustment value.

**Validates: Requirements 2.2, 2.3**

### Property 4: Agreement Score Below 50% Downgrades Tier

_For any_ setup with an agreement score < 50% and tier A, B, or C, applying the agreement tier adjustment SHALL downgrade the tier by exactly one level (A→B, B→C, C→D).

**Validates: Requirements 2.4, 2.5**

### Property 5: Tier D Cannot Be Downgraded

_For any_ setup with tier D and agreement score < 50%, applying the agreement tier adjustment SHALL keep the tier at D.

**Validates: Requirements 2.6**

### Property 6: Missing Agreement Score Preserves Tier

_For any_ setup where the agreement score is null, undefined, or missing from debate data, applying the agreement tier adjustment SHALL NOT change the tier.

**Validates: Requirements 2.7**

### Property 7: Debate Engine Produces Agreement Scores

_For any_ valid setup row data, calling `build_setup_debate(row)` SHALL return a result containing `consensus.agreement_score` as a numeric value.

**Validates: Requirements 3.1**

### Property 8: Agreement Score Data Flow Round Trip

_For any_ setup with debate data containing `debate_v1.consensus.agreement_score`, after report generation processing, the setup row SHALL contain `debate_agreement_score` with the same numeric value.

**Validates: Requirements 3.2, 3.3, 3.5**

### Property 9: Backend API Includes Agreement Score

_For any_ setup row returned by the backend API, the response SHALL include the `debate_agreement_score` field (which may be null).

**Validates: Requirements 3.4**

### Property 10: Agreement Score Validation and Clamping

_For any_ agreement score value processed by the system, if the value is numeric but outside the range [0, 100], it SHALL be clamped to the valid range, and if the value is non-numeric, it SHALL be treated as missing data, and in both cases processing SHALL continue without errors.

**Validates: Requirements 4.1, 4.2, 4.3, 4.5**

### Property 11: Invalid Agreement Score Logging

_For any_ agreement score that is clamped (outside 0-100 range) or missing (null/undefined), the system SHALL log a warning message indicating the condition.

**Validates: Requirements 4.4**

## Error Handling

### Frontend Error Handling

**Missing Agreement Score:**

- Display "N/A" instead of percentage
- No error thrown, card renders normally
- Fallback chain: `debate_agreement_score` → `debate_v1.consensus.agreement_score` → N/A

**Invalid Agreement Score:**

- Non-numeric values treated as missing
- Display "N/A"
- No console errors

**Missing Debate Data:**

- Agreement score section omitted or shows N/A
- Card renders with other available data
- No impact on other card fields

### Backend Error Handling

**Missing Agreement Score in Debate Data:**

- No tier adjustment applied
- Tier calculation continues with other factors
- No error logged (expected for setups without debate data)

**Invalid Agreement Score Value:**

- Clamp to [0, 100] range if numeric but out of bounds
- Log warning: `"Agreement score {value} clamped to valid range for {ticker}"`
- Continue processing with clamped value

**Null/Undefined Agreement Score:**

- Treat as missing data
- No tier adjustment
- Log info message if debate data exists but agreement score is missing

**Debate Data Structure Issues:**

- Check `isinstance(debate, dict)` before accessing
- Check `isinstance(consensus, dict)` before accessing
- Gracefully skip adjustment if structure is invalid
- No error thrown, processing continues

### Tier Calculation Error Handling

**Downgrade Logic:**

- Always check current tier before downgrading
- Use tier map with default fallback to "D"
- Never produce invalid tier values (only A/B/C/D allowed)

**Order of Operations:**

- Agreement adjustment happens after YOLO adjustments
- Agreement adjustment happens before debate state guardrails
- If multiple adjustments conflict, later adjustments override earlier ones
- Final tier is always valid (A/B/C/D)

## Testing Strategy

### Dual Testing Approach

This feature requires both unit tests and property-based tests for comprehensive coverage:

**Unit Tests** focus on:

- Specific examples of agreement score display formatting
- Edge cases: null values, NaN, undefined
- Specific tier downgrade scenarios (A→B, B→C, C→D, D→D)
- Integration between components (debate engine → report generator → API → frontend)
- Logging behavior for warnings

**Property-Based Tests** focus on:

- Universal properties across all agreement score values
- Tier adjustment rules hold for all score ranges
- Data flow preservation across the pipeline
- Validation and clamping for all possible input values
- Error resilience with random invalid inputs

### Property-Based Testing Configuration

**Library:** Hypothesis (Python backend), fast-check (JavaScript frontend)

**Test Configuration:**

- Minimum 100 iterations per property test
- Each test tagged with: `Feature: agreement-score-display-and-tier-integration, Property {number}: {property_text}`

**Generator Strategies:**

For agreement scores:

```python
# Valid agreement scores (40-100 range from debate engine)
valid_agreement_scores = st.floats(min_value=40.0, max_value=100.0)

# All possible numeric values (for validation testing)
any_numeric = st.floats(allow_nan=True, allow_infinity=True)

# Missing values
missing_values = st.sampled_from([None, float('nan')])

# Out of range values (for clamping tests)
out_of_range = st.one_of(
    st.floats(max_value=-0.1),
    st.floats(min_value=100.1, max_value=200.0)
)
```

For tiers:

```python
# Valid tiers
valid_tiers = st.sampled_from(["A", "B", "C", "D"])

# Tiers that can be downgraded
downgradeable_tiers = st.sampled_from(["A", "B", "C"])
```

For setup rows:

```python
# Setup row with debate data
setup_with_debate = st.fixed_dictionaries({
    "ticker": st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=("Lu",))),
    "setup_tier": valid_tiers,
    "score": st.floats(min_value=0.0, max_value=100.0),
    "debate_v1": st.fixed_dictionaries({
        "consensus": st.fixed_dictionaries({
            "agreement_score": valid_agreement_scores,
            "consensus_state": st.sampled_from(["ready", "conditional", "watch"]),
            "consensus_bias": st.sampled_from(["bullish", "bearish", "neutral"])
        })
    })
})
```

### Unit Test Coverage

**Frontend Tests:**

1. Test agreement score display in compact card with valid score
2. Test agreement score display in audit card with valid score
3. Test "N/A" display when score is null
4. Test "N/A" display when score is undefined
5. Test "N/A" display when score is NaN
6. Test formatting with one decimal place (65.0%, 50.5%, 100.0%)
7. Test fallback from debate_agreement_score to debate_v1.consensus.agreement_score

**Backend Tests:**

1. Test tier downgrade A→B when agreement < 50%
2. Test tier downgrade B→C when agreement < 50%
3. Test tier downgrade C→D when agreement < 50%
4. Test tier D stays D when agreement < 50%
5. Test no tier change when agreement >= 50%
6. Test no tier change when agreement is missing
7. Test agreement score extraction from debate_v1
8. Test agreement score clamping (negative → 0, >100 → 100)
9. Test warning logged when score is clamped
10. Test warning logged when score is missing but debate data exists

**Integration Tests:**

1. Test full data flow: debate engine → report generator → API → frontend
2. Test tier calculation order: score → YOLO → agreement → debate state
3. Test API response includes debate_agreement_score field
4. Test frontend renders agreement score from API data

### Property-Based Test Mapping

Each correctness property maps to a property-based test:

**Property 1 → Test:** `test_agreement_score_display_on_cards`

- Generate random setup rows with valid agreement scores
- Render cards and verify HTML contains formatted percentage

**Property 2 → Test:** `test_missing_agreement_score_displays_na`

- Generate random setup rows with null/undefined/NaN agreement scores
- Render cards and verify HTML contains "N/A"

**Property 3 → Test:** `test_agreement_above_50_preserves_tier`

- Generate random setups with agreement >= 50%
- Apply adjustment and verify tier unchanged

**Property 4 → Test:** `test_agreement_below_50_downgrades_tier`

- Generate random setups with agreement < 50% and tier A/B/C
- Apply adjustment and verify tier downgraded by one level

**Property 5 → Test:** `test_tier_d_cannot_be_downgraded`

- Generate random setups with tier D and agreement < 50%
- Apply adjustment and verify tier remains D

**Property 6 → Test:** `test_missing_agreement_preserves_tier`

- Generate random setups with missing agreement scores
- Apply adjustment and verify tier unchanged

**Property 7 → Test:** `test_debate_engine_produces_agreement_scores`

- Generate random valid setup rows
- Call build_setup_debate and verify consensus.agreement_score exists and is numeric

**Property 8 → Test:** `test_agreement_score_data_flow_round_trip`

- Generate random setups with debate_v1.consensus.agreement_score
- Process through report generator
- Verify debate_agreement_score matches original value

**Property 9 → Test:** `test_backend_api_includes_agreement_score`

- Generate random setup rows
- Call API endpoint
- Verify response includes debate_agreement_score field

**Property 10 → Test:** `test_agreement_score_validation_and_clamping`

- Generate random numeric values (including out of range, NaN, infinity)
- Process through validation
- Verify clamping for out-of-range, missing treatment for non-numeric, no errors

**Property 11 → Test:** `test_invalid_agreement_score_logging`

- Generate random invalid agreement scores (out of range or missing)
- Process through system
- Verify warning logged

### Test Execution

**Backend tests:**

```bash
pytest tests/test_agreement_score_tier_integration.py -v
pytest tests/test_agreement_score_tier_integration_property.py -v --hypothesis-show-statistics
```

**Frontend tests:**

```bash
# Run in browser console or with Jest/Vitest
npm test -- agreement-score
```

**Integration tests:**

```bash
pytest tests/test_agreement_score_integration.py -v
```

### Success Criteria

Tests pass when:

1. All unit tests pass (100% pass rate)
2. All property-based tests pass 100 iterations without counterexamples
3. Integration tests verify end-to-end data flow
4. No regressions in existing tier calculation tests
5. Frontend renders agreement scores correctly in manual testing
6. Tier downgrades appear in logs when agreement < 50%
