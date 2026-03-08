# Requirements Document

## Introduction

This feature adds agreement score visibility and integration into the trader_koo platform. The debate engine currently produces agreement scores (40-100%) representing consensus between Bull and Bear researchers, but these scores are neither displayed to users nor factored into tier calculations. This feature will surface agreement scores on setup cards and incorporate them as a risk factor in tier scoring logic.

## Glossary

- **Agreement_Score**: A numeric value (40-100%) representing the level of consensus between Bull and Bear researchers in the debate engine
- **Setup_Card**: A UI component displaying a trading setup with ticker, tier, bias, family, and score information
- **Tier**: A letter grade (A/B/C/D) assigned to setups based on their quality score
- **Tier_Calculator**: The system component that assigns tier grades based on setup scores and other factors
- **Dashboard**: The frontend interface displaying setup cards to users
- **Debate_Engine**: The system that generates Bull/Bear analysis and produces agreement scores

## Requirements

### Requirement 1: Display Agreement Score on Setup Cards

**User Story:** As a trader, I want to see the agreement score on each setup card, so that I can assess the level of consensus between Bull and Bear researchers.

#### Acceptance Criteria

1. WHEN a setup card is rendered in the compact list view, THE Dashboard SHALL display the agreement score as a percentage
2. WHEN a setup card is rendered in the audit cards view, THE Dashboard SHALL display the agreement score as a percentage
3. THE Dashboard SHALL format the agreement score with one decimal place (e.g., "65.0%")
4. THE Dashboard SHALL retrieve the agreement score from the debate_agreement_score field in the row data
5. WHEN the agreement score is missing or null, THE Dashboard SHALL display "N/A" instead of a percentage

### Requirement 2: Incorporate Agreement Score into Tier Calculation

**User Story:** As a system, I want to adjust tier assignments based on agreement scores, so that low consensus (high debate) is treated as a risk factor.

#### Acceptance Criteria

1. WHEN calculating a setup tier, THE Tier_Calculator SHALL retrieve the agreement score from the debate consensus data
2. WHEN the agreement score is above 75%, THE Tier_Calculator SHALL apply no tier adjustment
3. WHEN the agreement score is between 50% and 75% (inclusive), THE Tier_Calculator SHALL apply no tier adjustment
4. WHEN the agreement score is below 50%, THE Tier_Calculator SHALL downgrade the tier by one level
5. THE Tier_Calculator SHALL apply tier downgrades after score-based tier assignment (A→B, B→C, C→D)
6. WHEN a tier D setup would be downgraded, THE Tier_Calculator SHALL keep it at tier D (no tier E exists)
7. WHEN the agreement score is missing or null, THE Tier_Calculator SHALL apply no tier adjustment

### Requirement 3: Preserve Agreement Score Data Flow

**User Story:** As a system, I want to ensure agreement scores flow correctly from the debate engine through report generation to the frontend, so that all components have access to the data.

#### Acceptance Criteria

1. THE Debate_Engine SHALL continue to produce agreement_score values in the consensus object
2. THE Report_Generator SHALL extract agreement_score from debate_v1.consensus.agreement_score
3. THE Report_Generator SHALL store the agreement score in the debate_agreement_score field
4. THE Backend_API SHALL include debate_agreement_score in the setup data returned to the frontend
5. FOR ALL setups with debate data, the agreement score SHALL be available in the row data structure

### Requirement 4: Agreement Score Validation

**User Story:** As a system, I want to validate agreement score values, so that invalid data does not cause errors or incorrect tier assignments.

#### Acceptance Criteria

1. WHEN processing an agreement score, THE System SHALL verify it is a numeric value
2. WHEN an agreement score is outside the range 0-100, THE System SHALL clamp it to the valid range
3. WHEN an agreement score is null or undefined, THE System SHALL treat it as missing data
4. THE System SHALL log a warning when agreement scores are clamped or missing
5. THE System SHALL continue processing even when agreement scores are invalid or missing
