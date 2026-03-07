# Implementation Plan: Enterprise Platform Upgrade

## Overview

This implementation plan transforms trader_koo from a personal swing trading tool into an enterprise-grade platform. The plan follows a 6-phase migration strategy prioritizing security hardening and VIX methodology improvements (P0), followed by core enterprise features (P1), competitive features (P2), and advanced features (P3).

The implementation uses Python 3.11+ with FastAPI, PostgreSQL, Redis, and maintains backward compatibility throughout the migration. Each phase includes comprehensive testing with both unit tests and property-based tests using hypothesis.

## Phase 1: Security Hardening (P0)

### Objectives

Implement all P0 security requirements without breaking existing functionality. Eliminate secret exposures, harden authentication boundaries, and implement defense-in-depth security layers.

- [x] 1. Implement API key validation and strict mode

  - [x] 1.1 Add ADMIN_STRICT_API_KEY environment variable with default "1"
    - Create config validation module in `trader_koo/config.py`
    - Add `ADMIN_STRICT_API_KEY` to Config class with default value "1"
    - Implement startup validation that checks for `TRADER_KOO_API_KEY` presence when strict mode enabled
    - Add minimum length validation (32 characters) for API keys
    - Raise `ConfigError` with clear error message if validation fails
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ]\* 1.2 Write property test for API key length validation
    - **Property 1: API Key Length Validation**
    - **Validates: Requirements 1.5**
    - Use hypothesis to generate API key strings of varying lengths
    - Verify keys < 32 chars are rejected, keys >= 32 chars are accepted
    - Test with empty strings, whitespace, special characters
  - [ ]\* 1.3 Write unit tests for API key validation
    - Test startup fails without API key in strict mode
    - Test startup succeeds with valid API key
    - Test startup succeeds when strict mode disabled
    - Test error messages are clear and actionable
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Implement LLM output validation and sanitization

  - [ ] 2.1 Define JSON schemas for all LLM response formats
    - Create `trader_koo/llm/schemas.py` with Pydantic models
    - Define schemas for: narrative generation, pattern explanation, regime analysis
    - Include field type constraints, required fields, and length limits
    - _Requirements: 2.1, 7.1, 7.4, 7.5_
  - [ ] 2.2 Implement LLM output validator with fallback logic
    - Create `trader_koo/llm/validator.py` with validation functions
    - Validate LLM output against schemas using Pydantic
    - Implement length enforcement (narrative: 5000 chars, summary: 1000 chars)
    - Truncate with ellipsis if exceeds limits
    - Log validation failures with request context
    - _Requirements: 2.2, 2.3, 2.7, 7.3, 7.6, 7.7_
  - [ ] 2.3 Implement HTML/script sanitization
    - Strip HTML tags using bleach or html.escape
    - Remove script content and dangerous attributes
    - Escape special characters before HTML rendering
    - _Requirements: 2.4, 2.5_
  - [ ] 2.4 Implement deterministic fallback for LLM failures
    - Create template-based narrative generation
    - Use rule-based pattern explanations
    - Ensure fallback always returns valid schema-compliant output
    - _Requirements: 2.6_
  - [ ]\* 2.5 Write property test for LLM output validation
    - **Property 2: LLM Output Schema Validation with Fallback**
    - Generate random dictionaries with hypothesis
    - Verify invalid outputs trigger fallback
    - Verify all outputs (validated or fallback) pass schema validation
  - [ ]\* 2.6 Write unit tests for LLM sanitization
    - Test HTML tag stripping
    - Test script content removal
    - Test length truncation
    - Test special character escaping
    - _Requirements: 2.3, 2.4, 2.5_

- [x] 3. Implement email token expiration

  - [x] 3.1 Add expiration fields to email token model
    - Update database schema with `expires_at` timestamp
    - Set expiration to 7 days from creation
    - Store creation timestamp
    - _Requirements: 3.1, 3.2_
  - [x] 3.2 Implement token age validation
    - Check token age on authentication attempt
    - Reject tokens older than 7 days
    - Return clear error message for expired tokens
    - _Requirements: 3.3, 3.4_
  - [x] 3.3 Add expiration timestamp to email body
    - Include token expiration date in email template
    - Format as user-friendly date string
    - _Requirements: 3.5_
  - [x] 3.4 Implement admin token revocation endpoint
    - Create `/api/admin/tokens/revoke` endpoint
    - Accept token ID and mark as revoked
    - Reject authentication with revoked tokens
    - _Requirements: 3.6_
  - [ ]\* 3.5 Write property test for email token expiration
    - **Property 4: Email Token Expiration**
    - Generate random creation and auth times
    - Verify tokens expire exactly after 7 days
    - Test edge cases around expiration boundary
  - [ ]\* 3.6 Write unit tests for token revocation
    - Test admin can revoke token
    - Test revoked token is rejected
    - Test non-admin cannot revoke tokens

- [x] 4. Implement CORS restrictive defaults

  - [x] 4.1 Add CORS configuration with empty default
    - Default `TRADER_KOO_CORS_ORIGINS` to empty list
    - Parse comma-separated origin list from env var
    - _Requirements: 4.1, 4.2_
  - [x] 4.2 Implement origin validation
    - Validate origins match `https://` or `http://localhost` pattern
    - Reject invalid origin formats
    - _Requirements: 4.3_
  - [x] 4.3 Implement CORS request filtering
    - Reject requests from non-allowed origins
    - Log rejected CORS requests with origin and endpoint
    - _Requirements: 4.4, 4.5_
  - [x] 4.4 Add development mode localhost exception
    - Allow `http://localhost:*` in development mode
    - _Requirements: 4.6_
  - [x] 4.5 Set CORS credentials header
    - Include `Access-Control-Allow-Credentials: false` by default
    - _Requirements: 4.7_
  - [ ]\* 4.6 Write property test for CORS origin validation
    - **Property 6: CORS Origin Validation**
    - Generate random origin strings
    - Verify only valid patterns are accepted
  - [ ]\* 4.7 Write unit tests for CORS filtering
    - Test allowed origins pass
    - Test disallowed origins are rejected
    - Test rejection logging

- [x] 5. Implement admin auth boundary verification

  - [x] 5.1 Create admin endpoint registry
    - Maintain list of all `/api/admin/*` paths
    - Auto-discover admin endpoints at startup
    - _Requirements: 5.1_
  - [x] 5.2 Implement startup auth verification
    - Check all admin endpoints have auth middleware
    - Refuse to start if any endpoint lacks auth
    - Log error with missing endpoint details
    - _Requirements: 5.2, 5.3_
  - [x] 5.3 Create admin endpoint listing API
    - Endpoint: `/api/admin/routes`
    - Return all protected routes with auth status
    - _Requirements: 5.4_
  - [x] 5.4 Add auth decorator requirement for new endpoints
    - Require explicit `@require_auth` decorator
    - Document in developer guidelines
    - _Requirements: 5.6_
  - [ ]\* 5.5 Write unit tests for admin endpoint authentication
    - Test all admin endpoints require auth (T-040)
    - Test unauthenticated requests return 401
    - Test authenticated requests succeed
    - _Requirements: 5.5_

- [x] 6. Implement secret exposure hardening

  - [ ] 6.1 Define secret patterns list
    - List secret env var names: API keys, passwords, tokens
    - Define regex patterns for secret detection
    - _Requirements: 6.1_
  - [ ] 6.2 Implement log sanitization
    - Intercept all log calls
    - Redact values matching secret patterns
    - Replace with `[REDACTED]`
    - _Requirements: 6.2, 6.3_
  - [ ] 6.3 Implement error response sanitization
    - Strip env vars from stack traces
    - Remove config values from error responses
    - _Requirements: 6.4_
  - [ ] 6.4 Validate status/health endpoints
    - Ensure `/api/status` doesn't expose secrets
    - Ensure `/api/health` doesn't expose secrets
    - Ensure `/api/config` only returns non-sensitive config
    - _Requirements: 6.5, 6.6_
  - [ ]\* 6.5 Write property test for secret redaction
    - **Property 8: Secret Redaction in Logs**
    - Generate dicts with secret-like keys
    - Verify secrets are redacted in logs
  - [ ]\* 6.6 Write unit tests for secret sanitization (T-041)
    - Test secrets not in log output
    - Test secrets not in error responses
    - Test status endpoints don't expose secrets
    - _Requirements: 6.7_

- [x] 7. Implement LLM output guardrail enforcement
  - [ ] 7.1 Define comprehensive JSON schemas
    - Use JSON Schema validator library
    - Define schemas for all LLM response types
    - _Requirements: 7.1_
  - [ ] 7.2 Implement schema validation with detailed errors
    - Validate against schema on every LLM response
    - Log validation errors with schema path and actual value
    - _Requirements: 7.2, 7.3_
  - [ ] 7.3 Enforce required fields and type constraints
    - Check required fields are present
    - Validate field types (string, number, boolean, array)
    - Enforce string length limits
    - _Requirements: 7.4, 7.5, 7.6_
  - [ ] 7.4 Implement failure counter and fallback
    - Increment counter on validation failure
    - Use fallback content when validation fails
    - _Requirements: 7.7_
  - [ ] 7.5 Expose validation metrics
    - Add LLM validation failure rate to `/api/admin/health`
    - Track failures over time
    - _Requirements: 7.8_
  - [ ]\* 7.6 Write property test for LLM guardrails (T-042)
    - **Property 2: LLM Output Schema Validation with Fallback**
    - Test with random invalid LLM outputs
    - Verify fallback is used
    - Verify metrics are updated

## Phase 2: VIX Methodology Improvements (P0)

### Objectives

Improve VIX analysis accuracy and reliability with multi-source redundancy, adaptive thresholds, and enhanced pattern detection.

- [x] 8. Implement VIX trap/reclaim terminology

  - [x] 8.1 Update VIX pattern labeling
    - Label failed breakouts as "failed_breakout" or "bull_trap"
    - Label failed breakdowns as "failed_breakdown" or "bear_trap"
    - Label support reclaims as "support_reclaim"
    - Label resistance reclaims as "resistance_reclaim"
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 8.2 Add glossary definitions to UI
    - Create glossary component with trap/reclaim definitions
    - Display in VIX analysis tab
    - _Requirements: 8.5_
  - [x] 8.3 Ensure consistent terminology across surfaces
    - Update API responses
    - Update UI displays
    - Update report templates
    - Update email templates
    - _Requirements: 8.6_
  - [ ]\* 8.4 Write unit tests for trap/reclaim labeling
    - Test bull trap detection
    - Test bear trap detection
    - Test support reclaim detection
    - Test resistance reclaim detection

- [x] 9. Implement term structure fallback

  - [x] 9.1 Add VIX6M fallback logic
    - Try VIX3M first
    - Fall back to VIX6M if VIX3M unavailable
    - _Requirements: 9.1_
  - [x] 9.2 Implement synthetic term structure calculation
    - Calculate from VXX/UVXY when both VIX3M and VIX6M unavailable
    - _Requirements: 9.2_
  - [x] 9.3 Add source labeling
    - Label as "VIX3M", "VIX6M", or "synthetic"
    - Include in all displays and API responses
    - _Requirements: 9.3_
  - [x] 9.4 Add logging for data source
    - Log which source was used
    - Log availability status
    - _Requirements: 9.4_
  - [x] 9.5 Handle complete unavailability
    - Display "Term structure unavailable" message
    - _Requirements: 9.5_
  - [x] 9.6 Add timestamp to displays
    - Include data source timestamp
    - _Requirements: 9.6_
  - [ ]\* 9.7 Write unit tests for term structure fallback
    - Test VIX3M success
    - Test VIX6M fallback
    - Test synthetic calculation
    - Test complete unavailability

- [x] 10. Implement key level source labeling

  - [x] 10.1 Add source field to key levels
    - Label as "pivot_cluster", "ma_anchor", or "fallback"
    - _Requirements: 10.1_
  - [x] 10.2 Display source in UI
    - Show source label next to each level
    - _Requirements: 10.2_
  - [x] 10.3 Include source in API responses
    - Add "source" field to level objects
    - _Requirements: 10.3_
  - [x] 10.4 Implement level prioritization
    - Prioritize pivot_cluster > ma_anchor > fallback
    - _Requirements: 10.4_
  - [x] 10.5 Add source legend to UI
    - Explain each source type
    - _Requirements: 10.5_
  - [x] 10.6 Include source in reports
    - Add to narrative text
    - _Requirements: 10.6_
  - [ ]\* 10.7 Write unit tests for key level source labeling
    - Test source assignment
    - Test prioritization
    - Test UI display

- [x] 11. Implement VIX percentile prominence

  - [x] 11.1 Calculate VIX percentile
    - Use 252-day rolling window
    - _Requirements: 11.1_
  - [x] 11.2 Display prominently in health score
    - Large font in summary card
    - _Requirements: 11.2_
  - [x] 11.3 Add color coding
    - Green: 0-30, Yellow: 30-70, Red: 70-100
    - _Requirements: 11.3_
  - [x] 11.4 Include in health score calculation
    - Use as primary factor
    - _Requirements: 11.4_
  - [x] 11.5 Add warning for elevated volatility
    - Display when percentile > 80
    - _Requirements: 11.5_
  - [x] 11.6 Include in API responses
    - Add to all regime analysis responses
    - _Requirements: 11.6_
  - [ ]\* 11.7 Write unit tests for VIX percentile
    - Test calculation
    - Test color coding
    - Test warning display

- [x] 12. Implement multi-source data redundancy

  - [x] 12.1 Implement yfinance as primary source
    - Fetch data from yfinance first
    - _Requirements: 12.1_
  - [x] 12.2 Implement Alpha Vantage fallback
    - Try Alpha Vantage if yfinance fails
    - _Requirements: 12.2_
  - [x] 12.3 Implement CSV fallback
    - Load from local CSV if both fail
    - _Requirements: 12.3_
  - [x] 12.4 Add source logging
    - Log which source was used for each fetch
    - _Requirements: 12.4_
  - [x] 12.5 Track success/failure rates
    - Expose via admin health endpoint
    - _Requirements: 12.5_
  - [x] 12.6 Add alerting for high failure rates
    - Alert when primary source failure > 10%
    - _Requirements: 12.6_
  - [x] 12.7 Include source in API responses
    - Add data source and timestamp
    - _Requirements: 12.7_
  - [ ]\* 12.8 Write unit tests for multi-source redundancy
    - Test primary source success
    - Test secondary fallback
    - Test CSV fallback
    - Test failure tracking

- [x] 13. Implement adaptive compression thresholds

  - [x] 13.1 Calculate dynamic thresholds
    - Based on 90-day VIX percentile
    - _Requirements: 13.1_
  - [x] 13.2 Use tight thresholds in low vol
    - 20th/80th percentile when VIX percentile < 30
    - _Requirements: 13.2_
  - [x] 13.3 Use moderate thresholds in normal vol
    - 25th/75th percentile when VIX percentile 30-70
    - _Requirements: 13.3_
  - [x] 13.4 Use wide thresholds in high vol
    - 30th/70th percentile when VIX percentile > 70
    - _Requirements: 13.4_
  - [x] 13.5 Display current thresholds
    - Show in VIX analysis tab
    - _Requirements: 13.5_
  - [x] 13.6 Label compression signals
    - Include threshold regime (tight/moderate/wide)
    - _Requirements: 13.6_
  - [ ]\* 13.7 Write unit tests for adaptive thresholds
    - Test threshold calculation
    - Test regime classification
    - Test signal labeling

- [x] 14. Implement enhanced trap/reclaim detection
  - [x] 14.1 Detect bull traps
    - VIX breaks resistance, closes below within 3 bars
    - Calculate confidence from volume and reversal speed
    - _Requirements: 14.1, 14.3_
  - [x] 14.2 Detect bear traps
    - VIX breaks support, closes above within 3 bars
    - Calculate confidence from volume and reversal speed
    - _Requirements: 14.2, 14.3_
  - [x] 14.3 Detect support reclaims
    - VIX closes above broken support for 2+ bars
    - _Requirements: 14.4_
  - [x] 14.4 Detect resistance reclaims
    - VIX closes below broken resistance for 2+ bars
    - _Requirements: 14.5_
  - [x] 14.5 Include in regime analysis
    - Add patterns with confidence scores
    - _Requirements: 14.6_
  - [x] 14.6 Display on VIX chart
    - Add distinct visual markers
    - _Requirements: 14.7_
  - [ ]\* 14.7 Write unit tests for trap/reclaim detection
    - Test bull trap detection
    - Test bear trap detection
    - Test support reclaim detection
    - Test resistance reclaim detection
    - Test confidence calculation

## Phase 3: Core Enterprise Features (P1)

### Objectives

Enable multi-user enterprise deployment with RBAC, audit logging, monitoring, and operational features.

- [x] 15. Implement audit logging

  - [x] 15.1 Create audit log database schema
    - Create audit_logs table with partitioning
    - Add indexes for timestamp, user_id, event_type
    - _Requirements: 15.5_
  - [x] 15.2 Implement audit logger service
    - Log admin API requests
    - Log authentication attempts
    - Log data modifications
    - _Requirements: 15.1, 15.2, 15.3_
  - [x] 15.3 Ensure immutability
    - Append-only, no updates or deletes
    - _Requirements: 15.4_
  - [x] 15.4 Implement query endpoint
    - Filter by date, user, action type
    - _Requirements: 15.6_
  - [x] 15.5 Implement retention policy
    - Retain for minimum 90 days
    - _Requirements: 15.7_
  - [x] 15.6 Implement external export
    - Export to S3/Azure Blob for long-term retention
    - _Requirements: 15.8_
  - [ ]\* 15.7 Write unit tests for audit logging
    - Test log creation
    - Test immutability
    - Test querying
    - Test retention

- [x] 16. Implement multi-user RBAC

  - [x] 16.1 Create user database schema
    - Create users, sessions, email_tokens tables
    - _Requirements: 16.2_
  - [x] 16.2 Implement authentication service
    - API key auth
    - JWT auth
    - Email token auth
    - _Requirements: 16.3_
  - [x] 16.3 Implement role-based permissions
    - Admin: full access
    - Analyst: read/write analysis, no admin
    - Viewer: read-only
    - _Requirements: 16.4, 16.5, 16.6, 16.7_
  - [x] 16.4 Implement user management endpoints
    - Create, update, deactivate users
    - _Requirements: 16.8_
  - [x] 16.5 Log role changes
    - Add to audit log
    - _Requirements: 16.9_
  - [x] 16.6 Implement password complexity
    - Min 12 chars, uppercase, lowercase, number, special
    - _Requirements: 16.10_
  - [ ]\* 16.7 Write unit tests for RBAC
    - Test authentication
    - Test authorization
    - Test role enforcement
    - Test password validation

- [~] 17. Implement API rate limiting

  - [ ] 17.1 Implement rate limiter service
    - Per-IP limits for public endpoints (100/min)
    - Per-user limits for auth endpoints (1000/hour)
    - _Requirements: 17.1, 17.2_
  - [ ] 17.2 Return HTTP 429 when exceeded
    - Include Retry-After header
    - _Requirements: 17.3_
  - [ ] 17.3 Use sliding window algorithm
    - Accurate rate calculation
    - _Requirements: 17.4_
  - [ ] 17.4 Store state in Redis
    - _Requirements: 17.5_
  - [ ] 17.5 Implement admin status endpoint
    - View current rate limit status
    - _Requirements: 17.6_
  - [ ] 17.6 Allow admin overrides
    - Temporarily increase limits for specific users
    - _Requirements: 17.7_
  - [ ] 17.7 Log violations
    - Log with IP, user, endpoint
    - _Requirements: 17.8_
  - [ ]\* 17.8 Write unit tests for rate limiting
    - Test limit enforcement
    - Test sliding window
    - Test admin overrides

- [~] 18. Implement webhook notifications

  - [ ] 18.1 Create webhook database schema
    - Create webhooks, webhook_deliveries tables
    - _Requirements: 18.1_
  - [ ] 18.2 Implement webhook registration
    - Support URL, events, auth headers, HMAC
    - _Requirements: 18.1, 18.8_
  - [ ] 18.3 Implement event triggering
    - Trigger for: pattern_detected, regime_change, alert_triggered, report_generated
    - _Requirements: 18.2_
  - [ ] 18.4 Implement delivery with retry
    - POST JSON payload
    - Retry 3 times with exponential backoff
    - _Requirements: 18.3, 18.4, 18.5_
  - [ ] 18.5 Log deliveries
    - Log status code, response time
    - _Requirements: 18.6_
  - [ ] 18.6 Implement delivery history endpoint
    - View history and failure rate
    - _Requirements: 18.7_
  - [ ] 18.7 Validate webhook URLs
    - Require HTTPS (except localhost)
    - _Requirements: 18.9_
  - [ ] 18.8 Implement timeout
    - 10 second timeout
    - _Requirements: 18.10_
  - [ ]\* 18.9 Write unit tests for webhooks
    - Test registration
    - Test delivery
    - Test retry logic
    - Test HMAC signatures

- [~] 19. Implement data export

  - [ ] 19.1 Implement dashboard export
    - Support CSV, JSON, Excel formats
    - _Requirements: 19.1_
  - [ ] 19.2 Implement pattern export
    - Include all detection metadata
    - _Requirements: 19.2_
  - [ ] 19.3 Implement VIX analysis export
    - Include full time series
    - _Requirements: 19.3_
  - [ ] 19.4 Implement audit log export
    - CSV format
    - _Requirements: 19.4_
  - [ ] 19.5 Include metadata
    - Column headers, data types
    - _Requirements: 19.5_
  - [ ] 19.6 Support date range filtering
    - _Requirements: 19.6_
  - [ ] 19.7 Enforce rate limits
    - 10 exports per hour per user
    - _Requirements: 19.7_
  - [ ] 19.8 Log export requests
    - Add to audit log
    - _Requirements: 19.8_
  - [ ]\* 19.9 Write unit tests for data export
    - Test each export format
    - Test filtering
    - Test rate limiting

- [~] 20. Implement backup and restore

  - [ ] 20.1 Create backup database schema
    - Create backups table
    - _Requirements: 20.1_
  - [ ] 20.2 Implement backup service
    - Full backup daily at 00:00 UTC
    - Incremental every 6 hours
    - _Requirements: 20.1, 20.2_
  - [ ] 20.3 Store in external storage
    - Support S3, Azure Blob, local volume
    - _Requirements: 20.3_
  - [ ] 20.4 Implement retention policy
    - Daily: 30 days, Weekly: 90 days
    - _Requirements: 20.4, 20.5_
  - [ ] 20.5 Implement manual backup endpoint
    - _Requirements: 20.6_
  - [ ] 20.6 Implement list backups endpoint
    - Show size and timestamp
    - _Requirements: 20.7_
  - [ ] 20.7 Implement restore endpoint
    - With confirmation step
    - _Requirements: 20.8_
  - [ ] 20.8 Verify backup integrity
    - Use checksum
    - _Requirements: 20.9_
  - [ ] 20.9 Alert on failures
    - _Requirements: 20.10_
  - [ ]\* 20.10 Write unit tests for backup/restore
    - Test backup creation
    - Test integrity verification
    - Test restore

- [~] 21. Implement health monitoring

  - [ ] 21.1 Implement Prometheus metrics
    - Expose at /metrics endpoint
    - _Requirements: 21.1_
  - [ ] 21.2 Track request metrics
    - Count, latency, error rate per endpoint
    - _Requirements: 21.2_
  - [ ] 21.3 Track database metrics
    - Query count and latency
    - _Requirements: 21.3_
  - [ ] 21.4 Track LLM metrics
    - Request count, latency, success rate, token usage
    - _Requirements: 21.4_
  - [ ] 21.5 Track YOLO metrics
    - Detection count and processing time
    - _Requirements: 21.5_
  - [ ] 21.6 Track data ingestion metrics
    - Success rate per source
    - _Requirements: 21.6_
  - [ ] 21.7 Implement health dashboard
    - Admin endpoint showing all metrics
    - _Requirements: 21.7_
  - [ ] 21.8 Implement alerting
    - Alert on error rate > 5%
    - Alert on latency p95 > 2s
    - _Requirements: 21.8, 21.9_
  - [ ] 21.9 Track uptime
    - Calculate SLA compliance (99.9% target)
    - _Requirements: 21.10_
  - [ ]\* 21.10 Write unit tests for health monitoring
    - Test metric recording
    - Test alerting
    - Test SLA calculation

- [~] 22. Implement white-label support
  - [ ] 22.1 Implement logo upload
    - Admin endpoint for custom logo
    - _Requirements: 22.1_
  - [ ] 22.2 Implement color scheme config
    - Primary, secondary, accent colors
    - _Requirements: 22.2_
  - [ ] 22.3 Implement custom domain config
    - Via environment variable
    - _Requirements: 22.3_
  - [ ] 22.4 Implement custom email templates
    - With variable substitution
    - _Requirements: 22.4_
  - [ ] 22.5 Implement custom footer
    - Text and links
    - _Requirements: 22.5_
  - [ ] 22.6 Store branding in database
    - With versioning
    - _Requirements: 22.6_
  - [ ] 22.7 Apply branding to all surfaces
    - Dashboard, reports, emails
    - _Requirements: 22.7_
  - [ ] 22.8 Validate logo requirements
    - PNG/SVG, max 500KB
    - _Requirements: 22.8_
  - [ ]\* 22.9 Write unit tests for white-label
    - Test branding application
    - Test logo validation

## Phase 4: Repository Management & CI/CD (P0)

### Objectives

Establish proper version control, CI/CD pipeline, and documentation practices.

- [~] 41. Configure .gitignore

  - [ ] 41.1 Exclude planning documents
    - Exclude planning/\*.md except planning/README.md
    - Exclude planning/modules/\*_/_.md
    - _Requirements: 41.1, 41.2_
  - [ ] 41.2 Exclude memory documents
    - Exclude memory/\*.md
    - _Requirements: 41.3_
  - [ ] 41.3 Include critical docs
    - Include AGENTS.md, ARCHITECTURE.md, SECURITY.md, LICENSE, README.md
    - _Requirements: 41.4_
  - [ ] 41.4 Exclude development artifacts
    - Exclude .env, \*.log, **pycache**/, .pytest_cache/, .venv/
    - _Requirements: 41.5_
  - [ ] 41.5 Exclude data directory
    - Exclude data/ (database, reports, logs)
    - _Requirements: 41.6_
  - [ ] 41.6 Include .env.example
    - Template for environment variables
    - _Requirements: 41.7_

- [~] 42. Implement CI/CD pipeline

  - [ ] 42.1 Create GitHub Actions workflow
    - Run on PR and push to main
    - _Requirements: 42.1, 42.2_
  - [ ] 42.2 Add linting checks
    - flake8, black, mypy
    - _Requirements: 42.3_
  - [ ] 42.3 Add unit tests with coverage
    - Minimum 70% coverage
    - _Requirements: 42.4_
  - [ ] 42.4 Add security scanning
    - bandit, safety
    - _Requirements: 42.5_
  - [ ] 42.5 Fail on check failures
    - Block merge if any check fails
    - _Requirements: 42.6_
  - [ ] 42.6 Add deployment workflow
    - Deploy to Railway on main branch
    - _Requirements: 42.7, 42.8_
  - [ ]\* 42.7 Test CI/CD pipeline
    - Verify all checks run
    - Verify deployment works

- [~] 43. Document version control practices
  - [ ] 43.1 Create planning/README.md
    - Document which files are tracked
    - _Requirements: 43.1, 43.2_
  - [ ] 43.2 Explain module docs exclusion
    - Working files excluded from git
    - _Requirements: 43.3_
  - [ ] 43.3 Add .gitattributes
    - Proper line ending handling
    - _Requirements: 43.4_
  - [ ] 43.4 Document rollup regeneration
    - How to regenerate from modules
    - _Requirements: 43.5_
  - [ ] 43.5 Add pre-commit hooks
    - Prevent accidental commits
    - _Requirements: 43.6_

## Summary

This implementation plan covers 43 major tasks across 4 phases:

- **Phase 1 (P0)**: 7 tasks - Security hardening
- **Phase 2 (P0)**: 7 tasks - VIX methodology improvements
- **Phase 3 (P1)**: 8 tasks - Core enterprise features
- **Phase 4 (P0)**: 3 tasks - Repository management & CI/CD

Each task includes detailed sub-tasks with requirement traceability. Optional tasks are marked with `*`. Property-based tests are included for all correctness properties defined in the design document.
