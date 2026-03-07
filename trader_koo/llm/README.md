# LLM Output Validation and Sanitization

This module provides comprehensive validation and sanitization for LLM-generated content, implementing security requirements 2.1-2.7 and 7.1-7.7 from the enterprise platform upgrade specification.

## Components

### 1. Schemas (`schemas.py`)

Pydantic models for validating LLM response formats:

- **NarrativeGeneration**: Validates narrative text (observation, action, risk_note)
  - Max lengths: observation/action 5000 chars, risk_note 1000 chars
- **PatternExplanation**: Validates pattern analysis responses
  - Includes confidence scores (0-1 range)
- **RegimeAnalysis**: Validates market regime analysis
  - Summary max 1000 chars, analysis max 5000 chars
- **SetupRewrite**: Validates setup copy rewrites (used by existing code)
  - Observation max 260 chars, action max 180 chars, risk_note max 80 chars

### 2. Validator (`validator.py`)

Core validation logic with fallback support:

- **validate_llm_output()**: Validates LLM output against schema

  - Returns ValidationResult with success/failure status
  - Logs validation errors with context
  - Provides detailed error messages for debugging

- **Fallback generators**:
  - `generate_fallback_narrative()`: Template-based narrative generation
  - `generate_fallback_pattern_explanation()`: Rule-based pattern explanations
  - `generate_fallback_regime_analysis()`: Deterministic regime analysis

### 3. Sanitizer (`sanitizer.py`)

HTML/script sanitization and length enforcement:

- **truncate_with_ellipsis()**: Truncates text to max length with "..."
- **strip_html_tags()**: Removes all HTML tags
- **strip_script_content()**: Removes script tags and inline event handlers
- **escape_special_characters()**: Escapes HTML special characters
- **sanitize_text()**: Full sanitization (strips HTML/scripts, truncates)
- **sanitize_html()**: Sanitizes and escapes for HTML rendering
- **sanitize_llm_output()**: Sanitizes entire output dictionary

### 4. Fallback (`fallback.py`)

Deterministic fallback logic for LLM failures:

- **generate_template_narrative()**: Context-based narrative templates
- **generate_rule_based_pattern_explanation()**: Pattern explanation templates

## Usage Examples

### Basic Validation

```python
from trader_koo.llm import validate_llm_output, NarrativeGeneration

# Validate LLM output
llm_output = {
    "observation": "Market shows bullish momentum",
    "action": "Consider entry above resistance",
    "risk_note": "Use stop loss below support"
}

result = validate_llm_output(
    llm_output,
    NarrativeGeneration,
    context={"ticker": "AAPL", "source": "daily_report"}
)

if result.is_valid:
    # Use validated data
    narrative = result.data
    print(narrative.observation)
else:
    # Log errors and use fallback
    print(f"Validation failed: {result.errors}")
    # Use fallback logic
```

### Validation with Fallback

```python
from trader_koo.llm import validate_llm_output, NarrativeGeneration
from trader_koo.llm.validator import generate_fallback_narrative

# Try to validate LLM output
result = validate_llm_output(llm_output, NarrativeGeneration)

if result.is_valid:
    narrative = result.data.model_dump()
else:
    # Use deterministic fallback
    narrative = generate_fallback_narrative(context)
```

### Sanitization

```python
from trader_koo.llm import sanitize_text, sanitize_html, truncate_with_ellipsis

# Strip HTML and scripts
clean_text = sanitize_text("<p>Hello <script>alert('xss')</script></p>")
# Result: "Hello"

# Sanitize and truncate
short_text = sanitize_text(long_text, max_length=100)

# Sanitize for HTML rendering (escapes special chars)
safe_html = sanitize_html("<script>alert('xss')</script>")
# Result: "" (script removed, nothing left to escape)

# Sanitize entire output dictionary
from trader_koo.llm import sanitize_llm_output

sanitized = sanitize_llm_output(
    llm_output,
    field_limits={
        "observation": 5000,
        "action": 5000,
        "risk_note": 1000
    }
)
```

### Complete Workflow

```python
from trader_koo.llm import (
    validate_llm_output,
    sanitize_llm_output,
    NarrativeGeneration,
)
from trader_koo.llm.validator import generate_fallback_narrative

def process_llm_response(raw_output: dict, context: dict) -> dict:
    """Process LLM response with validation and sanitization."""

    # Step 1: Sanitize raw output
    sanitized = sanitize_llm_output(
        raw_output,
        field_limits={
            "observation": 5000,
            "action": 5000,
            "risk_note": 1000
        }
    )

    # Step 2: Validate against schema
    result = validate_llm_output(
        sanitized,
        NarrativeGeneration,
        context=context
    )

    # Step 3: Use validated data or fallback
    if result.is_valid:
        return result.data.model_dump()
    else:
        # Log validation failure
        logger.warning(
            "LLM validation failed, using fallback",
            extra={"errors": result.errors, "context": context}
        )
        return generate_fallback_narrative(context)
```

## Testing

### Unit Tests

Run unit tests for sanitization:

```bash
pytest tests/test_llm_sanitizer.py -v
```

### Property-Based Tests

Run property-based tests for validation (uses Hypothesis):

```bash
pytest tests/test_llm_validator_property.py -v
```

Property tests verify universal properties across all inputs:

- Valid inputs pass validation
- Invalid inputs fail with appropriate errors
- Length violations are detected
- Missing required fields are detected
- Fallback generators always produce valid output

## Security Features

1. **Schema Validation**: All LLM output validated against strict schemas
2. **Length Enforcement**: Maximum lengths enforced with truncation
3. **HTML Stripping**: All HTML tags removed before storage
4. **Script Removal**: Script tags and inline handlers removed
5. **XSS Prevention**: Special characters escaped for HTML rendering
6. **Fallback Logic**: Deterministic fallback when validation fails
7. **Comprehensive Logging**: All validation failures logged with context

## Requirements Mapping

- **Requirement 2.1**: JSON schema validation (schemas.py, validator.py)
- **Requirement 2.2**: Length enforcement (schemas.py, sanitizer.py)
- **Requirement 2.3**: Truncation with ellipsis (sanitizer.py)
- **Requirement 2.4**: HTML/script stripping (sanitizer.py)
- **Requirement 2.5**: Special character escaping (sanitizer.py)
- **Requirement 2.6**: Fallback to deterministic output (validator.py, fallback.py)
- **Requirement 2.7**: Validation failure logging (validator.py)
- **Requirements 7.1-7.7**: Schema enforcement and validation (schemas.py, validator.py)

## Integration with Existing Code

The existing `llm_narrative.py` module can be updated to use these new validation and sanitization functions:

```python
# In llm_narrative.py
from trader_koo.llm import validate_llm_output, SetupRewrite
from trader_koo.llm.validator import generate_fallback_narrative

def maybe_rewrite_setup_copy(row: dict[str, Any], *, source: str) -> dict[str, str]:
    # ... existing code ...

    try:
        rewritten, usage_meta = _azure_chat_rewrite(context)

        # Validate LLM output
        result = validate_llm_output(
            rewritten,
            SetupRewrite,
            context={"ticker": context.get("ticker"), "source": source}
        )

        if not result.is_valid:
            # Log validation failure and use fallback
            logger.warning(
                "LLM output validation failed",
                extra={"errors": result.errors, "context": context}
            )
            return generate_fallback_narrative(context)

        # Use validated data
        return result.data.model_dump()

    except Exception as exc:
        # ... existing error handling ...
```
