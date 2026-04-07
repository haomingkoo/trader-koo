"""
Unit tests for LLM output sanitization.

Tests HTML sanitization, length truncation, and special character escaping.
Validates Requirements 2.2, 2.3, 2.4, 2.5.
"""

import pytest

from trader_koo.llm.sanitizer import (
    escape_special_characters,
    sanitize_html,
    sanitize_llm_output,
    sanitize_text,
    strip_html_tags,
    strip_script_content,
    truncate_with_ellipsis,
)


class TestTruncateWithEllipsis:
    """Test truncation with ellipsis functionality."""

    def test_no_truncation_needed(self):
        """Text shorter than max_length should not be truncated."""
        text = "Short text"
        result = truncate_with_ellipsis(text, 20)
        assert result == "Short text"

    def test_exact_length(self):
        """Text exactly at max_length should not be truncated."""
        text = "Exactly ten"
        result = truncate_with_ellipsis(text, 11)
        assert result == "Exactly ten"

    def test_truncation_with_ellipsis(self):
        """Text longer than max_length should be truncated with ellipsis."""
        text = "This is a very long text that needs truncation"
        result = truncate_with_ellipsis(text, 20)
        assert len(result) == 20
        assert result.endswith("...")
        # Exact content may vary based on word boundaries, just check structure
        assert result.startswith("This is")

    def test_very_short_max_length(self):
        """Max length of 3 or less should return just ellipsis."""
        text = "Long text"
        assert truncate_with_ellipsis(text, 3) == "..."
        assert truncate_with_ellipsis(text, 2) == ".."
        assert truncate_with_ellipsis(text, 1) == "."

    def test_empty_text(self):
        """Empty text should return empty string."""
        assert truncate_with_ellipsis("", 10) == ""
        assert truncate_with_ellipsis(None, 10) == ""

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be stripped."""
        text = "  Text with spaces  "
        result = truncate_with_ellipsis(text, 100)
        assert result == "Text with spaces"


class TestStripHtmlTags:
    """Test HTML tag stripping functionality."""

    def test_simple_tags(self):
        """Simple HTML tags should be removed."""
        text = "<p>Hello world</p>"
        result = strip_html_tags(text)
        assert result == "Hello world"

    def test_nested_tags(self):
        """Nested HTML tags should be removed."""
        text = "<div><p>Hello <b>world</b></p></div>"
        result = strip_html_tags(text)
        assert result == "Hello world"

    def test_self_closing_tags(self):
        """Self-closing tags should be removed."""
        text = "Line 1<br/>Line 2<hr/>Line 3"
        result = strip_html_tags(text)
        assert result == "Line 1Line 2Line 3"

    def test_tags_with_attributes(self):
        """Tags with attributes should be removed."""
        text = '<a href="http://example.com">Link</a>'
        result = strip_html_tags(text)
        assert result == "Link"

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces should be collapsed to single space."""
        text = "Text   with    multiple     spaces"
        result = strip_html_tags(text)
        assert result == "Text with multiple spaces"

    def test_empty_text(self):
        """Empty text should return empty string."""
        assert strip_html_tags("") == ""
        assert strip_html_tags(None) == ""


class TestStripScriptContent:
    """Test script content removal functionality."""

    def test_script_tag_removal(self):
        """Script tags and content should be removed."""
        text = "Hello <script>alert('xss')</script> world"
        result = strip_script_content(text)
        assert result == "Hello  world"

    def test_script_with_attributes(self):
        """Script tags with attributes should be removed."""
        text = '<script type="text/javascript">alert("xss")</script>'
        result = strip_script_content(text)
        assert result == ""

    def test_multiple_scripts(self):
        """Multiple script tags should be removed."""
        text = "<script>alert(1)</script>Text<script>alert(2)</script>"
        result = strip_script_content(text)
        assert result == "Text"

    def test_case_insensitive(self):
        """Script tag removal should be case-insensitive."""
        text = "Hello <SCRIPT>alert('xss')</SCRIPT> world"
        result = strip_script_content(text)
        assert result == "Hello  world"

    def test_inline_event_handlers(self):
        """Inline event handlers should be removed."""
        text = '<div onclick="alert(\'xss\')">Click me</div>'
        result = strip_script_content(text)
        assert 'onclick' not in result

    def test_empty_text(self):
        """Empty text should return empty string."""
        assert strip_script_content("") == ""
        assert strip_script_content(None) == ""


class TestEscapeSpecialCharacters:
    """Test special character escaping functionality."""

    def test_basic_escaping(self):
        """Basic HTML special characters should be escaped."""
        text = "<script>alert('xss')</script>"
        result = escape_special_characters(text)
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&#x27;" in result or "&apos;" in result

    def test_ampersand_escaping(self):
        """Ampersands should be escaped."""
        text = "Tom & Jerry"
        result = escape_special_characters(text)
        assert result == "Tom &amp; Jerry"

    def test_quote_escaping(self):
        """Quotes should be escaped."""
        text = 'He said "Hello"'
        result = escape_special_characters(text)
        assert "&quot;" in result or "&#x22;" in result

    def test_empty_text(self):
        """Empty text should return empty string."""
        assert escape_special_characters("") == ""
        assert escape_special_characters(None) == ""


class TestSanitizeText:
    """Test text sanitization functionality."""

    def test_full_sanitization(self):
        """Text should be sanitized by removing HTML and scripts."""
        text = "<p>Hello <script>alert('xss')</script> world</p>"
        result = sanitize_text(text)
        # Multiple spaces are collapsed to single space
        assert result == "Hello world"
        assert "<" not in result
        assert "script" not in result

    def test_sanitization_with_truncation(self):
        """Text should be sanitized and truncated."""
        text = "<p>This is a very long text that needs truncation</p>"
        result = sanitize_text(text, max_length=20)
        assert len(result) <= 20
        assert "<" not in result
        assert result.endswith("...")

    def test_plain_text_unchanged(self):
        """Plain text without HTML should remain unchanged."""
        text = "Plain text without HTML"
        result = sanitize_text(text)
        assert result == text

    def test_empty_text(self):
        """Empty text should return empty string."""
        assert sanitize_text("") == ""
        assert sanitize_text(None) == ""


class TestSanitizeHtml:
    """Test HTML sanitization for rendering."""

    def test_html_escaping(self):
        """Text should be sanitized and escaped for HTML rendering."""
        text = "<script>alert('xss')</script>"
        result = sanitize_html(text)
        # Script tags are stripped first, then remaining text is escaped
        # Since script content is removed, result may be empty or contain escaped remnants
        assert "<script>" not in result
        assert "alert" not in result or "&" in result

    def test_html_with_truncation(self):
        """Text should be sanitized, truncated, and escaped."""
        text = "<p>This is a very long text</p>"
        result = sanitize_html(text, max_length=15)
        assert len(result) <= 20  # Escaped characters may be longer
        assert "&lt;" not in result or result.startswith("This")

    def test_empty_text(self):
        """Empty text should return empty string."""
        assert sanitize_html("") == ""
        assert sanitize_html(None) == ""


class TestSanitizeLlmOutput:
    """Test LLM output dictionary sanitization."""

    def test_sanitize_string_fields(self):
        """String fields should be sanitized."""
        output = {
            "observation": "<p>Market is bullish</p>",
            "action": "<script>alert('xss')</script>Buy now",
        }
        result = sanitize_llm_output(output)
        assert result["observation"] == "Market is bullish"
        assert result["action"] == "Buy now"
        assert "<" not in result["observation"]
        assert "script" not in result["action"]

    def test_sanitize_with_field_limits(self):
        """String fields should be sanitized and truncated per field limits."""
        output = {
            "observation": "This is a very long observation that needs truncation",
            "action": "Short action",
        }
        field_limits = {
            "observation": 20,
            "action": 100,
        }
        result = sanitize_llm_output(output, field_limits=field_limits)
        assert len(result["observation"]) <= 20
        assert result["observation"].endswith("...")
        assert result["action"] == "Short action"

    def test_nested_dictionaries(self):
        """Nested dictionaries should be sanitized recursively."""
        output = {
            "data": {
                "text": "<p>Nested text</p>",
            }
        }
        result = sanitize_llm_output(output)
        assert result["data"]["text"] == "Nested text"

    def test_list_fields(self):
        """List fields with strings should be sanitized."""
        output = {
            "items": [
                "<p>Item 1</p>",
                "<p>Item 2</p>",
            ]
        }
        result = sanitize_llm_output(output)
        assert result["items"][0] == "Item 1"
        assert result["items"][1] == "Item 2"

    def test_non_string_fields_unchanged(self):
        """Non-string fields should remain unchanged."""
        output = {
            "count": 42,
            "ratio": 3.14,
            "enabled": True,
        }
        result = sanitize_llm_output(output)
        assert result["count"] == 42
        assert result["ratio"] == 3.14
        assert result["enabled"] is True

    def test_empty_output(self):
        """Empty output should return empty dict."""
        assert sanitize_llm_output({}) == {}
        assert sanitize_llm_output(None) == {}


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_characters(self):
        """Unicode characters should be preserved."""
        text = "Hello 世界 🌍"
        result = sanitize_text(text)
        assert "世界" in result
        assert "🌍" in result

    def test_malformed_html(self):
        """Malformed HTML should be handled gracefully."""
        text = "<p>Unclosed tag"
        result = sanitize_text(text)
        assert "<" not in result
        assert result == "Unclosed tag"

    def test_deeply_nested_html(self):
        """Deeply nested HTML should be stripped."""
        text = "<div><div><div><p>Deep</p></div></div></div>"
        result = sanitize_text(text)
        assert result == "Deep"

    def test_mixed_content(self):
        """Mixed HTML and plain text should be handled."""
        text = "Plain <b>bold</b> more plain <i>italic</i> end"
        result = sanitize_text(text)
        assert result == "Plain bold more plain italic end"
