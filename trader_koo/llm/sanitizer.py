"""
LLM Output Sanitizer

Sanitizes LLM-generated content to prevent XSS and ensure safe rendering.
Implements Requirements 2.2, 2.3, 2.4, 2.5.
"""

from __future__ import annotations

import html
import re
from typing import Any


def truncate_with_ellipsis(text: str, max_length: int) -> str:
    """
    Truncate text to maximum length with ellipsis.
    
    Implements Requirements 2.2, 2.3.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (including ellipsis)
    
    Returns:
        Truncated text with ellipsis if needed
    
    Example:
        >>> truncate_with_ellipsis("This is a long text", 10)
        'This is...'
    """
    if not text:
        return ""
    
    text = str(text).strip()
    
    if len(text) <= max_length:
        return text
    
    # Reserve 3 characters for ellipsis
    if max_length <= 3:
        return "..."[:max_length]
    
    truncated = text[:max_length - 3].strip()
    return f"{truncated}..."


def strip_html_tags(text: str) -> str:
    """
    Strip all HTML tags from text.
    
    Implements Requirements 2.4.
    
    Args:
        text: Text potentially containing HTML tags
    
    Returns:
        Text with all HTML tags removed
    
    Example:
        >>> strip_html_tags("<p>Hello <b>world</b></p>")
        'Hello world'
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", str(text))
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def strip_script_content(text: str) -> str:
    """
    Remove script tags and their content.
    
    Implements Requirements 2.4.
    
    Args:
        text: Text potentially containing script tags
    
    Returns:
        Text with script tags and content removed
    
    Example:
        >>> strip_script_content("Hello <script>alert('xss')</script> world")
        'Hello  world'
    """
    if not text:
        return ""
    
    # Remove script tags and content (case-insensitive)
    text = re.sub(r"<script[^>]*>.*?</script>", "", str(text), flags=re.IGNORECASE | re.DOTALL)
    
    # Remove inline event handlers
    text = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+on\w+\s*=\s*\S+", "", text, flags=re.IGNORECASE)
    
    return text


def escape_special_characters(text: str) -> str:
    """
    Escape special HTML characters for safe rendering.
    
    Implements Requirements 2.5.
    
    Args:
        text: Text to escape
    
    Returns:
        Text with special characters escaped
    
    Example:
        >>> escape_special_characters("<script>alert('xss')</script>")
        '&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;'
    """
    if not text:
        return ""
    
    return html.escape(str(text), quote=True)


def sanitize_text(text: str, max_length: int | None = None) -> str:
    """
    Sanitize text by stripping HTML/scripts and optionally truncating.
    
    Implements Requirements 2.2, 2.3, 2.4.
    
    Args:
        text: Text to sanitize
        max_length: Optional maximum length (will truncate with ellipsis)
    
    Returns:
        Sanitized text
    
    Example:
        >>> sanitize_text("<p>Hello <script>alert('xss')</script></p>", max_length=10)
        'Hello'
    """
    if not text:
        return ""
    
    # Strip script content first
    text = strip_script_content(text)
    
    # Strip HTML tags
    text = strip_html_tags(text)
    
    # Truncate if needed
    if max_length is not None and max_length > 0:
        text = truncate_with_ellipsis(text, max_length)
    
    return text


def sanitize_html(text: str, max_length: int | None = None) -> str:
    """
    Sanitize text for HTML rendering by escaping special characters.
    
    Implements Requirements 2.4, 2.5.
    
    Use this when you need to render text in HTML context.
    
    Args:
        text: Text to sanitize for HTML
        max_length: Optional maximum length (will truncate with ellipsis)
    
    Returns:
        HTML-safe text with escaped special characters
    
    Example:
        >>> sanitize_html("<script>alert('xss')</script>")
        '&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;'
    """
    if not text:
        return ""
    
    # First sanitize (strip tags/scripts)
    text = sanitize_text(text, max_length=max_length)
    
    # Then escape for HTML rendering
    text = escape_special_characters(text)
    
    return text


def sanitize_llm_output(output: dict[str, Any], field_limits: dict[str, int] | None = None) -> dict[str, Any]:
    """
    Sanitize all text fields in LLM output dictionary.
    
    Implements Requirements 2.2, 2.3, 2.4, 2.5.
    
    Args:
        output: LLM output dictionary
        field_limits: Optional dictionary mapping field names to max lengths
    
    Returns:
        Sanitized output dictionary
    
    Example:
        >>> sanitize_llm_output(
        ...     {"observation": "<p>Market is bullish</p>"},
        ...     field_limits={"observation": 100}
        ... )
        {'observation': 'Market is bullish'}
    """
    if not isinstance(output, dict):
        return {}
    
    field_limits = field_limits or {}
    sanitized = {}
    
    for key, value in output.items():
        if isinstance(value, str):
            max_length = field_limits.get(key)
            sanitized[key] = sanitize_text(value, max_length=max_length)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_llm_output(value, field_limits=field_limits)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_text(item, max_length=field_limits.get(key))
                if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized
