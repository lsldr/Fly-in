"""Errors raised while reading and validating map configurations."""


class ConfigParseError(ValueError):
    """An invalid or unreadable map with line context when available."""
