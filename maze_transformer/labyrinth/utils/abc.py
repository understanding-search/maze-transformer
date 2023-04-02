"""Utilities relating to abstract base classes."""

from functools import wraps


def override(func):
    """Decorator used to mark that a method overrides (or implements) a method from a superclass."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper