"""
Database utility functions.

This module provides reusable helper functions for working with MongoDB.
"""

from bson import ObjectId
from bson.errors import InvalidId


def validate_object_id(value: str) -> ObjectId:
    """
    Validate and convert a string into a MongoDB ObjectId.

    Args:
        value: MongoDB ObjectId as a string.

    Returns:
        ObjectId: Valid MongoDB ObjectId instance.

    Raises:
        InvalidId: If the supplied value is not a valid ObjectId.

    Example:
        >>> object_id = validate_object_id("68863f6a9d8d9e5d6d1f4d10")
    """

    if not isinstance(value, str):
        raise InvalidId("ObjectId must be a string.")

    value = value.strip()

    if not value:
        raise InvalidId("ObjectId cannot be empty.")

    if not ObjectId.is_valid(value):
        raise InvalidId(f"'{value}' is not a valid MongoDB ObjectId.")

    return ObjectId(value)