"""
NxZen AI Studio

SQL Validator

Enterprise SQL validation layer.
"""

from __future__ import annotations

import re


class SQLValidationError(Exception):
    pass


class SQLValidator:

    ##########################################################
    # Allowed First Commands
    ##########################################################

    ALLOWED = {

        "SELECT",

        "INSERT",

        "UPDATE",

        "DELETE",

        "CREATE",

        "ALTER",

        "WITH",

    }

    ##########################################################
    # Dangerous Commands
    ##########################################################

    BLOCKED = {

        "ATTACH",

        "DETACH",

        "VACUUM",

        "REINDEX",

        "PRAGMA",

        "ANALYZE",

        "EXPLAIN",

    }

    ##########################################################
    # Validation
    ##########################################################

    @classmethod
    def validate(
        cls,
        query: str,
    ) -> None:

        query = query.strip()

        if not query:

            raise SQLValidationError(
                "Query cannot be empty."
            )

        ######################################################
        # Single statement only
        ######################################################

        cleaned = query.rstrip(";").strip()

        if ";" in cleaned:

            raise SQLValidationError(

                "Multiple SQL statements are not allowed."

            )

        upper = cleaned.upper()

        first = upper.split()[0]

        ######################################################
        # First keyword
        ######################################################

        if first in cls.BLOCKED:

            raise SQLValidationError(

                f"{first} statements are not allowed."

            )

        if first not in cls.ALLOWED:

            raise SQLValidationError(

                f"{first} is not supported."

            )

        ######################################################
        # CREATE validation
        ######################################################

        if first == "CREATE":

            allowed = (

                "CREATE TABLE",

                "CREATE TEMP TABLE",

            )

            if not upper.startswith(allowed):

                raise SQLValidationError(

                    "Only CREATE TABLE is allowed."

                )

        ######################################################
        # ALTER validation
        ######################################################

        if first == "ALTER":

            if not upper.startswith(

                "ALTER TABLE"

            ):

                raise SQLValidationError(

                    "Only ALTER TABLE is allowed."

                )

        ######################################################
        # Dangerous SQL Patterns
        ######################################################

        patterns = [

            r"\bATTACH\b",

            r"\bDETACH\b",

            r"\bVACUUM\b",

            r"\bREINDEX\b",

            r"\bANALYZE\b",

            r"\bPRAGMA\b",

            r"\bLOAD_EXTENSION\b",

            r"\bCREATE\s+VIEW\b",

            r"\bCREATE\s+TRIGGER\b",

            r"\bCREATE\s+INDEX\b",

        ]

        for pattern in patterns:

            if re.search(pattern, upper):

                raise SQLValidationError(

                    "Dangerous SQL detected."

                )