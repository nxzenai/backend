"""
NxZen AI Studio

SQL Service

Business layer for the SQL Lab.

Responsibilities
----------------
• Validate SQL queries
• Execute SQL queries
• Load database schema
• Return execution statistics
• Manage user databases
"""

from __future__ import annotations

import time
import logging
import sqlite3
from app.core.exceptions.custom import AIStudioException
from app.modules.sql.lifecycle import parse_database_command, execute_database_command, database_state
from app.modules.sql.history import record_execution, record_reset

from app.modules.auth.models import UserModel

from app.modules.sql.repository import SQLRepository

from app.modules.sql.validator import (
    SQLValidator,
    SQLValidationError,
)


class SQLService:

    def __init__(
        self,
        repository: SQLRepository,
    ):

        self.repository = repository

    ##########################################################
    # Execute SQL Query
    ##########################################################

    def execute(
        self,
        current_user: UserModel,
        query: str,
    ) -> dict:

        ######################################################
        # Validate Query
        ######################################################

        query = query.strip()

        if not query:

            raise AIStudioException(
                "Query cannot be empty."
            )

        ######################################################
        # SQL Validation
        ######################################################

        command = parse_database_command(query)
        if not command:
            try:
                SQLValidator.validate(query)
            except SQLValidationError as exc:
                raise AIStudioException(str(exc), error_code="INVALID_SQL") from exc

        ######################################################
        # Execute Query
        ######################################################

        start = time.perf_counter()

        try:
            result = (execute_database_command(current_user, *command) if command else
                      self.repository.execute(current_user=current_user, query=query))
        except (ValueError, sqlite3.Error, OSError) as exc:
            self._record_execution(current_user.id, query, time.perf_counter() - start, False)
            message = str(exc) if isinstance(exc, ValueError) else "Database storage is unavailable or busy. Retry shortly; contact an administrator if it persists."
            raise AIStudioException(message, error_code="SQL_EXECUTION_FAILED") from exc
        except Exception:
            self._record_execution(current_user.id, query, time.perf_counter() - start, False)
            raise

        execution_time = round(

            time.perf_counter() - start,

            4,

        )

        result["execution_time"] = execution_time
        self._record_execution(current_user.id, query, execution_time, True)

        return result

    @staticmethod
    def _record_execution(*args):
        try:
            record_execution(*args)
        except Exception:
            # History outages must not mask a committed lifecycle result or its error.
            logging.getLogger(__name__).warning("SQL execution history could not be recorded")

    ##########################################################
    # Database Schema
    ##########################################################

    def schema(
        self,
        current_user: UserModel,
    ) -> dict:

        tables = self.repository.schema(

            current_user=current_user,

        )

        return {

            "tables": tables,
            **database_state(current_user),

        }

    ##########################################################
    # Database Statistics
    ##########################################################

    def statistics(
        self,
        current_user: UserModel,
    ) -> dict:

        return self.repository.statistics(

            current_user=current_user,

        )

    ##########################################################
    # List Tables
    ##########################################################

    def list_tables(
        self,
        current_user: UserModel,
    ) -> list[str]:

        return self.repository.list_tables(

            current_user=current_user,

        )

    ##########################################################
    # Check Table Exists
    ##########################################################

    def table_exists(
        self,
        current_user: UserModel,
        table_name: str,
    ) -> bool:

        return self.repository.table_exists(

            current_user=current_user,

            table_name=table_name,

        )

    ##########################################################
    # Reset Database
    ##########################################################

    def reset_database(
        self,
        current_user: UserModel,
    ) -> None:

        self.repository.reset_database(

            current_user=current_user,

        )
        record_reset(current_user.id)

    ##########################################################
    # Delete Database
    ##########################################################

    def delete_database(
        self,
        current_user: UserModel,
    ) -> None:

        self.repository.delete_database(

            current_user=current_user,

        )
