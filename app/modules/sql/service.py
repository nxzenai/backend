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

from app.modules.auth.models import UserModel

from app.modules.sql.repository import SQLRepository

from app.modules.sql.validator import (
    SQLValidator,
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

            raise ValueError(
                "Query cannot be empty."
            )

        ######################################################
        # SQL Validation
        ######################################################

        SQLValidator.validate(
            query,
        )

        ######################################################
        # Execute Query
        ######################################################

        start = time.perf_counter()

        result = self.repository.execute(

            current_user=current_user,

            query=query,

        )

        execution_time = round(

            time.perf_counter() - start,

            4,

        )

        result["execution_time"] = execution_time

        return result

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