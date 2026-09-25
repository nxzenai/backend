"""
NxZen AI Studio

SQL Repository

Responsibilities

• Manage per-user SQLite databases
• Auto-create databases
• Seed demo data
• Execute SQL queries
• Return schema information
"""

from __future__ import annotations

import sqlite3

from app.modules.auth.models import UserModel
from app.modules.sql.lifecycle import owned_database, owned_path, execute_database_command, fail, registry

from app.modules.sql.database import (
    get_connection,
)


class SQLRepository:

    def __init__(self):

        pass

    ##########################################################
    # Database
    ##########################################################

    def _connection(
        self,
        current_user: UserModel,
    ) -> sqlite3.Connection:

        owned = owned_database(current_user)
        if owned:
            connection = sqlite3.connect(owned_path(owned).resolve().as_uri() + "?mode=rw", uri=True,
                                         check_same_thread=False)
            connection.row_factory = sqlite3.Row
            return connection

        connection = get_connection(
            str(current_user.id),
        )

        self._initialize_database(
            connection,
        )

        return connection

    ##########################################################
    # Initialize Database
    ##########################################################

    def _initialize_database(
        self,
        connection: sqlite3.Connection,
    ) -> None:

        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS employees(

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT NOT NULL,

                department TEXT,

                salary REAL

            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS customers(

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT,

                city TEXT

            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS products(

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT,

                price REAL

            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS orders(

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                customer_id INTEGER,

                product_id INTEGER,

                quantity INTEGER

            )
            """
        )

        connection.commit()

        self._seed_database(
            connection,
        )

    ##########################################################
    # Seed Database
    ##########################################################

    def _seed_database(
        self,
        connection: sqlite3.Connection,
    ) -> None:

        cursor = connection.cursor()

        cursor.execute(

            """
            SELECT COUNT(*)

            FROM employees
            """

        )

        count = cursor.fetchone()[0]

        if count > 0:

            return

        ######################################################
        # Employees
        ######################################################

        cursor.executemany(

            """
            INSERT INTO employees
            (
                name,
                department,
                salary
            )
            VALUES
            (
                ?,
                ?,
                ?
            )
            """,

            [

                ("John", "AI", 75000),

                ("Alice", "Data Science", 90000),

                ("Bob", "Engineering", 65000),

                ("David", "Sales", 55000),

            ],

        )

        ######################################################
        # Customers
        ######################################################

        cursor.executemany(

            """
            INSERT INTO customers
            (
                name,
                city
            )
            VALUES
            (
                ?,
                ?
            )
            """,

            [

                ("Rahul", "Hyderabad"),

                ("Sneha", "Bangalore"),

                ("Amit", "Chennai"),

            ],

        )

        ######################################################
        # Products
        ######################################################

        cursor.executemany(

            """
            INSERT INTO products
            (
                name,
                price
            )
            VALUES
            (
                ?,
                ?
            )
            """,

            [

                ("Laptop", 65000),

                ("Keyboard", 1800),

                ("Mouse", 900),

            ],

        )
                ######################################################
        # Orders
        ######################################################

        cursor.executemany(

            """
            INSERT INTO orders
            (
                customer_id,
                product_id,
                quantity
            )
            VALUES
            (
                ?,
                ?,
                ?
            )
            """,

            [

                (1, 1, 2),

                (2, 2, 1),

                (3, 3, 5),

                (1, 2, 3),

            ],

        )

        connection.commit()

    ##########################################################
    # Execute SQL
    ##########################################################

    def execute(
        self,
        current_user: UserModel,
        query: str,
    ) -> dict:

        connection = self._connection(
            current_user,
        )

        cursor = connection.cursor()

        sql = query.strip()

        if not sql:

            raise ValueError(
                "Query cannot be empty."
            )

        first_keyword = (
            sql.split()[0]
            .upper()
        )

        ######################################################
        # Block Dangerous Commands
        ######################################################

        blocked = {

            "ATTACH",

            "DETACH",

            "VACUUM",

            "REINDEX",

            "ANALYZE",

            "PRAGMA",

        }

        if first_keyword in blocked:

            raise ValueError(

                f"{first_keyword} is not allowed."

            )

        ######################################################
        # Execute Query
        ######################################################

        try:

            # Enforce file isolation even when SQL uses comments or alternate spacing.
            connection.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY
                                      if action in {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
                                                    sqlite3.SQLITE_PRAGMA}
                                      else sqlite3.SQLITE_OK)
            cursor.execute(sql)

            ##################################################
            # SELECT
            ##################################################

            if cursor.description:

                columns = [

                    column[0]

                    for column

                    in cursor.description

                ]

                rows = [

                    list(row)

                    for row

                    in cursor.fetchall()

                ]

                return {

                    "columns": columns,

                    "rows": rows,

                }

            ##################################################
            # CREATE / INSERT / UPDATE / DELETE
            ##################################################

            connection.commit()

            return {

                "columns": [],

                "rows": [],

            }

        except sqlite3.Error as exc:

            connection.rollback()

            raise ValueError(

                "Database storage is unavailable or busy. Retry shortly."
                if getattr(exc, "sqlite_errorcode", 0) & 255 in {
                    sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_CANTOPEN,
                    sqlite3.SQLITE_IOERR, sqlite3.SQLITE_FULL, sqlite3.SQLITE_READONLY,
                    sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB,
                } else str(exc)

            ) from exc

        finally:

            connection.close()

    ##########################################################
    # Database Schema
    ##########################################################

    def schema(
        self,
        current_user: UserModel,
    ) -> list[dict]:

        connection = self._connection(
            current_user,
        )

        cursor = connection.cursor()

        cursor.execute(

            """
            SELECT name

            FROM sqlite_master

            WHERE type='table'

            ORDER BY name
            """

        )

        tables = []

        for row in cursor.fetchall():

            table_name = row[0]

            ##################################################
            # Hide SQLite Internal Tables
            ##################################################

            if table_name.startswith(

                "sqlite_"

            ):

                continue

            cursor.execute(

                'PRAGMA table_info("' + table_name.replace('"', '""') + '")'

            )

            columns = [

                column[1]

                for column

                in cursor.fetchall()

            ]

            tables.append(

                {

                    "name": table_name,

                    "columns": columns,

                }

            )

        connection.close()

        return tables
        ##########################################################
    # Table Exists
    ##########################################################

    def table_exists(
        self,
        current_user: UserModel,
        table_name: str,
    ) -> bool:

        connection = self._connection(
            current_user,
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT name

            FROM sqlite_master

            WHERE type='table'
            AND name=?
            """,
            (
                table_name,
            ),
        )

        exists = cursor.fetchone() is not None

        connection.close()

        return exists

    ##########################################################
    # List Tables
    ##########################################################

    def list_tables(
        self,
        current_user: UserModel,
    ) -> list[str]:

        connection = self._connection(
            current_user,
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT name

            FROM sqlite_master

            WHERE type='table'

            ORDER BY name
            """
        )

        tables = [

            row[0]

            for row in cursor.fetchall()

            if not row[0].startswith(
                "sqlite_"
            )

        ]

        connection.close()

        return tables

    ##########################################################
    # Reset User Database
    ##########################################################

    def reset_database(
        self,
        current_user: UserModel,
    ) -> None:

        with registry() as catalog:
            owned = catalog.execute("SELECT * FROM databases WHERE owner_user_id = ?",
                                    (str(current_user.id),)).fetchone()
            if not owned:
                fail("The default or untracked legacy database cannot be deleted or reset.", 403)
            db_path = owned_path(owned)
            db_path.unlink(missing_ok=True)
            connection = sqlite3.connect(db_path)
            try:
                self._initialize_database(connection)
            finally:
                connection.close()

    ##########################################################
    # Delete User Database
    ##########################################################

    def delete_database(
        self,
        current_user: UserModel,
    ) -> None:

        owned = owned_database(current_user)
        if not owned:
            fail("The default or untracked legacy database cannot be deleted.", 403)
        execute_database_command(current_user, "DROP", owned["name"])

    ##########################################################
    # Database Statistics
    ##########################################################

    def statistics(
        self,
        current_user: UserModel,
    ) -> dict:

        connection = self._connection(
            current_user,
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT name

            FROM sqlite_master

            WHERE type='table'
            """
        )

        tables = [

            row[0]

            for row in cursor.fetchall()

            if not row[0].startswith(
                "sqlite_"
            )

        ]

        total_rows = 0

        for table in tables:

            cursor.execute(
                f"SELECT COUNT(*) FROM {table}"
            )

            total_rows += cursor.fetchone()[0]

        connection.close()

        return {

            "database": (owned_database(current_user) or {}).get("name", f"{current_user.id}.db"),

            "tables": len(tables),

            "rows": total_rows,

        }
