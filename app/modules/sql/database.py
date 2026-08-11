from pathlib import Path
import sqlite3

from app.modules.sql.constants import (
    SQL_STORAGE_FOLDER,
)

##########################################################
# Ensure storage exists
##########################################################

SQL_STORAGE_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

##########################################################
# Get User Database
##########################################################


def database_path(
    user_id: str,
) -> Path:

    return (
        SQL_STORAGE_FOLDER
        / f"{user_id}.db"
    )


##########################################################
# SQLite Connection
##########################################################


def get_connection(
    user_id: str,
) -> sqlite3.Connection:

    path = database_path(
        user_id,
    )

    connection = sqlite3.connect(

        path,

        check_same_thread=False,

    )

    connection.row_factory = sqlite3.Row

    return connection