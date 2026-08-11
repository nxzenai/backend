from app.modules.sql.repository import (
    SQLRepository,
)

from app.modules.sql.service import (
    SQLService,
)


def get_sql_service():

    repository = SQLRepository()

    return SQLService(
        repository,
    )