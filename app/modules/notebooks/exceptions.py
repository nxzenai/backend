from app.core.exceptions.custom import AIStudioException


class NotebookNotFound(AIStudioException):

    def __init__(self):
        super().__init__(
            message="Notebook not found.",
            status_code=404,
            error_code="NOTEBOOK_NOT_FOUND",
        )


class NotebookPermissionDenied(AIStudioException):

    def __init__(self):
        super().__init__(
            message="You don't have permission to access this notebook.",
            status_code=403,
            error_code="NOTEBOOK_PERMISSION_DENIED",
        )


class NotebookTitleRequired(AIStudioException):

    def __init__(self):
        super().__init__(
            message="Notebook title cannot be empty.",
            status_code=400,
            error_code="NOTEBOOK_TITLE_REQUIRED",
        )

class CellNotFound(AIStudioException):

    def __init__(self):

        super().__init__(
            message="Cell not found.",
            status_code=404,
            error_code="CELL_NOT_FOUND",
        )


class InvalidCellType(AIStudioException):

    def __init__(self):

        super().__init__(
            message="Invalid cell type.",
            status_code=400,
            error_code="INVALID_CELL_TYPE",
        )
class InvalidNotebookId(AIStudioException):

    def __init__(self):

        super().__init__(
            message="Invalid notebook id.",
            status_code=400,
            error_code="INVALID_NOTEBOOK_ID",
        )