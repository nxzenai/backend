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


class InvalidCellOrder(AIStudioException):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=400,
            error_code="INVALID_CELL_ORDER",
        )


class NotebookFileNotFound(AIStudioException):
    def __init__(self):
        super().__init__("Notebook file not found.", 404, "NOTEBOOK_FILE_NOT_FOUND")


class InvalidNotebookFile(AIStudioException):
    def __init__(self, message: str = "Invalid notebook file."):
        super().__init__(message, 422, "INVALID_NOTEBOOK_FILE")


class NotebookFileTooLarge(AIStudioException):
    def __init__(self):
        super().__init__(
            "Notebook file exceeds the configured size limit.",
            413,
            "NOTEBOOK_FILE_TOO_LARGE",
        )


class InvalidNotebookImport(AIStudioException):
    def __init__(self, message: str = "Invalid Jupyter notebook."):
        super().__init__(message, 422, "INVALID_NOTEBOOK_IMPORT")


class NotebookImportTooLarge(AIStudioException):
    def __init__(self):
        super().__init__(
            "Jupyter notebook exceeds the configured size limit.",
            413,
            "NOTEBOOK_IMPORT_TOO_LARGE",
        )


class NotebookExampleNotFound(AIStudioException):
    def __init__(self):
        super().__init__(
            "Notebook example not found.", 404, "NOTEBOOK_EXAMPLE_NOT_FOUND"
        )
