NOTEBOOK_COLLECTION = "notebooks"

MAX_NOTEBOOK_TITLE = 150

MAX_NOTEBOOK_DESCRIPTION = 2000

DEFAULT_VISIBILITY = "private"

MAX_CELL_SOURCE_LENGTH = 100000

SUPPORTED_CELL_TYPES = (
    "markdown",
    "code",
)

SUPPORTED_NOTEBOOK_FILE_EXTENSIONS = {".csv", ".json", ".txt", ".xlsx"}
SUPPORTED_NOTEBOOK_MIME_TYPES = {
    ".csv": {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    },
    ".json": {
        "application/json",
        "text/json",
        "text/plain",
        "application/octet-stream",
    },
    ".txt": {"text/plain", "application/octet-stream"},
    ".xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
    },
}
