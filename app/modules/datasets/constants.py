from pathlib import Path

###########################################################
# Upload Configuration
###########################################################

UPLOAD_DIRECTORY = (
    Path("uploads") / "datasets"
)

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

###########################################################
# Supported File Types
###########################################################

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx",
}

SUPPORTED_CONTENT_TYPES = {
    "text/csv",

    "application/vnd.ms-excel",

    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

###########################################################
# Upload Limits
###########################################################

MAX_UPLOAD_SIZE = (
    100 * 1024 * 1024
)

MAX_PREVIEW_ROWS = 100

###########################################################
# Pandas Readers
###########################################################

CSV_EXTENSION = ".csv"

EXCEL_EXTENSIONS = {
    ".xls",
    ".xlsx",
}

###########################################################
# Dataset Metadata
###########################################################

DEFAULT_ENCODING = "utf-8"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

###########################################################
# Statistics
###########################################################

SUMMARY_COLUMNS = [
    "rows",
    "columns",
    "missing_values",
    "memory_usage",
    "file_size",
]

###########################################################
# Allowed Export Formats
###########################################################

EXPORT_FORMATS = {
    "csv",
    "xlsx",
}

###########################################################
# Default Pagination
###########################################################

DEFAULT_PAGE = 1

DEFAULT_LIMIT = 20

MAX_LIMIT = 100