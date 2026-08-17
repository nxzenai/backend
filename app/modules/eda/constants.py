from pathlib import Path

EDA_STORAGE_DIRECTORY = Path("uploads") / "eda"
EDA_TEMP_DIRECTORY = EDA_STORAGE_DIRECTORY / ".tmp"
EDA_TRASH_DIRECTORY = EDA_STORAGE_DIRECTORY / ".trash"
EDA_REPORT_DIRECTORY = EDA_STORAGE_DIRECTORY / "reports"
LEGACY_STORAGE_DIRECTORY = Path("uploads") / "datasets"

for directory in (
    EDA_STORAGE_DIRECTORY,
    EDA_TEMP_DIRECTORY,
    EDA_TRASH_DIRECTORY,
    EDA_REPORT_DIRECTORY,
):
    directory.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
SUPPORTED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",
}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
DEFAULT_PREVIEW_SIZE = 25
MAX_PREVIEW_SIZE = 100
MAX_CACHED_PREVIEW_ROWS = 200
MAX_HISTOGRAM_BINS = 100
MAX_CATEGORY_VALUES = 50
MAX_CHART_POINTS = 2_000
MAX_GROUP_RESULTS = 200
ANALYSIS_VERSION = "1.0"
NEAR_CONSTANT_THRESHOLD = 0.95
HIGH_CARDINALITY_MIN_UNIQUE = 50
HIGH_CARDINALITY_RATIO = 0.50
