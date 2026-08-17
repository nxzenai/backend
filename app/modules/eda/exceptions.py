from app.core.exceptions.custom import AIStudioException


class EDAError(AIStudioException):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        error_code: str = "EDA_ERROR",
        details=None,
    ):
        super().__init__(message, status_code, error_code, details)


class EDAInvalidRequest(EDAError):
    def __init__(self, message: str, details=None):
        super().__init__(message, 400, "EDA_INVALID_REQUEST", details)


class EDAProjectNotFound(EDAError):
    def __init__(self):
        super().__init__("EDA project not found.", 404, "EDA_PROJECT_NOT_FOUND")


class EDAUnsupportedFile(EDAError):
    def __init__(
        self, message: str = "Unsupported file type. Upload CSV, XLS, or XLSX."
    ):
        super().__init__(message, 415, "EDA_UNSUPPORTED_FILE")


class EDAUploadTooLarge(EDAError):
    def __init__(self):
        super().__init__(
            "Upload exceeds the 100 MB limit.", 413, "EDA_UPLOAD_TOO_LARGE"
        )


class EDAConflict(EDAError):
    def __init__(self, message: str):
        super().__init__(message, 409, "EDA_CONFLICT")
