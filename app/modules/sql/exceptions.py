from app.shared.exceptions.base import AppException


class SQLExecutionFailed(AppException):

    status_code = 400

    message = "SQL query execution failed."


class InvalidSQLQuery(AppException):

    status_code = 400

    message = "Invalid SQL query."


class UnsupportedOperation(AppException):

    status_code = 403

    message = "This SQL operation is not allowed."