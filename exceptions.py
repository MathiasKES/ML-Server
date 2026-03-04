class MLServerPyError(Exception):
    pass

class NotInitializedError(MLServerPyError):
    pass

class AuthError(MLServerPyError):
    pass

class RequestFailedError(MLServerPyError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
