from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
import bcrypt


argon2_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """
    Hash new passwords using Argon2.
    """
    return argon2_hasher.hash(password)


def verify_password(
    password: str,
    hashed_password: str,
) -> bool:
    """
    Supports:
    - Argon2 hashes
    - Legacy bcrypt hashes
    """

    if not password or not hashed_password:
        return False

    # Current Argon2 hashes
    if hashed_password.startswith("$argon2"):
        try:
            return argon2_hasher.verify(
                hashed_password,
                password,
            )
        except (
            VerifyMismatchError,
            InvalidHashError,
        ):
            return False

    # Legacy bcrypt hashes
    if hashed_password.startswith(
        ("$2a$", "$2b$", "$2y$")
    ):
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"),
                hashed_password.encode("utf-8"),
            )
        except (ValueError, TypeError):
            return False

    return False


def needs_rehash(hashed_password: str) -> bool:
    """
    True when the stored hash is not Argon2.
    """
    return not hashed_password.startswith("$argon2")