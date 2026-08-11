
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from passlib.context import CryptContext


# -------------------------------------------------
# Password Hashers
# -------------------------------------------------

# Current password hashing algorithm
argon2_hasher = PasswordHasher()


# Legacy password hashing algorithm
# Used for users created before the Argon2 migration.
bcrypt_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


# -------------------------------------------------
# Hash Password
# -------------------------------------------------

def hash_password(password: str) -> str:
    """
    Hash new passwords using Argon2.

    All newly created users will use Argon2.
    """
    return argon2_hasher.hash(password)


# -------------------------------------------------
# Verify Password
# -------------------------------------------------

def verify_password(
    password: str,
    hashed_password: str,
) -> bool:
    """
    Verify a password against either:

    1. Argon2 hash - current
    2. bcrypt hash - legacy

    This allows existing users to continue logging in
    after migrating from bcrypt to Argon2.
    """

    if not hashed_password:
        return False

    # ---------------------------------------------
    # Current Argon2 hashes
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Legacy bcrypt hashes
    # ---------------------------------------------

    if hashed_password.startswith(
        ("$2a$", "$2b$", "$2y$")
    ):
        try:
            return bcrypt_context.verify(
                password,
                hashed_password,
            )

        except Exception:
            return False

    # ---------------------------------------------
    # Unknown hash format
    # ---------------------------------------------

    return False


# -------------------------------------------------
# Check Whether Hash Needs Migration
# -------------------------------------------------

def needs_rehash(
    hashed_password: str,
) -> bool:
    """
    Returns True if the stored password hash
    is not using the current Argon2 algorithm.
    """

    return not hashed_password.startswith("$argon2")
