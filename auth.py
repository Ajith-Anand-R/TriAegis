from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext


class Role(str, Enum):
    Doctor = "Doctor"
    Nurse = "Nurse"
    Admin = "Admin"


pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
security = HTTPBearer(auto_error=False)

SECRET_KEY = os.getenv("TRIAEGIS_JWT_SECRET", "triaegis-dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
AUTH_DB_PATH = Path(__file__).resolve().parent / "database" / "patients.db"
MIN_PASSWORD_LENGTH = 8
USERNAME_PATTERN = re.compile(r"^[a-z0-9_.-]{3,32}$")


def security_configuration_status() -> Dict[str, object]:
    environment = os.getenv("TRIAEGIS_ENV", "development").strip().lower() or "development"
    using_default_secret = SECRET_KEY == "triaegis-dev-secret-key-change-in-production"
    production_mode = environment in {"prod", "production"}
    secure = not (production_mode and using_default_secret)

    recommendation = (
        "Set TRIAEGIS_JWT_SECRET to a strong random value before production deployment"
        if not secure
        else "Security configuration looks acceptable for current environment"
    )

    return {
        "environment": environment,
        "production_mode": production_mode,
        "using_default_secret": using_default_secret,
        "secure": secure,
        "recommendation": recommendation,
    }


def _hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def _normalize_username(username: str) -> str:
    normalized = str(username).strip().lower()
    if not normalized:
        raise ValueError("Username is required")
    if not USERNAME_PATTERN.fullmatch(normalized):
        raise ValueError("Username must be 3-32 chars and use only letters, numbers, _, -, or .")
    return normalized


def _normalize_role(role: str) -> str:
    raw = str(role).strip().lower()
    role_map = {
        "doctor": Role.Doctor.value,
        "nurse": Role.Nurse.value,
        "admin": Role.Admin.value,
    }
    normalized = role_map.get(raw)
    if not normalized:
        raise ValueError("Role must be one of: Doctor, Nurse, Admin")
    return normalized


def _validate_password_strength(password: str) -> None:
    value = str(password)
    if len(value) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")


DEFAULT_USERS: tuple[tuple[str, str, str], ...] = (
    ("doctor", "doctor123", Role.Doctor.value),
    ("nurse", "nurse123", Role.Nurse.value),
    ("admin", "admin123", Role.Admin.value),
)


def _connect() -> sqlite3.Connection:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _initialize_auth_store() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS auth_users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_auth_users_role ON auth_users(role);
            """
        )

        for username, plain_password, role in DEFAULT_USERS:
            conn.execute(
                """
                INSERT INTO auth_users (username, password_hash, role, is_active)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(username) DO UPDATE SET
                    password_hash = excluded.password_hash,
                    role = excluded.role,
                    is_active = 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (username, _hash_password(plain_password), role),
            )


def _fetch_user(username: str) -> Dict[str, str] | None:
    normalized = str(username).strip().lower()
    if not normalized:
        return None

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT username, password_hash, role, is_active
            FROM auth_users
            WHERE username = ?
            """,
            (normalized,),
        ).fetchone()

    if row is None:
        return None
    if int(row["is_active"] or 0) != 1:
        return None

    return {
        "username": str(row["username"]),
        "password_hash": str(row["password_hash"]),
        "role": str(row["role"]),
    }


_initialize_auth_store()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Dict[str, str] | None:
    try:
        normalized_username = _normalize_username(username)
    except ValueError:
        return None

    user = _fetch_user(normalized_username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {"username": user["username"], "role": user["role"]}


def list_users() -> list[Dict[str, object]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT username, role, is_active, created_at, updated_at
            FROM auth_users
            ORDER BY username ASC
            """
        ).fetchall()

    return [
        {
            "username": str(row["username"]),
            "role": str(row["role"]),
            "is_active": bool(int(row["is_active"] or 0)),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]


def create_user(username: str, password: str, role: str) -> Dict[str, object]:
    normalized_username = _normalize_username(username)
    normalized_role = _normalize_role(role)
    _validate_password_strength(password)

    with _connect() as conn:
        existing = conn.execute(
            "SELECT username FROM auth_users WHERE username = ?",
            (normalized_username,),
        ).fetchone()
        if existing is not None:
            raise ValueError("Username already exists")

        conn.execute(
            """
            INSERT INTO auth_users (username, password_hash, role, is_active)
            VALUES (?, ?, ?, 1)
            """,
            (normalized_username, _hash_password(password), normalized_role),
        )

    return {
        "username": normalized_username,
        "role": normalized_role,
        "is_active": True,
    }


def change_user_password(username: str, current_password: str, new_password: str) -> None:
    normalized_username = _normalize_username(username)
    user = _fetch_user(normalized_username)
    if user is None:
        raise ValueError("User not found")

    if not verify_password(current_password, user["password_hash"]):
        raise ValueError("Current password is incorrect")

    _validate_password_strength(new_password)
    if verify_password(new_password, user["password_hash"]):
        raise ValueError("New password must be different from current password")

    with _connect() as conn:
        conn.execute(
            """
            UPDATE auth_users
            SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
            WHERE username = ?
            """,
            (_hash_password(new_password), normalized_username),
        )


def set_user_active(username: str, is_active: bool, actor_username: str | None = None) -> Dict[str, object]:
    normalized_username = _normalize_username(username)
    normalized_actor = _normalize_username(actor_username) if actor_username else None

    if normalized_actor and normalized_actor == normalized_username and not is_active:
        raise ValueError("You cannot disable your own account")

    with _connect() as conn:
        existing = conn.execute(
            "SELECT username, role FROM auth_users WHERE username = ?",
            (normalized_username,),
        ).fetchone()
        if existing is None:
            raise ValueError("User not found")

        conn.execute(
            """
            UPDATE auth_users
            SET is_active = ?, updated_at = CURRENT_TIMESTAMP
            WHERE username = ?
            """,
            (1 if bool(is_active) else 0, normalized_username),
        )

        updated = conn.execute(
            "SELECT username, role, is_active FROM auth_users WHERE username = ?",
            (normalized_username,),
        ).fetchone()

    if updated is None:
        raise ValueError("User not found")

    return {
        "username": str(updated["username"]),
        "role": str(updated["role"]),
        "is_active": bool(int(updated["is_active"] or 0)),
    }


def create_access_token(data: Dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.now(UTC)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
        return {"username": str(username), "role": str(role)}
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> Dict[str, str]:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return verify_token(credentials.credentials)


def require_role(*roles: str):
    allowed_roles = set(roles)

    def _checker(current_user: Dict[str, str] = Depends(get_current_user)) -> Dict[str, str]:
        if current_user.get("role") not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user

    return _checker


def role_values() -> Iterable[str]:
    return [role.value for role in Role]
