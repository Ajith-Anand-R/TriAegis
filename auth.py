from __future__ import annotations

import os
from datetime import datetime, timedelta, UTC
from enum import Enum
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


def _hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


DEMO_USERS: Dict[str, Dict[str, str]] = {
    "doctor": {"username": "doctor", "password_hash": _hash_password("doctor123"), "role": Role.Doctor.value},
    "nurse": {"username": "nurse", "password_hash": _hash_password("nurse123"), "role": Role.Nurse.value},
    "admin": {"username": "admin", "password_hash": _hash_password("admin123"), "role": Role.Admin.value},
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Dict[str, str] | None:
    user = DEMO_USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {"username": user["username"], "role": user["role"]}


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
