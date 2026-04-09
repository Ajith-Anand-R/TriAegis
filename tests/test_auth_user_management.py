from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def _login(username: str, password: str) -> str:
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return str(response.json()["access_token"])


def test_register_requires_admin_role() -> None:
    token = _login("doctor", "doctor123")
    username = f"nurse_{uuid.uuid4().hex[:10]}"

    response = client.post(
        "/api/auth/register",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "username": username,
            "password": "StrongPass123",
            "role": "Nurse",
        },
    )

    assert response.status_code == 403


def test_register_by_admin_credentials_creates_user() -> None:
    username = f"doctor_{uuid.uuid4().hex[:10]}"
    password = "StrongPass123"

    response = client.post(
        "/api/auth/register-by-admin",
        json={
            "admin_username": "admin",
            "admin_password": "admin123",
            "username": username,
            "password": password,
            "role": "Doctor",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["created"]["username"] == username
    assert payload["created"]["role"] == "Doctor"

    login_response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert login_response.status_code == 200


def test_register_by_admin_credentials_rejects_non_admin() -> None:
    username = f"nurse_{uuid.uuid4().hex[:10]}"

    response = client.post(
        "/api/auth/register-by-admin",
        json={
            "admin_username": "doctor",
            "admin_password": "doctor123",
            "username": username,
            "password": "StrongPass123",
            "role": "Nurse",
        },
    )

    assert response.status_code == 403


def test_admin_register_login_and_password_change_cycle() -> None:
    admin_token = _login("admin", "admin123")
    username = f"doctor_{uuid.uuid4().hex[:10]}"
    initial_password = "StrongPass123"
    updated_password = "StrongPass456"

    register_response = client.post(
        "/api/auth/register",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "username": username,
            "password": initial_password,
            "role": "Doctor",
        },
    )
    assert register_response.status_code == 200
    payload = register_response.json()
    assert payload["created"]["username"] == username
    assert payload["created"]["role"] == "Doctor"

    user_token = _login(username, initial_password)
    password_response = client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {user_token}"},
        json={
            "current_password": initial_password,
            "new_password": updated_password,
        },
    )
    assert password_response.status_code == 200

    old_login_response = client.post(
        "/api/auth/login",
        json={"username": username, "password": initial_password},
    )
    assert old_login_response.status_code == 401

    new_login_response = client.post(
        "/api/auth/login",
        json={"username": username, "password": updated_password},
    )
    assert new_login_response.status_code == 200


def test_change_password_rejects_wrong_current_password() -> None:
    token = _login("doctor", "doctor123")

    response = client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "current_password": "wrong-password",
            "new_password": "doctor1234",
        },
    )

    assert response.status_code == 401


def test_admin_can_list_users() -> None:
    token = _login("admin", "admin123")

    response = client.get(
        "/api/auth/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "users" in payload
    assert any(str(row.get("username")) == "admin" for row in payload["users"])


def test_admin_can_disable_then_enable_user() -> None:
    admin_token = _login("admin", "admin123")
    username = f"nurse_{uuid.uuid4().hex[:10]}"
    password = "StrongPass123"

    create_response = client.post(
        "/api/auth/register",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "username": username,
            "password": password,
            "role": "Nurse",
        },
    )
    assert create_response.status_code == 200

    disable_response = client.patch(
        "/api/auth/users/status",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"username": username, "is_active": False},
    )
    assert disable_response.status_code == 200
    assert disable_response.json()["updated"]["is_active"] is False

    disabled_login = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert disabled_login.status_code == 401

    enable_response = client.patch(
        "/api/auth/users/status",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"username": username, "is_active": True},
    )
    assert enable_response.status_code == 200
    assert enable_response.json()["updated"]["is_active"] is True

    enabled_login = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert enabled_login.status_code == 200


def test_admin_cannot_disable_own_account() -> None:
    admin_token = _login("admin", "admin123")
    response = client.patch(
        "/api/auth/users/status",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"username": "admin", "is_active": False},
    )
    assert response.status_code == 422
