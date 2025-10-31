import json
import os
import hashlib
import secrets
from typing import Optional, Dict

USERS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.json")


def _ensure_store() -> None:
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    if not os.path.exists(USERS_PATH):
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump({"users": []}, f, indent=2)


def _read_store() -> Dict:
    _ensure_store()
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_store(data: Dict) -> None:
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def create_user(username: str, password: str, contact: str = "", email: str = "", address: str = "") -> bool:
    store = _read_store()
    if any(u["username"].lower() == username.lower() for u in store["users"]):
        return False
    salt = secrets.token_hex(16)
    store["users"].append(
        {
            "username": username,
            "salt": salt,
            "password_hash": _hash_password(password, salt),
            "contact": contact,
            "email": email,
            "address": address,
        }
    )
    _write_store(store)
    return True


def verify_user(username: str, password: str) -> bool:
    store = _read_store()
    for user in store["users"]:
        if user["username"].lower() == username.lower():
            return _hash_password(password, user["salt"]) == user["password_hash"]
    return False


def get_user(username: str) -> Optional[Dict]:
    store = _read_store()
    for user in store["users"]:
        if user["username"].lower() == username.lower():
            return user
    return None
