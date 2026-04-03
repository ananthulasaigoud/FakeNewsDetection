import os
import hashlib
import secrets
import sqlite3
from typing import Optional, Dict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.db")

def get_db_connection():
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to SQLite (this automatically creates the file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This lets us access columns by name like dictionary keys
    
    # Ensure table exists
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            contact TEXT,
            email TEXT,
            address TEXT
        )
    ''')
    conn.commit()
    return conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def create_user(username: str, password: str, contact: str = "", email: str = "", address: str = "") -> bool:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Check if user already exists
        cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username,))
        if cursor.fetchone():
            return False
            
        salt = secrets.token_hex(16)
        password_hash = _hash_password(password, salt)
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt, contact, email, address) VALUES (?, ?, ?, ?, ?, ?)",
            (username, password_hash, salt, contact, email, address)
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error creating user in SQLite: {e}")
        return False
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT salt, password_hash FROM users WHERE LOWER(username) = LOWER(?)", (username,))
        user_row = cursor.fetchone()
        
        if user_row:
            computed_hash = _hash_password(password, user_row['salt'])
            return computed_hash == user_row['password_hash']
        return False
    except sqlite3.Error as e:
        print(f"Error verifying user in SQLite: {e}")
        return False
    finally:
        conn.close()


def get_user(username: str) -> Optional[Dict]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, contact, email, address FROM users WHERE LOWER(username) = LOWER(?)", (username,))
        user_row = cursor.fetchone()
        
        if user_row:
            return dict(user_row)
        return None
    except sqlite3.Error as e:
        print(f"Error getting user from SQLite: {e}")
        return None
    finally:
        conn.close()

