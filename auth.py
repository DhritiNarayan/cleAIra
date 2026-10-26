import os
import re
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.environ.get("CLEAIRA_DB_PATH", "users.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Backward-compatible alias used by the existing cleAIra app.
def get_db():
    return get_connection()


def _columns(conn):
    return {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()

    columns = _columns(conn)

    # Migrate the existing users table without deleting users.db.
    if "email_verified" not in columns:
        conn.execute(
            "ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 1"
        )

    if "verification_token" not in columns:
        conn.execute(
            "ALTER TABLE users ADD COLUMN verification_token TEXT"
        )

    if "verification_created_at" not in columns:
        conn.execute(
            "ALTER TABLE users ADD COLUMN verification_created_at TEXT"
        )

    conn.commit()
    conn.close()


def validate_register(username, email, password, confirm):
    errors = []

    if not username:
        errors.append("Username is required.")
    elif len(username) < 3:
        errors.append("Username must be at least 3 characters.")

    if not email:
        errors.append("Email is required.")
    elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        errors.append("Enter a valid email address.")

    if not password:
        errors.append("Password is required.")
    elif len(password) < 8:
        errors.append("Password must be at least 8 characters.")

    if password != confirm:
        errors.append("Passwords do not match.")

    conn = get_connection()

    if conn.execute(
        "SELECT 1 FROM users WHERE email = ?",
        (email,),
    ).fetchone():
        errors.append("Email is already registered.")

    conn.close()
    return errors


def create_user(username, email, password):
    conn = get_connection()
    password_hash = generate_password_hash(password)

    cursor = conn.execute(
        """
        INSERT INTO users
            (username, email, password_hash, email_verified)
        VALUES (?, ?, ?, 0)
        """,
        (username, email, password_hash),
    )

    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id


def get_user_by_id(user_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_login(login_value):
    conn = get_connection()
    row = conn.execute(
        """
        SELECT * FROM users
        WHERE lower(username) = lower(?)
           OR lower(email) = lower(?)
        LIMIT 1
        """,
        (login_value, login_value),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def verify_password(user, password):
    return bool(
        user
        and user.get("password_hash")
        and check_password_hash(user["password_hash"], password)
    )


def update_password(user_id, new_password):
    conn = get_connection()
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (generate_password_hash(new_password), user_id),
    )
    conn.commit()
    conn.close()


def set_verification_token(user_id, token):
    conn = get_connection()
    conn.execute(
        """
        UPDATE users
        SET verification_token = ?,
            verification_created_at = ?
        WHERE id = ?
        """,
        (token, datetime.utcnow().isoformat(), user_id),
    )
    conn.commit()
    conn.close()


def get_user_by_verification_token(token):
    conn = get_connection()
    row = conn.execute(
        """
        SELECT * FROM users
        WHERE verification_token = ?
          AND email_verified = 0
        LIMIT 1
        """,
        (token,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def mark_email_verified(user_id):
    conn = get_connection()
    conn.execute(
        """
        UPDATE users
        SET email_verified = 1,
            verification_token = NULL,
            verification_created_at = NULL
        WHERE id = ?
        """,
        (user_id,),
    )
    conn.commit()
    conn.close()
