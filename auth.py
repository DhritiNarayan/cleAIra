import os
import re
from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from werkzeug.security import generate_password_hash, check_password_hash


def get_connection():
    """Open a PostgreSQL connection using the Neon DATABASE_URL."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add your Neon PostgreSQL connection string "
            "to your environment variables."
        )
    return psycopg.connect(database_url, row_factory=dict_row)


# Backward-compatible alias used by the existing cleAIra app and history.py.
def get_db():
    return get_connection()


def init_db():
    """Create the application tables if they do not already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                email_verified BOOLEAN NOT NULL DEFAULT TRUE,
                verification_token TEXT,
                verification_created_at TIMESTAMPTZ
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                source TEXT NOT NULL DEFAULT 'Pasted text',
                preview TEXT NOT NULL DEFAULT '(empty document)',
                results_json TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_user_id_id ON reviews (user_id, id DESC)"
        )


def validate_register(username, email, password, confirm):
    errors = []
    username = (username or "").strip()
    email = (email or "").strip()

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

    if email:
        with get_connection() as conn:
            if conn.execute(
                "SELECT 1 FROM users WHERE lower(email) = lower(%s)",
                (email,),
            ).fetchone():
                errors.append("Email is already registered.")

    return errors


def create_user(username, email, password):
    password_hash = generate_password_hash(password)
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO users (username, email, password_hash, email_verified)
            VALUES (%s, %s, %s, FALSE)
            RETURNING id
            """,
            (username.strip(), email.strip(), password_hash),
        ).fetchone()
        return row["id"]


def get_user_by_id(user_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_login(login_value):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM users
            WHERE lower(username) = lower(%s)
               OR lower(email) = lower(%s)
            LIMIT 1
            """,
            (login_value, login_value),
        ).fetchone()
        return dict(row) if row else None


def verify_password(user, password):
    return bool(
        user
        and user.get("password_hash")
        and check_password_hash(user["password_hash"], password)
    )


def update_password(user_id, new_password):
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (generate_password_hash(new_password), user_id),
        )


def set_verification_token(user_id, token):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET verification_token = %s,
                verification_created_at = %s
            WHERE id = %s
            """,
            (token, datetime.utcnow(), user_id),
        )


def get_user_by_verification_token(token):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM users
            WHERE verification_token = %s
              AND email_verified = FALSE
            LIMIT 1
            """,
            (token,),
        ).fetchone()
        return dict(row) if row else None


def mark_email_verified(user_id):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET email_verified = TRUE,
                verification_token = NULL,
                verification_created_at = NULL
            WHERE id = %s
            """,
            (user_id,),
        )
