import os
import re
import sqlite3

from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            preview TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()

    existing = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    if existing == 0:
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (
                "admin",
                "admin@cleaira.local",
                generate_password_hash("admin123"),
            ),
        )
        conn.commit()

    conn.close()


def get_user_by_id(user_id):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return user


def get_user_by_login(login_value):
    conn = get_db()
    user = conn.execute(
        """
        SELECT * FROM users
        WHERE username = ? OR email = ?
        """,
        (login_value, login_value),
    ).fetchone()
    conn.close()
    return user


def username_taken(username):
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()
    return row is not None


def email_taken(email):
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM users WHERE email = ?",
        (email,),
    ).fetchone()
    conn.close()
    return row is not None


def create_user(username, email, password):
    conn = get_db()
    cursor = conn.execute(
        """
        INSERT INTO users (username, email, password_hash)
        VALUES (?, ?, ?)
        """,
        (username, email, generate_password_hash(password)),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id


def update_password(user_id, password):
    conn = get_db()
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (generate_password_hash(password), user_id),
    )
    conn.commit()
    conn.close()


def verify_password(user, password):
    return check_password_hash(user["password_hash"], password)


def validate_register(username, email, password, confirm):
    errors = []

    if not USERNAME_RE.match(username):
        errors.append(
            "Username must be 3-30 characters and use only letters, numbers, or underscores."
        )
    elif username_taken(username):
        errors.append("That username is already taken.")

    if not EMAIL_RE.match(email):
        errors.append("Enter a valid email address.")
    elif email_taken(email):
        errors.append("That email is already registered.")

    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")

    if password != confirm:
        errors.append("Passwords do not match.")

    return errors
