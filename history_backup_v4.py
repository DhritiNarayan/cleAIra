import json

from auth import get_db


def save_review(user_id, source, preview, results):
    preview = (preview or "").strip()
    if len(preview) > 220:
        preview = preview[:217] + "..."

    conn = get_db()
    cursor = conn.execute(
        """
        INSERT INTO reviews (user_id, source, preview, results_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            source or "Pasted text",
            preview or "(empty document)",
            json.dumps(results),
        ),
    )
    conn.commit()
    review_id = cursor.lastrowid
    conn.close()
    return review_id


def list_reviews(user_id):
    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, source, preview, results_json, created_at
        FROM reviews
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,),
    ).fetchall()
    conn.close()

    reviews = []
    for row in rows:
        results = json.loads(row["results_json"])
        reviews.append(
            {
                "id": row["id"],
                "source": row["source"],
                "preview": row["preview"],
                "created_at": row["created_at"],
                "finding_count": len(results),
            }
        )
    return reviews


def get_review(user_id, review_id):
    conn = get_db()
    row = conn.execute(
        """
        SELECT id, source, preview, results_json, created_at
        FROM reviews
        WHERE id = ? AND user_id = ?
        """,
        (review_id, user_id),
    ).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "source": row["source"],
        "preview": row["preview"],
        "created_at": row["created_at"],
        "results": json.loads(row["results_json"]),
    }


def delete_review(user_id, review_id):
    conn = get_db()
    cursor = conn.execute(
        "DELETE FROM reviews WHERE id = ? AND user_id = ?",
        (review_id, user_id),
    )
    conn.commit()
    deleted = cursor.rowcount
    conn.close()
    return deleted > 0


def delete_selected_reviews(user_id, review_ids):
    if not review_ids:
        return 0

    clean_ids = []
    for value in review_ids:
        try:
            clean_ids.append(int(value))
        except (TypeError, ValueError):
            continue

    if not clean_ids:
        return 0

    placeholders = ", ".join("?" for _ in clean_ids)
    conn = get_db()
    cursor = conn.execute(
        f"DELETE FROM reviews WHERE user_id = ? AND id IN ({placeholders})",
        (user_id, *clean_ids),
    )
    conn.commit()
    deleted = cursor.rowcount
    conn.close()
    return deleted
