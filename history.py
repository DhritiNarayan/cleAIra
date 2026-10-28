import json

from auth import get_db


def save_review(user_id, source, preview, results, summary=None, overall_analysis=None):
    preview = (preview or "").strip()
    if len(preview) > 220:
        preview = preview[:217] + "..."

    review_payload = {
        "results": results,
        "summary": summary,
        "overall_analysis": overall_analysis,
    }

    with get_db() as conn:
        row = conn.execute(
            """
            INSERT INTO reviews (user_id, source, preview, results_json)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                source or "Pasted text",
                preview or "(empty document)",
                json.dumps(review_payload),
            ),
        ).fetchone()
        return row["id"]


def _decode_review_payload(raw_json):
    data = json.loads(raw_json)

    if isinstance(data, list):
        return {"results": data, "summary": None, "overall_analysis": None}

    if isinstance(data, dict):
        return {
            "results": data.get("results", []),
            "summary": data.get("summary"),
            "overall_analysis": data.get("overall_analysis"),
        }

    return {"results": [], "summary": None, "overall_analysis": None}


def list_reviews(user_id):
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, source, preview, results_json, created_at
            FROM reviews
            WHERE user_id = %s
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()

    reviews = []
    for row in rows:
        payload = _decode_review_payload(row["results_json"])
        reviews.append({
            "id": row["id"],
            "source": row["source"],
            "preview": row["preview"],
            "created_at": row["created_at"],
            "finding_count": len(payload["results"]),
        })
    return reviews


def get_review(user_id, review_id):
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id, source, preview, results_json, created_at
            FROM reviews
            WHERE id = %s AND user_id = %s
            """,
            (review_id, user_id),
        ).fetchone()

    if not row:
        return None

    payload = _decode_review_payload(row["results_json"])
    return {
        "id": row["id"],
        "source": row["source"],
        "preview": row["preview"],
        "created_at": row["created_at"],
        "results": payload["results"],
        "summary": payload["summary"],
        "overall_analysis": payload["overall_analysis"],
    }


def delete_review(user_id, review_id):
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM reviews WHERE id = %s AND user_id = %s",
            (review_id, user_id),
        )
        return cursor.rowcount > 0


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

    # Build placeholders only for integer IDs, never for user-supplied SQL.
    placeholders = ", ".join("%s" for _ in clean_ids)
    with get_db() as conn:
        cursor = conn.execute(
            f"DELETE FROM reviews WHERE user_id = %s AND id IN ({placeholders})",
            (user_id, *clean_ids),
        )
        return cursor.rowcount
