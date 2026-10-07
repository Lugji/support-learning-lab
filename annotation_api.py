from contextlib import closing
from pathlib import Path
import json
import sqlite3

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "annotations.sqlite3"
MANIFEST_PATH = ROOT / "artifacts" / "annotation_manifest.json"

router = APIRouter(prefix="/annotation", tags=["Annotation"])


def read_manifest():
    if not DB_PATH.exists() or not MANIFEST_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Bitte zuerst prepare_annotation.py ausführen.",
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def connect_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


class AnnotationRequest(BaseModel):
    sample_id: int = Field(ge=0)
    category: str = Field(min_length=1, max_length=100)


@router.get("/next")
def next_sample():
    manifest = read_manifest()

    with closing(connect_db()) as connection:
        sample = connection.execute("""
            SELECT s.*
            FROM samples AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM annotations AS a
                WHERE a.sample_id = s.id
            )
            ORDER BY s.margin ASC, s.id ASC
            LIMIT 1
        """).fetchone()

        total = connection.execute(
            "SELECT COUNT(*) FROM samples"
        ).fetchone()[0]
        completed = connection.execute(
            "SELECT COUNT(*) FROM annotations"
        ).fetchone()[0]

    return {
        "sample": None if sample is None else {
            "id": sample["id"],
            "text": sample["text"],
            "margin": sample["margin"],
            "predictions": json.loads(sample["predictions_json"]),
        },
        "categories": manifest["categories"],
        "model_version": manifest["model_version"],
        "completed": completed,
        "remaining": total - completed,
    }


@router.post("", status_code=201)
def save_annotation(request: AnnotationRequest):
    manifest = read_manifest()
    category = request.category.strip()

    if category not in manifest["categories"]:
        raise HTTPException(
            status_code=422,
            detail="Bitte eine der 77 gültigen Kategorien auswählen.",
        )

    with closing(connect_db()) as connection:
        with connection:
            exists = connection.execute(
                "SELECT 1 FROM samples WHERE id = ?",
                (request.sample_id,),
            ).fetchone()

            if exists is None:
                raise HTTPException(
                    status_code=404,
                    detail="Dieser Text existiert nicht in der Warteschlange.",
                )

            try:
                connection.execute(
                    "INSERT INTO annotations (sample_id, category) VALUES (?, ?)",
                    (request.sample_id, category),
                )
            except sqlite3.IntegrityError:
                raise HTTPException(
                    status_code=409,
                    detail="Dieser Text wurde bereits annotiert.",
                ) from None

    return {
        "saved": True,
        "sample_id": request.sample_id,
        "category": category,
    }
