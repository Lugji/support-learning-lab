from contextlib import closing
import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import annotation_api


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "annotations.sqlite3"
    manifest_path = tmp_path / "manifest.json"

    manifest_path.write_text(json.dumps({
        "categories": ["card_arrival", "cash_withdrawal"],
        "model_version": "test-model",
    }))

    with closing(sqlite3.connect(db_path)) as connection:
        connection.executescript("""
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                margin REAL NOT NULL,
                predictions_json TEXT NOT NULL
            );
            CREATE TABLE annotations (
                sample_id INTEGER PRIMARY KEY REFERENCES samples(id),
                category TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        connection.executemany(
            "INSERT INTO samples VALUES (?, ?, ?, ?)",
            [
                (1, "Where is my card?", 0.20, "[]"),
                (2, "I cannot withdraw cash.", 0.01, "[]"),
            ],
        )
        connection.commit()

    monkeypatch.setattr(annotation_api, "DB_PATH", db_path)
    monkeypatch.setattr(annotation_api, "MANIFEST_PATH", manifest_path)

    app = FastAPI()
    app.include_router(annotation_api.router)

    with TestClient(app) as test_client:
        yield test_client


def test_lowest_margin_is_shown_first(client):
    response = client.get("/annotation/next")
    assert response.status_code == 200
    body = response.json()
    assert body["sample"]["id"] == 2
    assert body["completed"] == 0
    assert body["remaining"] == 2


def test_save_persists_category_and_advances_queue(client):
    response = client.post("/annotation", json={
        "sample_id": 2, "category": "cash_withdrawal",
    })
    assert response.status_code == 201

    with closing(sqlite3.connect(annotation_api.DB_PATH)) as connection:
        stored = connection.execute(
            "SELECT category FROM annotations WHERE sample_id = 2"
        ).fetchone()
    assert stored == ("cash_withdrawal",)

    body = client.get("/annotation/next").json()
    assert body["sample"]["id"] == 1
    assert body["completed"] == 1
    assert body["remaining"] == 1


def test_duplicate_annotation_is_rejected(client):
    payload = {"sample_id": 2, "category": "cash_withdrawal"}
    assert client.post("/annotation", json=payload).status_code == 201
    assert client.post("/annotation", json=payload).status_code == 409
    assert client.get("/annotation/next").json()["completed"] == 1


def test_invalid_category_does_not_change_queue(client):
    response = client.post("/annotation", json={
        "sample_id": 2, "category": "invented_category",
    })
    assert response.status_code == 422
    body = client.get("/annotation/next").json()
    assert body["completed"] == 0
    assert body["sample"]["id"] == 2


def test_unknown_sample_is_rejected(client):
    response = client.post("/annotation", json={
        "sample_id": 999, "category": "card_arrival",
    })
    assert response.status_code == 404
    assert client.get("/annotation/next").json()["completed"] == 0


def test_completed_queue_returns_no_sample(client):
    for sample_id, category in [
        (1, "card_arrival"),
        (2, "cash_withdrawal"),
    ]:
        response = client.post("/annotation", json={
            "sample_id": sample_id, "category": category,
        })
        assert response.status_code == 201

    body = client.get("/annotation/next").json()
    assert body["sample"] is None
    assert body["completed"] == 2
    assert body["remaining"] == 0
