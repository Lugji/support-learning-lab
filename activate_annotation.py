from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import socket
import sqlite3
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DB = ROOT / "data" / "annotations.sqlite3"
MANIFEST = ARTIFACTS / "annotation_manifest.json"
PENDING = ARTIFACTS / "annotation_activation_pending.json"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def restore(backup):
    with closing(sqlite3.connect(backup / "annotations.sqlite3")) as source:
        with closing(sqlite3.connect(DB)) as destination:
            source.backup(destination)
    write_json(
        MANIFEST,
        json.loads((backup / "manifest.json").read_text(encoding="utf-8")),
    )


def main():
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            raise SystemExit("Bitte zuerst das Backend mit Control+C stoppen.")

    if PENDING.exists():
        pending = json.loads(PENDING.read_text(encoding="utf-8"))
        restore(Path(pending["backup"]))
        PENDING.unlink()
        raise SystemExit(
            "Unterbrochene Aktivierung zurückgesetzt. Befehl erneut ausführen."
        )

    if len(sys.argv) != 2 or not re.fullmatch(r"[0-9a-f]{12}", sys.argv[1]):
        raise SystemExit("Aufruf: python activate_annotation.py TRAININGSLAUF")

    run_id = sys.argv[1]
    run_dir = ARTIFACTS / "annotation_runs" / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads(
        (ROOT / "reports" / f"annotation_training_{run_id}.json")
        .read_text(encoding="utf-8")
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if metadata["run_id"] != run_id or report["run_id"] != run_id:
        raise SystemExit("Trainingslauf und Dateien passen nicht zusammen.")
    if metadata["source_manifest"] != manifest:
        raise SystemExit("Das aktive Modell hat sich geändert. Bitte neu trainieren.")
    if report["macro_f1_after"] < report["macro_f1_before"]:
        raise SystemExit("Dieser Kandidat hat einen niedrigeren Validierungswert.")

    with closing(sqlite3.connect(f"{DB.as_uri()}?mode=ro", uri=True)) as connection:
        annotations = [
            {"sample_id": row[0], "category": row[1]}
            for row in connection.execute(
                "SELECT sample_id, category FROM annotations ORDER BY sample_id"
            )
        ]
        samples = connection.execute(
            "SELECT id, text FROM samples ORDER BY id"
        ).fetchall()

    if annotations != metadata["annotations"]:
        raise SystemExit("Die Labels haben sich geändert. Bitte erneut trainieren.")

    model = joblib.load(run_dir / "model.joblib")
    if list(model.classes_) != manifest["categories"]:
        raise SystemExit("Die Modellkategorien passen nicht zur Warteschlange.")

    print("Berechne neue Modellvorschläge …", flush=True)
    probabilities = model.predict_proba([text for _, text in samples])
    if not np.isfinite(probabilities).all():
        raise SystemExit("Ungültige Modellwerte. Aktivierung abgebrochen.")

    updates = []
    for (sample_id, _), scores in zip(samples, probabilities):
        top = scores.argsort()[-3:][::-1]
        predictions = [
            {
                "category": str(model.classes_[index]),
                "probability": float(scores[index]),
            }
            for index in top
        ]
        updates.append((
            float(scores[top[0]] - scores[top[1]]),
            json.dumps(predictions),
            sample_id,
        ))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup = ROOT / "data" / "backups" / f"activation_{stamp}"
    backup.mkdir(parents=True)
    write_json(backup / "manifest.json", manifest)
    with closing(sqlite3.connect(DB)) as source:
        with closing(sqlite3.connect(backup / "annotations.sqlite3")) as destination:
            source.backup(destination)

    new_manifest = {
        **manifest,
        "model_version": run_id,
        "model_file": f"annotation_runs/{run_id}/model.joblib",
    }
    write_json(PENDING, {"backup": str(backup), "run_id": run_id})

    try:
        with closing(sqlite3.connect(DB)) as connection:
            with connection:
                connection.executemany(
                    "UPDATE samples SET margin = ?, predictions_json = ? WHERE id = ?",
                    updates,
                )
                write_json(MANIFEST, new_manifest)
        PENDING.unlink()
    except BaseException:
        restore(backup)
        PENDING.unlink(missing_ok=True)
        raise

    print(f"Annotation-Modell aktiviert: {run_id}")
    print(f"Annotationen erhalten: {len(annotations)}")
    print(f"Offene Texte neu sortiert: {len(samples) - len(annotations)}")
    print(f"Sicherung: {backup}")


if __name__ == "__main__":
    main()
