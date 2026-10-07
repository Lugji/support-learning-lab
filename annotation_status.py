import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from annotation_api import read_manifest

ROOT = Path(__file__).resolve().parent
router = APIRouter(prefix="/annotation", tags=["Annotation"])


@router.get("/status")
def annotation_status():
    manifest = read_manifest()
    version = manifest["model_version"]
    training = None

    if re.fullmatch(r"[0-9a-f]{12}", version):
        path = ROOT / "reports" / f"annotation_training_{version}.json"
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail="Der Trainingsbericht des aktiven Modells fehlt.",
            )
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["run_id"] != version:
            raise HTTPException(
                status_code=503,
                detail="Trainingsbericht und Modellversion passen nicht zusammen.",
            )
        training = {
            key: report[key]
            for key in (
                "training_examples",
                "human_annotations",
                "validation_examples",
                "macro_f1_before",
                "macro_f1_after",
                "difference",
            )
        }

    return {"model_version": version, "training": training}
