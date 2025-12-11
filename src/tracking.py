"""Lightweight MLflow logging helpers for the T2KAAA app."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

from PIL import Image

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

_DEFAULT_TRACKING_URI = os.environ.get("T2KAAA_MLFLOW_TRACKING_URI", "file:mlruns")


def _ensure_local_tracking_dir(uri: str) -> None:
    if not uri.startswith("file:"):
        return
    raw_path = uri[5:]
    if raw_path.startswith("//"):
        raw_path = raw_path[2:]
    local_path = Path(raw_path)
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    local_path.mkdir(parents=True, exist_ok=True)

try:  # pragma: no cover - MLflow is optional at runtime
    import mlflow
except Exception:  # pylint: disable=broad-except
    mlflow = None  # type: ignore[assignment]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _mlflow_available() -> bool:
    return mlflow is not None and _env_flag("T2KAAA_ENABLE_MLFLOW", True)


def _prepare_client() -> bool:
    if not _mlflow_available():
        return False
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
    _ensure_local_tracking_dir(tracking_uri)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = os.environ.get("T2KAAA_MLFLOW_EXPERIMENT", "T2KAAA")
    try:
        mlflow.set_experiment(experiment)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to set MLflow experiment '{experiment}': {exc}")
        return False
    return True


@contextlib.contextmanager
def start_run(run_name: str, tags: Mapping[str, Any] | None = None):
    if not _prepare_client():
        yield None
        return
    try:
        with mlflow.start_run(run_name=run_name, tags=tags, nested=True) as run:  # type: ignore[attr-defined]
            yield run
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] MLflow run '{run_name}' failed: {exc}")
        yield None


def log_event(
    run_name: str,
    *,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    tags: Mapping[str, Any] | None = None,
    text_artifacts: Mapping[str, str] | None = None,
    image_artifacts: Mapping[str, Image.Image] | None = None,
) -> None:
    if not _prepare_client():
        return
    with start_run(run_name, tags=tags):
        _log_params(params)
        _log_metrics(metrics)
        _log_texts(text_artifacts)
        _log_images(image_artifacts)


def _log_params(params: Mapping[str, Any] | None):
    if not params or not _mlflow_available():
        return
    try:
        cleaned: Dict[str, Any] = {k: v for k, v in params.items() if v is not None}
        if cleaned:
            mlflow.log_params(cleaned)  # type: ignore[attr-defined]
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to log MLflow params: {exc}")


def _log_metrics(metrics: Mapping[str, float] | None):
    if not metrics or not _mlflow_available():
        return
    try:
        mlflow.log_metrics(metrics)  # type: ignore[attr-defined]
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to log MLflow metrics: {exc}")


def _log_texts(texts: Mapping[str, str] | None):
    if not texts or not _mlflow_available():
        return
    for filename, content in texts.items():
        if not content:
            continue
        try:
            mlflow.log_text(str(content), artifact_file=filename)  # type: ignore[attr-defined]
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to log MLflow text artifact '{filename}': {exc}")


def _log_images(images: Mapping[str, Image.Image] | None):
    if not images or not _mlflow_available():
        return
    for filename, image in images.items():
        if image is None:
            continue
        try:
            if hasattr(mlflow, "log_image"):
                mlflow.log_image(image, artifact_file=filename)  # type: ignore[attr-defined]
            else:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp, format="PNG")
                    tmp_path = tmp.name
                    parent = Path(filename).parent
                    artifact_path = None if str(parent) in {"", "."} else str(parent)
                    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)  # type: ignore[attr-defined]
                    os.unlink(tmp_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to log MLflow image artifact '{filename}': {exc}")
