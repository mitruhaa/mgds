import base64
import binascii
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path

import requests
from flask import Flask, flash, redirect, render_template, request, session, url_for
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULTS_FOLDER = BASE_DIR / "inference_results"
ASSESSMENTS_DB = Path(
    os.environ.get("ASSESSMENTS_DB_PATH", str(BASE_DIR / "assessments.db"))
)

COLAB_API_URL = os.environ.get("MEDGEMMA_API_URL", "").rstrip("/")
COLAB_API_KEY = os.environ.get("MEDGEMMA_API_KEY", "change-this-simple-key")
try:
    COLAB_API_TIMEOUT = max(1, int(os.environ.get("MEDGEMMA_API_TIMEOUT", "600")))
except ValueError:
    COLAB_API_TIMEOUT = 600
SIMULATION_MODE = "Simulation Mode (Static Run)"
LIVE_MODE = "Live Colab API Tunnel"
PIPELINE_MODES = (SIMULATION_MODE, LIVE_MODE)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_REASONING_MARKER = "<unused94>"
ESCAPED_MODEL_REASONING_MARKER = "&lt;unused94&gt;"
MODEL_ROLE_MARKER_PATTERN = re.compile(
    rf"{re.escape(MODEL_REASONING_MARKER)}\s*(thought|analysis|final)\b",
    re.IGNORECASE,
)


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "mgds-control-room-dev-secret")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


class MedGemmaAPIError(RuntimeError):
    """Raised when the Colab API cannot return a usable inference result."""


def colab_request_headers():
    return {
        "X-API-Key": COLAB_API_KEY,
        "ngrok-skip-browser-warning": "true",
    }


def check_pipeline_connection():
    """Ping the Colab notebook's root endpoint to verify the ngrok tunnel."""
    if not COLAB_API_URL:
        return False

    try:
        response = requests.get(
            f"{COLAB_API_URL}/",
            headers=colab_request_headers(),
            timeout=5,
        )
        return response.status_code == 200 and response.json().get("status") in {"ok", "online"}
    except (requests.RequestException, ValueError):
        return False


def query_medgemma_model(image_path):
    """Send an image to the Colab notebook and return its full XAI result."""
    if not COLAB_API_URL:
        raise MedGemmaAPIError("MEDGEMMA_API_URL is not configured.")

    try:
        with image_path.open("rb") as image_file:
            files = {"image": (image_path.name, image_file, "application/octet-stream")}
            response = requests.post(
                f"{COLAB_API_URL}/process",
                files=files,
                headers=colab_request_headers(),
                timeout=COLAB_API_TIMEOUT,
            )
    except OSError as exc:
        raise MedGemmaAPIError("The uploaded image could not be read.") from exc
    except requests.Timeout as exc:
        raise MedGemmaAPIError(
            f"Colab inference exceeded the {COLAB_API_TIMEOUT}-second timeout."
        ) from exc
    except requests.RequestException as exc:
        raise MedGemmaAPIError("The Colab API request failed.") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise MedGemmaAPIError("The Colab API returned a non-JSON response.") from exc

    if response.status_code != 200:
        message = (
            payload.get("error", f"HTTP {response.status_code}")
            if isinstance(payload, dict)
            else None
        )
        raise MedGemmaAPIError(f"Colab API error: {message or response.reason}")

    return normalize_api_result(payload)


def normalize_api_result(payload):
    if not isinstance(payload, dict):
        raise MedGemmaAPIError("The Colab API response is not a JSON object.")

    try:
        prediction = str(payload["prediction"]).lower()
        confidence = float(payload["confidence"])
        yes_probability = float(payload["yes_probability"])
        no_probability = float(payload["no_probability"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MedGemmaAPIError("The Colab API response is missing prediction probabilities.") from exc

    if prediction not in {"yes", "no"}:
        raise MedGemmaAPIError("The Colab API returned an unsupported prediction label.")

    probabilities = (confidence, yes_probability, no_probability)
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
        raise MedGemmaAPIError("The Colab API returned a probability outside the 0-1 range.")

    explanation, model_reasoning = split_marked_explanation(payload.get("explanation", ""))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "yes_probability": yes_probability,
        "no_probability": no_probability,
        "explanation": explanation,
        "model_reasoning": model_reasoning,
        "xai_heatmap": payload.get("xai_heatmap"),
        "xai_type": str(payload.get("xai_type", "")).strip(),
        "xai_grid_size": payload.get("xai_grid_size"),
        "xai_note": str(payload.get("xai_note", "")).strip(),
    }


def split_marked_explanation(value):
    """Separate MedGemma role-marked reasoning from its final explanation."""
    remaining = str(value or "").replace(ESCAPED_MODEL_REASONING_MARKER, MODEL_REASONING_MARKER)
    role_markers = list(MODEL_ROLE_MARKER_PATTERN.finditer(remaining))
    if role_markers:
        return split_role_marked_explanation(remaining, role_markers)

    return split_symmetric_marked_explanation(remaining)


def split_role_marked_explanation(value, role_markers):
    explanation_parts = []
    reasoning_parts = []
    current_parts = explanation_parts
    position = 0

    for marker in role_markers:
        current_parts.append(value[position:marker.start()])
        role = marker.group(1).lower()
        current_parts = explanation_parts if role == "final" else reasoning_parts
        position = marker.end()

    current_parts.append(value[position:])
    return join_text_parts(explanation_parts), join_text_parts(reasoning_parts)


def split_symmetric_marked_explanation(remaining):
    """Support legacy output that uses two plain <unused94> delimiters."""
    explanation_parts = []
    reasoning_parts = []

    while True:
        start = remaining.find(MODEL_REASONING_MARKER)
        if start == -1:
            explanation_parts.append(remaining)
            break

        end = remaining.find(MODEL_REASONING_MARKER, start + len(MODEL_REASONING_MARKER))
        if end == -1:
            explanation_parts.append(remaining)
            break

        explanation_parts.append(remaining[:start])
        reasoning_parts.append(remaining[start + len(MODEL_REASONING_MARKER):end])
        remaining = remaining[end + len(MODEL_REASONING_MARKER):]

    return join_text_parts(explanation_parts), join_text_parts(reasoning_parts, separator="\n\n")


def join_text_parts(parts, separator="\n"):
    return separator.join(part.strip() for part in parts if part.strip())


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def selected_mode():
    mode = session.get("app_mode", SIMULATION_MODE)
    return mode if mode in PIPELINE_MODES else SIMULATION_MODE


def selected_qc_score():
    try:
        qc_score = int(session.get("qc_score", 7))
    except (TypeError, ValueError):
        qc_score = 7

    return max(1, min(qc_score, 10))


def current_image_path():
    filename = session.get("uploaded_filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        return None

    image_path = UPLOAD_FOLDER / filename
    return image_path if image_path.exists() else None


def save_uploaded_image(file_storage):
    if not file_storage or not file_storage.filename:
        return None

    if not allowed_file(file_storage.filename):
        raise ValueError("Only PNG, JPG, and JPEG retinal images are supported.")

    extension = file_storage.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"
    destination = UPLOAD_FOLDER / filename

    try:
        with Image.open(file_storage.stream) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file could not be verified as an image.") from exc

    file_storage.stream.seek(0)
    file_storage.save(destination)
    return filename


def result_path_from_session():
    filename = session.get("inference_result_filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        return None

    result_path = RESULTS_FOLDER / filename
    return result_path if result_path.exists() else None


def read_inference_result():
    result_path = result_path_from_session()
    if not result_path:
        return None

    try:
        with result_path.open(encoding="utf-8") as result_file:
            result = json.load(result_file)
    except (OSError, ValueError):
        return None

    return result if isinstance(result, dict) else None


def remove_inference_artifacts():
    result = read_inference_result() or {}
    delete_inference_files(result_path_from_session(), result)
    session.pop("inference_result_filename", None)


def delete_inference_files(result_path, result):
    xai_filename = result.get("xai_filename") if isinstance(result, dict) else None
    if isinstance(xai_filename, str) and Path(xai_filename).name == xai_filename:
        try:
            (UPLOAD_FOLDER / xai_filename).unlink(missing_ok=True)
        except OSError:
            pass

    if result_path:
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass


def persist_inference_result(result):
    """Store large Colab output server-side and keep only its identifier in the session."""
    previous_result = read_inference_result() or {}
    previous_result_path = result_path_from_session()
    stored_result = dict(result)
    heatmap_data_url = stored_result.pop("xai_heatmap", None)
    new_xai_filename = None
    result_path = None

    try:
        if heatmap_data_url:
            new_xai_filename = save_xai_heatmap(heatmap_data_url)
            stored_result["xai_filename"] = new_xai_filename

        result_filename = f"{uuid.uuid4().hex}.json"
        result_path = RESULTS_FOLDER / result_filename
        with result_path.open("w", encoding="utf-8") as result_file:
            json.dump(stored_result, result_file)
    except Exception:
        if new_xai_filename:
            try:
                (UPLOAD_FOLDER / new_xai_filename).unlink(missing_ok=True)
            except OSError:
                pass
        if result_path:
            try:
                result_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    session["inference_result_filename"] = result_filename
    delete_inference_files(previous_result_path, previous_result)


def save_xai_heatmap(data_url):
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise MedGemmaAPIError("The Colab API returned an invalid XAI heatmap.")

    try:
        image_bytes = base64.b64decode(data_url[len(prefix):], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != "PNG":
                raise MedGemmaAPIError("The XAI heatmap is not a PNG image.")
            image.verify()
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise MedGemmaAPIError("The Colab API returned an unreadable XAI heatmap.") from exc

    filename = f"{uuid.uuid4().hex}_xai.png"
    (UPLOAD_FOLDER / filename).write_bytes(image_bytes)
    return filename


def reset_case_state():
    image_path = current_image_path()
    remove_inference_artifacts()
    if image_path:
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            pass

    for key in ("uploaded_filename", "original_filename", "ai_probability", "route_confirmed"):
        session.pop(key, None)


def persist_assessment(audit_reason=None):
    """Write the current human oversight assessment to the SQLite ledger."""
    result = read_inference_result()
    image_path = current_image_path()
    image_name = session.get("original_filename")

    if not result or not image_path:
        raise ValueError("A complete inference result is required before assessment.")

    if not isinstance(image_name, str) or not image_name.strip():
        image_name = image_path.name

    try:
        prediction = str(result["prediction"]).lower()
        yes_probability = float(result["yes_probability"])
        no_probability = float(result["no_probability"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The inference result is missing assessment data.") from exc

    explanation, _ = split_marked_explanation(result.get("explanation", ""))
    explanation = explanation or "error"
    xai_image = read_xai_image(result)
    normalized_reason = str(audit_reason).strip() if audit_reason else None
    assessment_id = hashlib.sha256(
        f"{time.time_ns()}:{uuid.uuid4().hex}:{image_name}".encode("utf-8")
    ).hexdigest()

    ASSESSMENTS_DB.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(ASSESSMENTS_DB)
    try:
        with database:
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS assessments (
                    id TEXT PRIMARY KEY,
                    image_name TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    yes_probability REAL NOT NULL,
                    no_probability REAL NOT NULL,
                    explanation TEXT NOT NULL,
                    xai_image BLOB,
                    audit_reason TEXT
                )
                """
            )
            database.execute(
                """
                INSERT INTO assessments (
                    id,
                    image_name,
                    prediction,
                    yes_probability,
                    no_probability,
                    explanation,
                    xai_image,
                    audit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment_id,
                    image_name,
                    prediction,
                    yes_probability,
                    no_probability,
                    explanation,
                    xai_image,
                    normalized_reason,
                ),
            )
    finally:
        database.close()

    return assessment_id


def read_xai_image(result):
    xai_filename = result.get("xai_filename")
    if not isinstance(xai_filename, str) or Path(xai_filename).name != xai_filename:
        return None

    xai_path = UPLOAD_FOLDER / xai_filename
    return xai_path.read_bytes() if xai_path.exists() else None


@app.route("/", methods=["GET", "POST"])
def control_room():
    if request.method == "POST":
        mode = request.form.get("app_mode", selected_mode())
        session["app_mode"] = mode if mode in PIPELINE_MODES else SIMULATION_MODE
        session["qc_score"] = selected_qc_score_from_form()

        action = request.form.get("action", "save_case")

        if action == "clear_case":
            reset_case_state()
            flash("Case workspace reset.", "info")
            return redirect(url_for("control_room"))

        uploaded_image = request.files.get("fundus_image")
        if uploaded_image and uploaded_image.filename:
            try:
                filename = save_uploaded_image(uploaded_image)
            except ValueError as exc:
                flash(str(exc), "error")
                return redirect(url_for("control_room"))

            reset_case_state()
            session["uploaded_filename"] = filename
            session["original_filename"] = uploaded_image.filename
            flash("Retinal image loaded for diagnostic triage.", "success")

        if action == "run_inference":
            run_inference()
        elif action == "confirm_route":
            confirm_route()
        elif action == "commit_override":
            commit_override()

        return redirect(url_for("control_room"))

    return render_template("index.html", **build_view_state())


def selected_qc_score_from_form():
    try:
        raw_score = int(request.form.get("qc_score", selected_qc_score()))
    except (TypeError, ValueError):
        raw_score = selected_qc_score()

    return max(1, min(raw_score, 10))


def run_inference():
    image_path = current_image_path()
    qc_score = selected_qc_score()
    mode = selected_mode()

    if not image_path:
        flash("Submit a retinal image before running inference.", "error")
        return

    if qc_score < 5:
        flash("Automated run constrained: retinal data quality is below analytical bounds.", "error")
        return

    if mode == LIVE_MODE:
        if not check_pipeline_connection():
            flash(
                "Colab pipeline is offline. Verify MEDGEMMA_API_URL, keep the notebook API cell "
                "running, or switch to Simulation Mode.",
                "error",
            )
            return

        try:
            result = query_medgemma_model(image_path)
        except MedGemmaAPIError as exc:
            flash(f"Prediction pipeline error: {exc}", "error")
            return
    else:
        result = {
            "prediction": "yes",
            "confidence": 0.8732,
            "yes_probability": 0.8732,
            "no_probability": 0.1268,
            "explanation": (
                "Simulation output only. Live Colab mode returns a model-generated visual "
                "explanation for the uploaded retinal image."
            ),
            "model_reasoning": "",
            "xai_heatmap": None,
            "xai_type": "simulation",
            "xai_grid_size": None,
            "xai_note": "No image-specific heatmap is generated in Simulation Mode.",
        }

    try:
        persist_inference_result(result)
    except (MedGemmaAPIError, OSError, TypeError, ValueError) as exc:
        flash(f"Could not store the inference result: {exc}", "error")
        return

    session["ai_probability"] = result["yes_probability"]
    session.pop("route_confirmed", None)
    flash("MedGemma inference and XAI processing completed.", "success")


def confirm_route():
    if session.get("ai_probability") is None:
        flash("Run inference before confirming the AI verdict.", "error")
        return

    try:
        persist_assessment()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        flash(f"Could not store the oversight assessment: {exc}", "error")
        return

    session["route_confirmed"] = True
    flash("AI verdict confirmed and assessment stored in assessments.db.", "success")


def commit_override():
    audit_reason = request.form.get("audit_reason", "").strip()

    if session.get("ai_probability") is None:
        flash("Run inference before committing an override.", "error")
        return

    if not audit_reason:
        flash("Submission incomplete: rationale field cannot remain blank.", "error")
        return

    try:
        persist_assessment(audit_reason)
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        flash(f"Could not store the oversight assessment: {exc}", "error")
        return

    session["route_confirmed"] = False
    flash("Override assessment permanently logged to assessments.db.", "success")


def build_view_state():
    mode = selected_mode()
    qc_score = selected_qc_score()
    image_path = current_image_path()
    has_image = image_path is not None
    tunnel_up = check_pipeline_connection() if mode == LIVE_MODE else False
    is_executable = mode == SIMULATION_MODE or tunnel_up
    result = read_inference_result() or {}
    explanation, embedded_reasoning = split_marked_explanation(result.get("explanation", ""))
    explanation = explanation or "error"
    model_reasoning = str(result.get("model_reasoning") or embedded_reasoning).strip()
    probability = result.get("yes_probability", session.get("ai_probability"))
    confidence = result.get("confidence")
    no_probability = result.get("no_probability")
    xai_filename = result.get("xai_filename")
    xai_path = UPLOAD_FOLDER / xai_filename if isinstance(xai_filename, str) else None
    has_xai_heatmap = (
        xai_path is not None
        and Path(xai_filename).name == xai_filename
        and xai_path.exists()
    )

    return {
        "app_mode": mode,
        "pipeline_modes": PIPELINE_MODES,
        "live_mode": LIVE_MODE,
        "simulation_mode": SIMULATION_MODE,
        "colab_api_url": COLAB_API_URL,
        "api_configured": bool(COLAB_API_URL),
        "tunnel_up": tunnel_up,
        "is_executable": is_executable,
        "qc_score": qc_score,
        "qc_blocked": has_image and qc_score < 5,
        "has_image": has_image,
        "uploaded_image_url": url_for("static", filename=f"uploads/{image_path.name}") if has_image else None,
        "original_filename": session.get("original_filename"),
        "probability": probability,
        "risk_percent": f"{float(probability) * 100:.1f}" if probability is not None else None,
        "confidence_percent": f"{float(confidence) * 100:.1f}" if confidence is not None else None,
        "no_probability_percent": (
            f"{float(no_probability) * 100:.1f}" if no_probability is not None else None
        ),
        "prediction": result.get("prediction"),
        "explanation": explanation,
        "model_reasoning": model_reasoning,
        "xai_heatmap_url": (
            url_for("static", filename=f"uploads/{xai_filename}") if has_xai_heatmap else None
        ),
        "xai_type": result.get("xai_type"),
        "xai_grid_size": result.get("xai_grid_size"),
        "xai_note": result.get("xai_note"),
        "classification": classification_for(probability),
        "route_confirmed": session.get("route_confirmed", False),
    }


def classification_for(probability):
    if probability is None:
        return None

    return "EMERGENCY ROUTING" if float(probability) >= 0.5 else "ROUTINE ASSIGNMENT"


if __name__ == "__main__":
    app.run(debug=True)
