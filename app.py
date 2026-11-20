"""
Smart Waste Classification System - Flask Web Application
Provides interactive web interface and RESTful APIs for waste image classification.
"""

import os
import sys
import subprocess

# Auto-switch to virtual environment python if running under global system python
_venv_candidates = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "python.exe")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"))
]
for _vp in _venv_candidates:
    if os.path.exists(_vp) and sys.executable.lower() != _vp.lower():
        res = subprocess.call([_vp] + sys.argv)
        sys.exit(res)

import uuid
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from src.database import init_db, save_prediction, get_history, get_stats, clear_history, delete_prediction
from src.inference import WasteClassifier
from src.preprocessing import ALLOWED_EXTENSIONS, validate_image_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MAX_UPLOAD_SIZE_MB = 10

# Initialize Flask application
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Initialize SQLite database schema
init_db()

# Load the trained CNN model once at server startup (Singleton pattern)
print("[Startup] Initializing Waste Classification Engine...")
classifier = WasteClassifier()
app.classifier = classifier


def is_allowed_file(filename: str) -> bool:
    """Checks whether the uploaded filename possesses a permitted image extension."""
    if "." not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# ==============================================================================
# HTML Template Routes
# ==============================================================================

@app.route("/")
def home():
    """Renders the Project Home Page with introduction and category cards."""
    return render_template(
        "index.html",
        categories=classifier.recommendations,
        model_ready=classifier.is_loaded
    )


@app.route("/classify")
def classify_page():
    """Renders the interactive Image Upload & Classification workspace."""
    return render_template(
        "classify.html",
        categories=classifier.recommendations,
        model_ready=classifier.is_loaded,
        load_error=classifier.load_error
    )


@app.route("/dashboard")
def dashboard_page():
    """Renders the Analytics Dashboard with Chart.js visualizations."""
    stats = get_stats()
    return render_template(
        "dashboard.html",
        stats=stats,
        categories=classifier.recommendations,
        model_ready=classifier.is_loaded
    )


@app.route("/history")
def history_page():
    """Renders the Prediction History log stored in SQLite."""
    history = get_history(limit=100)
    return render_template(
        "history.html",
        history=history,
        model_ready=classifier.is_loaded
    )


@app.route("/about")
def about_page():
    """Renders the Project Documentation, CNN Architecture, and Problem Statement."""
    report_path = os.path.join(BASE_DIR, "models", "evaluation_report.json")
    metrics = None
    if os.path.isfile(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    return render_template(
        "about.html",
        categories=classifier.recommendations,
        model_ready=classifier.is_loaded,
        metrics=metrics
    )


# ==============================================================================
# RESTful API Endpoints
# ==============================================================================

@app.route("/api/health", methods=["GET"])
def api_health():
    """
    Health check endpoint.
    Reports operational status and whether model weights are loaded in memory.
    """
    return jsonify({
        "status": "healthy",
        "service": "Smart Waste Classification API",
        "model_loaded": classifier.is_loaded,
        "model_path": classifier.model_path if classifier.is_loaded else None,
        "supported_classes": list(classifier.recommendations.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": (
            "Model is active and ready for inference."
            if classifier.is_loaded
            else (classifier.load_error or "Model weights not loaded.")
        )
    })


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """
    Classification endpoint.
    Accepts an uploaded image, runs CNN inference, records results in SQLite,
    and returns predicted category, confidence score, and disposal guidance.
    """
    if "image" not in request.files and "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided in the request. Use key 'image' or 'file'."
        }), 400

    file = request.files.get("image") or request.files.get("file")

    if not file or file.filename.strip() == "":
        return jsonify({
            "success": False,
            "error": "No file selected. Please choose a valid image file."
        }), 400

    if not is_allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file extension. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    # If model is not loaded, return educational guide error rather than fake prediction
    if not classifier.is_loaded:
        return jsonify({
            "success": False,
            "error": (
                "Neural network model weights are not loaded. "
                "Please run 'python train.py' to generate model weights or train via Google Colab. "
                "See the About page or README.md for step-by-step instructions."
            ),
            "model_ready": False
        }), 503

    # Generate secure, collision-resistant unique filename
    raw_filename = secure_filename(file.filename) or "upload.jpg"
    ext = os.path.splitext(raw_filename)[1].lower() or ".jpg"
    unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    try:
        # Save file securely to disk
        file.save(saved_path)

        # Validate that saved file is indeed a valid image
        is_valid, validation_msg = validate_image_file(saved_path, max_size_mb=MAX_UPLOAD_SIZE_MB)
        if not is_valid:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            return jsonify({
                "success": False,
                "error": f"Image validation failed: {validation_msg}"
            }), 400

        # Perform CNN model inference
        result, inference_error = classifier.predict(saved_path)
        if inference_error:
            return jsonify({
                "success": False,
                "error": inference_error
            }), 500

        # Persist prediction entry in SQLite database
        row_id = save_prediction(
            original_filename=raw_filename,
            saved_filename=unique_filename,
            predicted_class=result["predicted_class"],
            confidence=result["confidence_ratio"],
            category_name=result["display_name"],
            recyclable=result["recyclable"]
        )

        result["history_id"] = row_id
        result["image_url"] = f"/static/uploads/{unique_filename}"

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except Exception:
                pass
        return jsonify({
            "success": False,
            "error": f"An unexpected server error occurred during processing: {str(e)}"
        }), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns aggregated classification statistics for dashboard charts."""
    try:
        stats = get_stats()
        return jsonify({
            "success": True,
            "data": stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to retrieve stats: {str(e)}"
        }), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns recent prediction records formatted for the frontend table."""
    try:
        limit = min(int(request.args.get("limit", 50)), 200)
        offset = max(int(request.args.get("offset", 0)), 0)
        history = get_history(limit=limit, offset=offset)
        return jsonify({
            "success": True,
            "count": len(history),
            "data": history
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to retrieve history: {str(e)}"
        }), 500


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Clears all stored prediction records from the database."""
    try:
        deleted_count = clear_history()
        return jsonify({
            "success": True,
            "message": f"Successfully deleted {deleted_count} prediction record(s).",
            "deleted_count": deleted_count
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to clear history: {str(e)}"
        }), 500


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def api_delete_history_item(record_id: int):
    """Deletes a specific prediction entry by ID."""
    try:
        deleted = delete_prediction(record_id)
        if deleted:
            return jsonify({
                "success": True,
                "message": f"Record #{record_id} deleted successfully."
            })
        return jsonify({
            "success": False,
            "error": f"Record #{record_id} not found."
        }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to delete record: {str(e)}"
        }), 500


@app.route("/reports/confusion-matrix")
def view_confusion_matrix():
    """Serves the generated confusion matrix PNG."""
    models_dir = os.path.join(BASE_DIR, "models")
    matrix_path = os.path.join(models_dir, "confusion_matrix.png")
    if os.path.isfile(matrix_path):
        return send_from_directory(models_dir, "confusion_matrix.png")
    return "Confusion matrix not generated yet.", 404


@app.route("/reports/training-history")
def view_training_history():
    """Serves the generated training history plot PNG."""
    models_dir = os.path.join(BASE_DIR, "models")
    plot_path = os.path.join(models_dir, "training_history.png")
    if os.path.isfile(plot_path):
        return send_from_directory(models_dir, "training_history.png")
    return "Training history plot not generated yet.", 404


# ==============================================================================
# Application Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Host on 127.0.0.1:5000 for standard local development
    print("=" * 60)
    print(" Smart Waste Classification System - Web Server Starting")
    print(" URL: http://127.0.0.1:5000")
    print(f" Model Status: {'LOADED' if classifier.is_loaded else 'AWAITING WEIGHTS'}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)
