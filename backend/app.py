from flask import Flask, jsonify, send_from_directory
import os


def create_app(config: dict = None):
    """Create and configure the Flask application."""
    # import inside to avoid circular import issues during static analysis
    from backend.pdf_manager import PDFManager
    from backend.routes_courses import courses_bp
    from backend.routes_chat import chat_bp
    # local import for optional feature
    try:
        from flask_cors import CORS
    except Exception:
        CORS = None

    app = Flask(__name__, static_folder=None)

    # Basic config with sensible defaults (can be overridden by `config`)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app.config.setdefault("PROJECT_ROOT", project_root)
    app.config.setdefault("PDF_DIR", os.path.join(project_root, "data", "pdf"))
    app.config.setdefault("IMG_DIR", os.path.join(project_root, "backend", "static"))
    app.config.setdefault("STATIC_URL_PREFIX", "/static")

    if config:
        app.config.update(config)

    # Enable CORS if available
    if CORS is not None:
        CORS(app)

    # Ensure static image dir exists
    os.makedirs(app.config["IMG_DIR"], exist_ok=True)

    # Initialize PDF manager and attach to app for easy access
    app.pdf_manager = PDFManager(pdf_dir=app.config["PDF_DIR"], img_dir=app.config["IMG_DIR"], static_url_prefix=app.config["STATIC_URL_PREFIX"])

    # Register blueprints
    app.register_blueprint(courses_bp)
    app.register_blueprint(chat_bp)

    @app.route("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    # Serve generated images under IMG_DIR at STATIC_URL_PREFIX
    @app.route(f"{app.config['STATIC_URL_PREFIX']}/<path:filename>")
    def static_images(filename):
        # filename expected like "<course_id>_<title>/1.jpg"
        return send_from_directory(app.config["IMG_DIR"], filename)

    # JSON error handler for HTTP exceptions
    from werkzeug.exceptions import HTTPException

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        response = e.get_response()
        response.data = jsonify({
            "code": e.code,
            "name": e.name,
            "message": e.description
        }).data
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def handle_generic_exception(e):
        # in production you might hide details
        return jsonify({"code": 5000, "error": "InternalServerError", "message": str(e)}), 500

    return app


if __name__ == "__main__":
    # Simple runner for local development
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)
