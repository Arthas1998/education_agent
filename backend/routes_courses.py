from flask import Blueprint, current_app, jsonify, abort

courses_bp = Blueprint("courses", __name__, url_prefix="/api")


@courses_bp.route("/courses", methods=["GET"])
def list_courses():
    pm = current_app.pdf_manager
    return jsonify(pm.list_courses())


@courses_bp.route("/course/<course_id>/page/<int:page>", methods=["GET"])
def get_page(course_id, page):
    pm = current_app.pdf_manager
    meta = pm.courses.get(course_id)
    if not meta:
        return abort(404, description="course not found")
    if page < 1 or page > meta.total_pages:
        return abort(400, description="page out of range")
    img_url = pm.get_page_image_url(course_id, page)
    if img_url:
        return jsonify({
            "page": page,
            "total_pages": meta.total_pages,
            "image_url": img_url,
        })
    # if not exists, trigger async conversion and return 202 to indicate generating
    try:
        pm.convert_async(course_id)
    except Exception:
        pass
    return (jsonify({"status": "generating"}), 202)

