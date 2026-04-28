# dashboard.py
import cv2
import os
from flask        import Flask, Response, render_template, jsonify, send_from_directory
from shared_state import state

app = Flask(__name__, template_folder="templates", static_folder="static")


def _generate_frames():
    import time
    while True:
        frame = state.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/map")
def survivor_map():
    """Serves folium map from SharedState memory — no file reading."""
    html = state.get_map_html()
    if html:
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}
    return (
        "<div style='font-family:Arial;padding:30px;text-align:center;color:#555;'>"
        "<h3>🗺️ Map loading — waiting for first detection...</h3>"
        "</div>"
    ), 200


@app.route("/api/stats")
def api_stats():
    return jsonify(state.get_stats())


@app.route("/api/logs")
def api_logs():
    return jsonify(state.get_log()[:20])


@app.route("/api/gps")
def api_gps():
    return jsonify(state.get_gps_points())


@app.route("/screenshots/<path:filename>")
def screenshots(filename):
    folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "screenshots"
    )
    return send_from_directory(folder, filename)


def run_dashboard(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
