import os
import json
import uuid
import queue
import threading
import time

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from debate_manager import DebateManager

app = Flask(__name__)
CORS(app)

# session_id -> DebateManager
sessions: dict[str, DebateManager] = {}
# session_id -> queue.Queue  (thread-safe event stream)
event_queues: dict[str, queue.Queue] = {}


# ── Static ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ── Session management ───────────────────────────────────────────────────────

@app.route("/api/session/create", methods=["POST"])
def create_session():
    data = request.get_json(force=True)

    debaters = data.get("debaters", [])
    if len(debaters) < 2:
        return jsonify({"error": "Select at least 2 debaters"}), 400

    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    session_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()

    manager = DebateManager(
        session_id=session_id,
        debaters=debaters,
        topic=topic,
        context=data.get("context", ""),
        max_rounds=int(data.get("rounds", 2)),
        event_queue=q,
    )

    sessions[session_id] = manager
    event_queues[session_id] = q

    return jsonify({"session_id": session_id})


@app.route("/api/session/<session_id>/status")
def session_status(session_id):
    manager = sessions.get(session_id)
    if not manager:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "phase": manager.phase,
        "current_round": manager.current_round,
        "max_rounds": manager.max_rounds,
        "debaters": manager.debaters,
    })


# ── SSE stream ───────────────────────────────────────────────────────────────

@app.route("/api/stream/<session_id>")
def stream_events(session_id):
    q = event_queues.get(session_id)
    if not q:
        return jsonify({"error": "Session not found"}), 404

    def generate():
        while True:
            try:
                event = q.get(timeout=20)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "debate_ended":
                    break
            except queue.Empty:
                # heartbeat to keep connection alive
                yield 'data: {"type":"heartbeat"}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Debate actions ────────────────────────────────────────────────────────────

@app.route("/api/session/<session_id>/facilitate", methods=["POST"])
def run_facilitation(session_id):
    manager = sessions.get(session_id)
    if not manager:
        return jsonify({"error": "Session not found"}), 404
    if manager.is_busy:
        return jsonify({"error": "Session is busy"}), 409

    threading.Thread(target=manager.run_facilitation, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/session/<session_id>/round", methods=["POST"])
def run_round(session_id):
    manager = sessions.get(session_id)
    if not manager:
        return jsonify({"error": "Session not found"}), 404
    if manager.is_busy:
        return jsonify({"error": "Session is busy"}), 409
    if manager.current_round >= manager.max_rounds:
        return jsonify({"error": "Maximum rounds reached"}), 400

    threading.Thread(target=manager.run_round, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/session/<session_id>/interrupt", methods=["POST"])
def interrupt(session_id):
    data = request.get_json(force=True)
    manager = sessions.get(session_id)
    q = event_queues.get(session_id)

    if not manager or not q:
        return jsonify({"error": "Session not found"}), 404

    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400

    manager.add_user_input(message)
    q.put({"type": "user_input", "text": message, "ts": time.time()})
    return jsonify({"status": "added"})


@app.route("/api/session/<session_id>/summarize", methods=["POST"])
def summarize(session_id):
    manager = sessions.get(session_id)
    if not manager:
        return jsonify({"error": "Session not found"}), 404
    if manager.is_busy:
        return jsonify({"error": "Session is busy"}), 409

    threading.Thread(target=manager.run_summary, daemon=True).start()
    return jsonify({"status": "started"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"AI Debate Forum running at http://localhost:{port}")
    app.run(debug=True, port=port, threaded=True)
