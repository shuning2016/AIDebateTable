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

    data = request.get_json(force=True) or {}
    plan = (data.get("plan") or "").strip()

    threading.Thread(target=manager.run_round, args=(plan,), daemon=True).start()
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


# ── Context generation ────────────────────────────────────────────────────────

@app.route("/api/context/generate", methods=["POST"])
def generate_context():
    """Claude acts as insight provider, streaming a research briefing for the topic."""
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    from ai_clients import AIClient
    ai = AIClient()

    system_prompt = (
        "You are Claude, serving as an expert research analyst and debate insight provider. "
        "Your goal is to prepare comprehensive, balanced, and factual background context "
        "so that all AI debaters can engage with the topic at a high level. "
        "Be precise, neutral, and concrete. Do not argue a position."
    )

    prompt = f"""Prepare a comprehensive debate briefing for the following topic:

**{topic}**

Other AI systems (GPT, Grok, Gemini, and others) will use this briefing to debate this topic. Equip them with rich, balanced context.

Structure your briefing exactly as follows:

## 🔍 Topic Overview
2–3 sentences defining and framing the topic clearly.

## 📊 Key Facts & Data
5–7 concrete facts, statistics, or data points directly relevant to this debate.

## 🕐 Recent Developments & Trends
3–4 key recent developments, trends, or shifts in thinking that shape the current debate landscape.

## 🏛️ Major Perspectives
The main positions held by different stakeholders, experts, or schools of thought. Present each perspective fairly and accurately.

## ⚖️ Core Tensions
The fundamental trade-offs, value conflicts, or empirical disagreements at the heart of this debate.

## 🎯 Key Questions for the Debate
3–4 specific, debatable questions that the AI debaters should try to answer.

Keep the total under 650 words. Be precise, evidence-grounded, and balanced."""

    def stream_context():
        try:
            for chunk in ai.stream("claude", prompt, system_prompt=system_prompt):
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_context(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"AI Debate Forum running at http://localhost:{port}")
    app.run(debug=True, port=port, threaded=True)
