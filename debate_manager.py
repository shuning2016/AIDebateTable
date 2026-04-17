"""
DebateManager owns the full state of one debate session and drives all AI calls.
All AI text is streamed as events into the provided queue.Queue so the SSE
endpoint in app.py can forward them to the browser in real time.
"""

import re
import time
import threading
import queue
from typing import Optional

from ai_clients import AIClient

# ── Debater personas ──────────────────────────────────────────────────────────

PERSONAS: dict[str, str] = {
    "grok": (
        "You are Grok, xAI's AI assistant. You are known for intellectual courage, "
        "wit, and a willingness to challenge conventional wisdom and mainstream narratives. "
        "You debate with sharp logic, occasional irreverence, and genuine curiosity. "
        "You are not afraid to stake out a contrarian position if the evidence supports it."
    ),
    "claude": (
        "You are Claude, Anthropic's AI assistant. You are known for careful reasoning, "
        "intellectual honesty, and nuanced ethical thinking. You debate by steelmanning "
        "opposing views before refuting them, acknowledging uncertainty, and grounding "
        "arguments in both evidence and principle."
    ),
    "gpt": (
        "You are GPT, OpenAI's AI assistant. You are known for comprehensive, balanced "
        "analysis and clear communication. You debate by systematically examining all "
        "angles, synthesising diverse viewpoints, and presenting well-structured arguments "
        "that are accessible and thorough."
    ),
    "gemini": (
        "You are Gemini, Google's AI assistant. You are known for being data-driven, "
        "evidence-focused, and integrating information from diverse sources. You debate "
        "with precision, citing concrete facts and statistics, and connecting ideas "
        "across domains to build rigorous arguments."
    ),
}


class DebateManager:
    """Manages one debate session from facilitation through summary."""

    def __init__(
        self,
        session_id: str,
        debaters: list[str],
        topic: str,
        context: str,
        event_queue: queue.Queue,
    ):
        self.session_id = session_id
        self.debaters = debaters
        self.topic = topic
        self.context = context
        self.q = event_queue

        self.current_round: int = 0
        self.phase: str = "setup"
        self.is_busy: bool = False

        self.history: list[dict] = []
        self.user_inputs: list[str] = []
        self.next_round_plan: str = ""
        self._lock = threading.Lock()

        self.ai = AIClient()

    # ── Public mutators ───────────────────────────────────────────────────────

    def add_user_input(self, message: str) -> None:
        with self._lock:
            self.user_inputs.append(message)
            self.history.append({
                "role": "user_input",
                "speaker": "moderator",
                "text": message,
                "ts": time.time(),
            })

    def set_next_round_plan(self, plan: str) -> None:
        with self._lock:
            self.next_round_plan = plan

    # ── Debate phases ─────────────────────────────────────────────────────────

    def run_facilitation(self) -> None:
        """Facilitator opens the debate with background and framing."""
        self._begin("facilitation")
        debaters_list = " · ".join(d.upper() for d in self.debaters)

        prompt = f"""You are a neutral, expert debate facilitator opening a structured AI debate.

**Topic**: {self.topic}
**Context provided**: {self.context or "None"}
**Debaters**: {debaters_list}

Structure your opening in these four sections:

### 📚 Background
Provide 2–3 paragraphs of key facts, history, and context that all debaters and the audience need. Be factual and balanced.

### 🔑 Central Questions
List 3–5 specific questions this debate will try to answer (bullet points).

### 🗺️ Landscape of Views
Briefly preview the main positions debaters are likely to take and why reasonable people disagree.

### 🎙️ Opening the Debate
Formally welcome the debaters ({debaters_list}) and invite them to present their opening arguments in Round 1.

Keep the total under 450 words. Be engaging and neutral."""

        self._stream_speaker("facilitator", prompt)
        self._end("ready")

    def run_round(self, plan: str = "") -> None:
        """Each selected debater speaks once, then the facilitator runs a round summary."""
        self._begin(f"round_{self.current_round + 1}")
        self.current_round += 1

        self._push({
            "type": "phase_change",
            "phase": f"round_{self.current_round}",
            "round": self.current_round,
            "label": f"Round {self.current_round}",
        })

        # Snapshot and clear per-round state
        with self._lock:
            round_user_inputs = list(self.user_inputs)
            self.user_inputs.clear()
            round_plan = plan or self.next_round_plan
            self.next_round_plan = ""

        for debater in self.debaters:
            prompt = self._build_debater_prompt(debater, round_user_inputs, round_plan)
            persona = PERSONAS.get(debater, "You are a debate participant.")
            self._stream_speaker(debater, prompt, system_prompt=persona, round_num=self.current_round)

        # Auto-run round summary after all debaters finish
        self._run_round_summary()

        self._end("ready")

    def run_summary(self) -> None:
        """Facilitator produces a final summary of the full debate."""
        self._begin("summary")

        transcript = self._full_transcript()

        prompt = f"""You are the debate facilitator. The debate on **"{self.topic}"** has concluded \
after {self.current_round} round(s) with participants: {', '.join(d.upper() for d in self.debaters)}.

**Full Transcript**:
{transcript}

Write a structured final summary with exactly these sections:

## ✅ Points of Agreement
Specific conclusions or facts that all or most debaters converged on. Be concrete.

## ⚔️ Persistent Disagreements
The core positions where debaters fundamentally diverged. Explain *why* they disagreed—different values, different facts, different weight given to evidence.

## 💡 Strongest Arguments
For each debater, identify their single most compelling argument (cite specific reasoning they used).

## ❓ Open Questions
3–5 important questions the debate raised but did not fully resolve.

## 📋 Overall Assessment
What did this debate accomplish? What should be explored next?

Be specific—reference actual statements from the transcript. Keep each section concise."""

        self._stream_speaker("summary", prompt)
        self._end("done")
        self._push({"type": "debate_ended"})

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _push(self, event: dict) -> None:
        self.q.put(event)

    def _begin(self, phase: str) -> None:
        self.phase = phase
        self.is_busy = True

    def _end(self, phase: str) -> None:
        self.phase = phase
        self.is_busy = False

    def _run_round_summary(self) -> None:
        """Facilitator summarizes the completed round and proposes the next round focus."""
        self._push({
            "type": "phase_change",
            "phase": f"round_summary_{self.current_round}",
            "round": self.current_round,
            "label": f"Round {self.current_round} Summary",
        })

        round_transcript = self._round_transcript(self.current_round)

        prompt = f"""You are the debate facilitator. Round {self.current_round} has just concluded.

**Topic**: {self.topic}
**Participants**: {', '.join(d.upper() for d in self.debaters)}

**Round {self.current_round} Transcript**:
{round_transcript}

Write a concise round debrief in exactly these sections:

## ✅ Points of Agreement
What the debaters explicitly or implicitly agreed on this round. Be specific — name the claim and which debaters share it.

## ⚔️ Points of Disagreement
Where they fundamentally diverged. Name which debater holds which position and why the gap exists.

## 💡 Strongest Argument This Round
The single most compelling point made (any debater) and why it advances the debate.

## 🗺️ Proposed Focus for Round {self.current_round + 1}
3–4 concrete questions or unresolved tensions the next round should tackle, ranked by importance.

Keep under 320 words. Be crisp and actionable — the human moderator will review and may revise this plan before triggering the next round."""

        self._stream_speaker("round_summary", prompt)

        # Extract the proposed plan section to pre-fill the UI textarea
        with self._lock:
            summary_text = self.history[-1]["text"] if self.history else ""

        proposed_plan = self._extract_proposed_plan(summary_text)

        self._push({
            "type": "round_summary_done",
            "round": self.current_round,
            "proposed_plan": proposed_plan,
        })

    def _extract_proposed_plan(self, text: str) -> str:
        """Extract the 'Proposed Focus' section from the round summary text."""
        match = re.search(
            r"##[^\n]*Proposed[^\n]*\n+(.*?)(?:\n##|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    def _stream_speaker(
        self,
        speaker: str,
        prompt: str,
        system_prompt: str = "",
        round_num: Optional[int] = None,
    ) -> None:
        """Stream an AI response for one speaker, pushing events to the queue."""
        meta: dict = {"type": "speaker_start", "speaker": speaker}
        if round_num is not None:
            meta["round"] = round_num
        self._push(meta)

        facilitator_model = self.ai.get_facilitator_model()
        model_key = speaker if speaker in ("claude", "gpt", "grok", "gemini") else facilitator_model

        full_text = ""
        try:
            for chunk in self.ai.stream(model_key, prompt, system_prompt=system_prompt):
                full_text += chunk
                self._push({"type": "text_chunk", "speaker": speaker, "text": chunk})
        except Exception as exc:
            error_msg = f"\n[Stream error: {exc}]"
            full_text += error_msg
            self._push({"type": "text_chunk", "speaker": speaker, "text": error_msg})

        entry: dict = {
            "role": "facilitator" if speaker in ("facilitator", "summary", "round_summary") else "debater",
            "speaker": speaker,
            "text": full_text,
            "ts": time.time(),
        }
        if round_num is not None:
            entry["round"] = round_num

        with self._lock:
            self.history.append(entry)

        done_meta: dict = {"type": "speaker_done", "speaker": speaker}
        if round_num is not None:
            done_meta["round"] = round_num
        self._push(done_meta)

    def _build_debater_prompt(
        self,
        debater: str,
        round_user_inputs: list[str],
        round_plan: str,
    ) -> str:
        history_text = self._history_for_debater()

        moderator_block = ""
        if round_user_inputs:
            inputs = "\n".join(f'  {i+1}. "{m}"' for i, m in enumerate(round_user_inputs))
            moderator_block = f"""
---
🎯 **THE HUMAN HAS ENTERED THE DEBATE — THIS IS YOUR TOP PRIORITY:**
{inputs}

You MUST treat this as a direct challenge or contribution to the debate:
- **Quote or closely paraphrase** their specific point(s) in your response.
- **Take a clear stance**: do you agree, partially agree, or disagree — and why?
- **Build your argument around it**: let their input reshape or sharpen your position.
- Do NOT bury it at the end. Engage with it as the first or central part of your response.
---"""

        plan_block = ""
        if round_plan:
            plan_block = f"""
---
📋 **FOCUS FOR THIS ROUND** (set by the human moderator):
{round_plan}

Structure your argument around these questions/areas. Do not go off-topic.
---"""

        return f"""You are participating in **Round {self.current_round}** of a structured debate.

**Topic**: {self.topic}
**Context**: {self.context or "None provided"}
{moderator_block}{plan_block}
**Debate so far**:
{history_text}

**Your task for Round {self.current_round}**:
- If Round 1: State your opening position clearly and explain your core reasoning.
- If Round 2+: Directly respond to what the other debaters said. Quote or reference specific points you agree or disagree with.
- Be specific, evidence-based, and intellectually honest.
- Acknowledge the strongest opposing argument before refuting it.
- Keep your response to 2–4 focused paragraphs.

Write your Round {self.current_round} argument now:"""

    def _history_for_debater(self) -> str:
        """Condensed history suitable for inclusion in a debater prompt."""
        if not self.history:
            return "No prior statements — this is the opening."

        parts = []
        for entry in self.history:
            role = entry["role"]
            speaker = entry["speaker"].upper()
            text = entry["text"]
            snippet = text if len(text) <= 700 else text[:700] + "…"

            if role == "facilitator":
                if entry["speaker"] == "round_summary":
                    rnd = entry.get("round", "?")
                    parts.append(f"[FACILITATOR — Round {rnd} Summary]\n{snippet}")
                else:
                    parts.append(f"[FACILITATOR — Opening]\n{snippet}")
            elif role == "debater":
                rnd = entry.get("round", "?")
                parts.append(f"[{speaker} — Round {rnd}]\n{snippet}")
            elif role == "user_input":
                parts.append(f"[MODERATOR INPUT]\n{text}")

        return "\n\n---\n\n".join(parts)

    def _round_transcript(self, round_num: int) -> str:
        """Returns only the debater entries for a specific round."""
        parts = []
        for entry in self.history:
            if entry.get("round") == round_num and entry["role"] == "debater":
                parts.append(f"[{entry['speaker'].upper()}]\n{entry['text']}")
        return "\n\n---\n\n".join(parts) if parts else "No entries for this round."

    def _full_transcript(self) -> str:
        """Complete, untruncated transcript for the final summary prompt."""
        parts = []
        for entry in self.history:
            role = entry["role"]
            speaker = entry["speaker"].upper()
            text = entry["text"]

            if role == "facilitator":
                if entry["speaker"] == "round_summary":
                    rnd = entry.get("round", "?")
                    parts.append(f"=== FACILITATOR (Round {rnd} Summary) ===\n{text}")
                else:
                    parts.append(f"=== FACILITATOR (Introduction) ===\n{text}")
            elif role == "debater":
                rnd = entry.get("round", "?")
                parts.append(f"=== {speaker} (Round {rnd}) ===\n{text}")
            elif role == "user_input":
                parts.append(f"=== MODERATOR INPUT ===\n{text}")

        return "\n\n".join(parts)
