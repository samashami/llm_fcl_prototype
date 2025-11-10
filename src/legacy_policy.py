# src/legacy_policy.py
from typing import Dict, Any

class Policy:
    """
    Rule-based placeholder for an LLM-guided controller.
    It reads round metrics and returns training knobs for the next round.
    """
    def __init__(self):
        self._lr = 0.01
        self._replay = 0.20

    def decide(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        acc = summary.get("accuracy_global", 0.0)
        acc_delta = summary.get("acc_delta", 0.0)
        worst_forgetting = max(summary.get("forgetting_per_class", [0.0]) or [0.0])

        note = []

        # If forgetting spikes, increase replay a bit (cap 0.6)
        if worst_forgetting > 0.05:
            self._replay = min(0.60, self._replay + 0.05)
            note.append("↑forgetting -> ↑replay")

        # If accuracy stalled, lower LR slightly to stabilize; else gently raise
        if acc_delta < 0.002:
            self._lr = max(0.001, self._lr * 0.8)
            note.append("stalled -> ↓lr")
        else:
            self._lr = min(0.02, self._lr * 1.2)
            note.append("improving -> ↑lr")

        return {
            "lr": float(self._lr),
            "replay_ratio": float(round(self._replay, 3)),
            "notes": "; ".join(note) or "default"
        }
