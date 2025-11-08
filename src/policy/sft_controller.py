# src/policy/sft_controller.py
from typing import Any, Dict, Callable, Optional

class SFTPolicyController:
    """
    Thin wrapper around an action_fn(state) -> action dict.
    - Stable .name for logging
    - Optional validator/clamp
    - Ensures 'policy_source' is set
    """
    def __init__(
        self,
        action_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        name: str = "SFT_v0",
        hard_validate: bool = False,
        validator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.name = name
        self._action_fn = action_fn
        self._validator = validator
        self._hard = hard_validate

    def get_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        action = self._action_fn(state) or {}
        if self._validator is not None:
            try:
                action = self._validator(action)
            except Exception:
                if self._hard:
                    raise
                # soft mode: keep original action on validator error
                pass
        if "policy_source" not in action:
            action["policy_source"] = self.name
        return action