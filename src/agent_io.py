"""
agent_io.py
------------
Handles State/Action JSON I/O, schema validation, and safe clamping
for controllers (Mock, Controller-v4, API, LLM, etc.).
"""

import json, os
from typing import Any, Dict, List, Optional

# === Global safe bounds (matches ROADMAP) ===
BOUNDS = {
    "replay_ratio": (0.20, 0.70),
    "lr_scale": (0.80, 1.20),
    "ewc_lambda": (0.0, 1000.0),
    "client_selection_k": (2, 4),
    "fedprox_mu": (0.0, 0.1),
}

# --- Basic file I/O ---
def save_json(data: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

# --- Helper: clamp to [min,max] ---
def clamp(val: Any, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(val)))
    except Exception:
        # fallback to a safe default (lower bound)
        return lo

# === Core: validate & clamp Action JSON ===
def validate_and_clamp_action(
    action: Dict[str, Any],
    n_clients: int,
    policy_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ensure all fields are within safe bounds and fill defaults if missing.
    If policy_source is provided, tag it; otherwise omit the field.
    """
    act = dict(action) if isinstance(action, dict) else {}

    # ---- Top-level fields ----
    k_lo, k_hi_cfg = BOUNDS["client_selection_k"]
    k_hi = min(k_hi_cfg, max(1, int(n_clients)))  # can’t exceed actual clients
    k_in = act.get("client_selection_k", 4)
    act["client_selection_k"] = int(clamp(k_in, k_lo, k_hi))

    agg = act.get("aggregation", {"method": "FedAvg"}) or {"method": "FedAvg"}
    method = agg.get("method", "FedAvg")
    if method not in {"FedAvg", "FedProx"}:
        method = "FedAvg"
    if method == "FedProx":
        mu_lo, mu_hi = BOUNDS["fedprox_mu"]
        mu = clamp(agg.get("mu", 0.0), mu_lo, mu_hi)
        act["aggregation"] = {"method": "FedProx", "mu": mu}
    else:
        act["aggregation"] = {"method": "FedAvg"}

    # ---- Per-client params ----
    rr_lo, rr_hi = BOUNDS["replay_ratio"]
    lr_lo, lr_hi = BOUNDS["lr_scale"]
    ew_lo, ew_hi = BOUNDS["ewc_lambda"]

    safe_clients: List[Dict[str, Any]] = []
    for c in act.get("client_params", []):
        try:
            cid = int(c.get("id"))
        except Exception:
            # if id missing/bad, assign sequentially
            cid = len(safe_clients)

        rr  = clamp(c.get("replay_ratio", 0.50), rr_lo, rr_hi)
        lrs = clamp(c.get("lr_scale", 1.00),    lr_lo, lr_hi)
        ewc = clamp(c.get("ewc_lambda", 0.0),   ew_lo, ew_hi)

        safe_clients.append({
            "id": cid,
            "replay_ratio": rr,
            "lr_scale": lrs,
            "ewc_lambda": ewc,
        })

    # only keep as many as selected
    act["client_params"] = safe_clients[: act["client_selection_k"]]

    # ---- Tag for provenance (optional here; always added when writing) ----
    if policy_source is not None:
        act["policy_source"] = policy_source

    return act

# === Adapters expected by run_llm_fcl_controller ===
def validate_action(action: Dict[str, Any], k_max: int) -> Dict[str, Any]:
    """
    Compatibility wrapper used by the training script.
    k_max should be the number of available clients (or an upper bound).
    """
    return validate_and_clamp_action(action, n_clients=int(k_max), policy_source=None)

def write_state_json(state: Dict[str, Any], run_dir: str, round_id: int) -> str:
    """
    Persist the controller State snapshot for round_id. Returns the written filepath.
    """
    path = os.path.join(run_dir, f"state_round_{int(round_id)}.json")
    return save_json(state, path)

def write_action_json(
    action: Dict[str, Any],
    run_dir: str,
    round_id: int,
    policy_source: str = "Mock",
) -> str:
    """
    Persist the Action with a policy_source tag. Returns the written filepath.
    Assumes 'action' has already been validated/clamped; if not, upstream should call validate_action.
    """
    out = dict(action) if isinstance(action, dict) else {}
    out["policy_source"] = policy_source
    path = os.path.join(run_dir, f"action_round_{int(round_id)}.json")
    return save_json(out, path)

# === Quick self-test ===
if __name__ == "__main__":
    bad_action = {
        "client_selection_k": 10,
        "aggregation": {"method": "XYZ"},
        "client_params": [
            {"id": 0, "replay_ratio": 0.95, "lr_scale": 2.0, "ewc_lambda": -5}
        ],
    }
    fixed = validate_and_clamp_action(bad_action, n_clients=4, policy_source="Mock")
    print("Original:", json.dumps(bad_action, indent=2))
    print("→ Clamped:", json.dumps(fixed, indent=2))