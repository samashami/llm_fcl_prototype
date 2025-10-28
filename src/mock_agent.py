# src/mock_agent.py
from typing import Dict, Any, List
from src.agent_io import validate_and_clamp_action

def decide_action(state: Dict[str, Any], n_clients: int) -> Dict[str, Any]:
    """Simple, safe rules over state -> action (per our ROADMAP bounds)."""
    g = state.get("global", {})
    forget_mean = float(g.get("forget_mean", 0.0))
    divergence  = float(g.get("divergence", 0.0))

    # base defaults
    client_params: List[Dict[str, Any]] = []
    for c in state.get("clients", []):
        client_params.append({
            "id": int(c["id"]),
            "replay_ratio": 0.50,   # default
            "lr_scale": 1.00,       # default
            "ewc_lambda": 0.0       # keep 0.0 until EWC is wired
        })

    # simple rules
    if forget_mean > 0.03 or divergence > 0.10:
        for p in client_params:
            p["replay_ratio"] = 0.60  # push up a bit

    action = {
        "client_selection_k": min(n_clients, 4),
        "aggregation": {"method": "FedAvg"},
        "client_params": client_params,
    }
    # clamp & tag for provenance here or at write time
    return validate_and_clamp_action(action, n_clients=n_clients, policy_source="Mock")