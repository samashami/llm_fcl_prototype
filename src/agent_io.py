"""
agent_io.py
------------
Handles State/Action JSON I/O, schema validation, and safe clamping
for controllers (Mock, Controller-v4, API, LLM, etc.).
"""

import json, os

# === Global safe bounds (matches ROADMAP) ===
BOUNDS = {
    "replay_ratio": (0.20, 0.70),
    "lr_scale": (0.80, 1.20),
    "ewc_lambda": (0.0, 1000.0),
    "client_selection_k": (2, 4),
    "fedprox_mu": (0.0, 0.1)
}

# === Basic file I/O ===
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# === Helper: clamp to [min,max] ===
def clamp(val, lo, hi):
    try:
        return max(lo, min(hi, float(val)))
    except Exception:
        return lo  # fallback to lower bound if invalid

# === Validate & clamp Action JSON ===
def validate_and_clamp_action(action, n_clients, policy_source="Mock"):
    """Ensure all fields are within safe bounds and fill defaults if missing."""
    act = dict(action)  # shallow copy

    # ---- Top-level fields ----
    k_lo, k_hi = BOUNDS["client_selection_k"]
    act["client_selection_k"] = int(
        clamp(act.get("client_selection_k", 4), k_lo, min(k_hi, n_clients))
    )

    agg = act.get("aggregation", {"method": "FedAvg"})
    if agg.get("method") not in {"FedAvg", "FedProx"}:
        agg["method"] = "FedAvg"
    if agg["method"] == "FedProx":
        mu_lo, mu_hi = BOUNDS["fedprox_mu"]
        agg["mu"] = clamp(agg.get("mu", 0.0), mu_lo, mu_hi)
    act["aggregation"] = agg

    # ---- Per-client params ----
    clamped_clients = []
    for c in act.get("client_params", []):
        cid = c.get("id", len(clamped_clients))
        rr_lo, rr_hi = BOUNDS["replay_ratio"]
        lr_lo, lr_hi = BOUNDS["lr_scale"]
        ew_lo, ew_hi = BOUNDS["ewc_lambda"]

        clamped_clients.append({
            "id": cid,
            "replay_ratio": clamp(c.get("replay_ratio", 0.50), rr_lo, rr_hi),
            "lr_scale": clamp(c.get("lr_scale", 1.00), lr_lo, lr_hi),
            "ewc_lambda": clamp(c.get("ewc_lambda", 0.0), ew_lo, ew_hi)
        })
    act["client_params"] = clamped_clients[:act["client_selection_k"]]

    # ---- Tag for provenance ----
    act["policy_source"] = policy_source
    return act

# === Quick self-test ===
if __name__ == "__main__":
    bad_action = {
        "client_selection_k": 10,
        "aggregation": {"method": "XYZ"},
        "client_params": [
            {"id": 0, "replay_ratio": 0.95, "lr_scale": 2.0, "ewc_lambda": -5}
        ]
    }
    fixed = validate_and_clamp_action(bad_action, n_clients=4, policy_source="Mock")
    print("Original:", json.dumps(bad_action, indent=2))
    print("→ Clamped:", json.dumps(fixed, indent=2))