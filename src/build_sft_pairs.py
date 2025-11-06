# src/build_sft_pairs.py
import argparse, os, glob, json, math
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from src.agent_io import validate_action  # uses your existing clamp/validation

V4_REP_MIN, V4_REP_MAX = 0.20, 0.70
LR_SCALE_MIN, LR_SCALE_MAX = 0.8, 1.2

# ---- Edge-classification thresholds (tune as needed) ----
EDGE_FORGET_THR = 0.003   # mean forgetting high
EDGE_DIV_THR    = 0.003   # model divergence high
EDGE_DACC_THR   = -0.003  # accuracy drop vs previous round

@dataclass
class ClientSnap:
    id: int
    vloss: float | None
    vacc: float | None
    new_batch_size: int
    last_lr: float
    replay_capacity: int
    last_replay_ratio: float
    last_ewc_lambda: float

def _compact_state_for_sft(state: Dict[str, Any]) -> Dict[str, Any]:
    g = state["global"]
    keep_g = {
        "acc": float(g.get("acc", 0.0)),
        "ema_loss": float(g.get("ema_loss", g.get("loss", 0.0))),
        "forget_mean": float(g.get("forget_mean", 0.0)),
        "divergence": float(g.get("divergence", 0.0)),
    }
    comp_clients = []
    for c in state["clients"]:
        vloss = c.get("vloss", None)
        if isinstance(vloss, float) and math.isnan(vloss):
            vloss = None
        comp_clients.append({
            "id": int(c["id"]),
            "vloss": None if vloss is None else float(vloss),
            "vacc": float(c.get("vacc", 0.0)),
            "new_batch_size": int(c.get("new_batch_size", 0)),
            "last_lr": float(c.get("last_lr", 0.0)),
        })
    return {"global": keep_g, "clients": comp_clients}

def _is_edge(curr_state: Dict[str, Any], prev_state: Dict[str, Any] | None) -> bool:
    g = curr_state.get("global", {})
    forget_mean = float(g.get("forget_mean", 0.0))
    divergence  = float(g.get("divergence", 0.0))
    acc_curr    = float(g.get("acc", 0.0))

    acc_delta = 0.0
    if prev_state is not None:
        acc_prev = float(prev_state.get("global", {}).get("acc", acc_curr))
        acc_delta = acc_curr - acc_prev

    return (
        forget_mean >= EDGE_FORGET_THR
        or divergence >= EDGE_DIV_THR
        or acc_delta <= EDGE_DACC_THR
    )

def _ranked_lr_scales(vlosses: List[float | None]) -> List[float]:
    """
    Higher validation loss → needs more learning rate (toward 1.2).
    If all None, return 1.0s.
    """
    clean = [np.inf if (v is None or np.isnan(v)) else float(v) for v in vlosses]
    if all(v == np.inf for v in clean):
        return [1.0 for _ in clean]

    vmin, vmax = min(clean), max(clean)
    span = max(1e-8, vmax - vmin)
    # normalize: high vloss => rank ~1.0, low vloss => rank ~0.0
    ranks = [(v - vmin) / span for v in clean]
    # map rank -> [0.8, 1.2]
    return [LR_SCALE_MIN + (1.0 - r) * (LR_SCALE_MAX - LR_SCALE_MIN) for r in ranks]

def _teacher_action_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic teacher:
      - Replay up if forgetting/divergence high; down if stable.
      - Per-client lr_scale from vloss ranking.
    """
    g = state["global"]
    F_t = float(g.get("forget_mean", 0.0))
    div = float(g.get("divergence", 0.0))

    # base replay
    rep = 0.50
    if F_t > 0.05 or div > 0.10:
        rep += 0.10
    elif F_t < 0.01 and div < 0.05:
        rep -= 0.05
    rep = max(V4_REP_MIN, min(V4_REP_MAX, rep))

    vlosses = []
    ids = []
    for c in state["clients"]:
        v = c.get("vloss", None)
        if isinstance(v, float) and np.isnan(v):
            v = None
        vlosses.append(v)
        ids.append(int(c["id"]))
    scales = _ranked_lr_scales(vlosses)

    action = {
        "client_selection_k": len(ids),
        "aggregation": {"method": "FedAvg"},
        "client_params": [
            {"id": cid, "replay_ratio": float(rep), "lr_scale": float(sc), "ewc_lambda": 0.0}
            for cid, sc in zip(ids, scales)
        ],
    }
    # Validate/clamp & tag for traceability (teacher)
    return validate_action(action, n_clients=len(ids), policy_source="TeacherV0")

def _is_edge_state(state: Dict[str, Any]) -> bool:
    F_t = float(state["global"].get("forget_mean", 0.0))
    div = float(state["global"].get("divergence", 0.0))
    d_bad = (F_t > 0.05) or (div > 0.10)
    return d_bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", type=str, default="runs/*", help="Where to scan for state_round_*.json")
    ap.add_argument("--edge_ratio", type=float, default=0.5, help="Target fraction of 'edge' pairs")
    ap.add_argument("--max_pairs", type=int, default=5000)
    ap.add_argument("--out_jsonl", type=str, default="data/sft_pairs.jsonl")
    ap.add_argument("--out_stats", type=str, default="data/sft_stats.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    state_paths = sorted(glob.glob(os.path.join(args.runs_glob, "state_round_*.json")))
    pairs_edge, pairs_normal = [], []

    for sp in state_paths:
        try:
            with open(sp, "r") as f:
                state = json.load(f)
                # figure out previous round path in the SAME run folder
                dir_run = os.path.dirname(sp)                     # runs/<run_id>
                fname = os.path.basename(sp)                      # state_round_<r>.json
                r_idx = int(fname.split("_")[-1].split(".")[0])   # <r> as int

                prev_state = None
                if r_idx > 0:
                    prev_path = os.path.join(dir_run, f"state_round_{r_idx-1}.json")
                    if os.path.exists(prev_path):
                        with open(prev_path, "r") as pf:
                            prev_state = json.load(pf)

                bucket = "edge" if _is_edge(state, prev_state) else "normal"
        except Exception:
            continue

        comp = _compact_state_for_sft(state)

        # 1) Build teacher action from current state
        action = _teacher_action_from_state(state)

        # 2) validate & clamp once
        n_clients = len(state["clients"])
        action = validate_action(action, n_clients=n_clients, policy_source="TeacherV0")

        # 3) Create the training pair
        pair = {
            "state_small": comp,
            "action": action,
            "bucket": bucket,
            "meta": {
                "run_id": os.path.basename(dir_run),
                "round": r_idx,
                "source": "TeacherV0"
            }
        }

        # 4) Append to the correct bucket list
        (pairs_edge if bucket == "edge" else pairs_normal).append(pair)

    # balance by edge_ratio
    n_total = min(args.max_pairs, len(pairs_edge) + len(pairs_normal))
    n_edge = min(int(round(args.edge_ratio * n_total)), len(pairs_edge))
    n_norm = min(n_total - n_edge, len(pairs_normal))

    # if one bucket lacks, top up from the other
    if n_edge < int(round(args.edge_ratio * n_total)):
        need = int(round(args.edge_ratio * n_total)) - n_edge
        n_norm = min(n_norm + need, len(pairs_normal))
    elif n_norm < (n_total - n_edge):
        need = (n_total - n_edge) - n_norm
        n_edge = min(n_edge + need, len(pairs_edge))

    sel = pairs_edge[:n_edge] + pairs_normal[:n_norm]

    with open(args.out_jsonl, "w") as f:
        for row in sel:
            f.write(json.dumps(row) + "\n")

    stats = {
        "total_pairs": len(sel),
        "by_bucket": {"edge": n_edge, "normal": n_norm},
        "edge_pairs_original": len(pairs_edge),
        "target_edge_ratio": args.edge_ratio,
    }
    with open(args.out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {len(sel)} pairs → {args.out_jsonl}")
    print(f"Wrote stats → {args.out_stats}")

if __name__ == "__main__":
    main()