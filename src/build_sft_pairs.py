#!/usr/bin/env python3
"""
Build SFT pairs from runs/* {state_round_*.json, action_round_*.json}.
- Pairs each state with its action (same round, same run dir)
- Sanitizes NaN/Inf -> 0.0
- Adds simple deltas (delta_acc) and bytes_cum if missing
- Flags "edge cases" and optionally oversamples them to target ratio
- Writes:
  - data/sft_pairs.jsonl   (one JSON per line)
  - data/sft_stats.json    (counts, edge ratios)
"""

import argparse, json, math, os, glob, re, random
from collections import defaultdict
# run_llm_fcl_controller.py
from src._bootstrap_env import *  # sets TOKENIZERS_PARALLELISM=false early

def _is_num(x):
    return isinstance(x, (int, float))

def _sanitize_numbers(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numbers(v) for v in obj]
    if _is_num(obj):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    return obj

def _load_json(path):
    with open(path, "r") as f:
        text = f.read()
    # Python's json handles NaN by default in loads (maps to float('nan'))
    obj = json.loads(text)
    return _sanitize_numbers(obj)

def _round_of(fname):
    # .../state_round_3.json -> 3
    m = re.search(r"round_(\d+)\.json$", fname)
    return int(m.group(1)) if m else None

def _edge_case_flags(state, prev_state, thr_forget=0.05, thr_div=0.15, thr_dacc=-0.02):
    # Defaults if prev missing
    acc    = float(state.get("global", {}).get("acc", 0.0))
    prev_a = float(prev_state.get("global", {}).get("acc", acc)) if prev_state else acc
    dacc   = acc - prev_a
    forget_mean = float(state.get("global", {}).get("forget_mean", 0.0))
    divergence  = float(state.get("global", {}).get("divergence", 0.0))

    is_forget = forget_mean > thr_forget
    is_div    = divergence  > thr_div
    is_drop   = dacc        < thr_dacc

    return {
        "delta_acc": dacc,
        "is_forget_spike": bool(is_forget),
        "is_divergent": bool(is_div),
        "is_acc_drop": bool(is_drop),
        "edge_case": bool(is_forget or is_div or is_drop),
    }

def _collect_run_pairs(run_dir):
    """Return list of dicts with {round, state_path, action_path} for that run_dir."""
    states  = sorted(glob.glob(os.path.join(run_dir, "state_round_*.json")))
    actions = sorted(glob.glob(os.path.join(run_dir, "action_round_*.json")))
    by_r_state  = { _round_of(p): p for p in states  if _round_of(p) is not None }
    by_r_action = { _round_of(p): p for p in actions if _round_of(p) is not None }

    rounds = sorted(set(by_r_state.keys()) & set(by_r_action.keys()))
    out = []
    for r in rounds:
        out.append({
            "round": r,
            "state_path": by_r_state[r],
            "action_path": by_r_action[r],
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default="runs/*", help="Glob to run directories")
    ap.add_argument("--out_pairs", default="data/sft_pairs.jsonl")
    ap.add_argument("--out_stats", default="data/sft_stats.json")
    ap.add_argument("--target_edge_ratio", type=float, default=0.5,
                    help="Oversample edge cases up to this fraction of the dataset (0..1)")
    ap.add_argument("--thr_forget", type=float, default=0.05)
    ap.add_argument("--thr_div", type=float, default=0.15)
    ap.add_argument("--thr_dacc", type=float, default=-0.02)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_pairs), exist_ok=True)

    run_dirs = sorted([p for p in glob.glob(args.runs_glob) if os.path.isdir(p)])
    random.seed(args.seed)

    all_rows = []
    by_source_counts = defaultdict(int)
    edge_count = 0

    for rd in run_dirs:
        pairs = _collect_run_pairs(rd)
        # Load all states once to allow prev lookup
        state_by_round = {}
        for p in pairs:
            s = _load_json(p["state_path"])
            # add bytes_cum if missing
            g = s.setdefault("global", {})
            if "bytes_cum" not in g:
                g["bytes_cum"] = float(g.get("bytes_last_round", 0.0))
            state_by_round[p["round"]] = s

        # Accumulate bytes_cum realistically
        cum = 0.0
        for r in sorted(state_by_round.keys()):
            g = state_by_round[r]["global"]
            blr = float(g.get("bytes_last_round", 0.0))
            cum += blr
            g["bytes_cum"] = float(cum)

        for p in pairs:
            r = p["round"]
            state = state_by_round[r]
            action = _load_json(p["action_path"])
            source = action.get("policy_source", "Unknown")

            prev_state = state_by_round.get(r-1)
            flags = _edge_case_flags(state, prev_state,
                                     thr_forget=args.thr_forget,
                                     thr_div=args.thr_div,
                                     thr_dacc=args.thr_dacc)

            row = {
                "run_id": os.path.basename(rd),
                "round": r,
                "policy_source": source,
                "state": state,
                "action": action,
                "delta_acc": flags["delta_acc"],
                "edge_case": flags["edge_case"],
                "edge_flags": {
                    "forget_mean_gt_thr": flags["is_forget_spike"],
                    "divergence_gt_thr": flags["is_divergent"],
                    "acc_drop_lt_thr": flags["is_acc_drop"],
                },
            }
            all_rows.append(row)
            by_source_counts[source] += 1
            if flags["edge_case"]:
                edge_count += 1

    total = len(all_rows)
    if total == 0:
        print("No pairs found. Did you run any experiments?")
        return

    # Oversample edge cases up to target ratio
    cur_edge_ratio = edge_count / max(1, total)
    target_ratio   = max(0.0, min(1.0, args.target_edge_ratio))
    print(f"[Info] Found {edge_count}/{total} edge pairs (ratio={cur_edge_ratio:.3f}). Target={target_ratio:.3f}")

    if cur_edge_ratio < target_ratio and edge_count > 0:
        need_total = int(round((total - edge_count) / (1 - target_ratio)))
        need_edge  = max(0, need_total - edge_count)
        edge_rows  = [r for r in all_rows if r["edge_case"]]
        dup_rows   = []
        while len(dup_rows) < need_edge:
            dup_rows.append(random.choice(edge_rows))
        all_rows = all_rows + dup_rows
        print(f"[Info] Oversampled {len(dup_rows)} edge rows to reach ~{target_ratio:.2f} ratio "
              f"(new total={len(all_rows)}).")

    if args.shuffle:
        random.shuffle(all_rows)

    # Write pairs (JSONL)
    with open(args.out_pairs, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Stats
    out_stats = {
        "total_pairs": len(all_rows),
        "by_policy_source": dict(by_source_counts),
        "edge_pairs_original": edge_count,
        "target_edge_ratio": target_ratio,
    }
    with open(args.out_stats, "w") as f:
        json.dump(out_stats, f, indent=2)

    print(f"Wrote {args.out_pairs} ({len(all_rows)} lines)")
    print(f"Wrote {args.out_stats}")
    # Print a peek
    with open(args.out_pairs, "r") as f:
        for i, line in enumerate(f):
            if i >= 3: break
            print(line.strip())

if __name__ == "__main__":
    main()