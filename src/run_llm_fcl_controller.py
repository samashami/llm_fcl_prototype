# src/run_llm_fcl_controller.py

import argparse, time, copy, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import datasets, transforms
import pandas as pd

from src.model import build_resnet18
from src.fl import Client, Server
from src.strategies.replay import ReplayBuffer
from src.policy import Policy
from src.agent_io import save_json, validate_and_clamp_action
from src.agent_io import write_state_json, write_action_json, validate_action
from src.mock_agent import decide_action as mock_decide_action
import os, json

# ---------------------------
# Controller v4 hyperparams
# ---------------------------
V4_LR_MIN, V4_LR_MAX = 1e-4, 2e-3
V4_REP_MIN, V4_REP_MAX = 0.20, 0.70
V4_DEADBAND = 0.003
V4_REP_STEP_HIGH = 0.10
V4_REP_STEP_LOW  = 0.05
V4_FORGET_THR    = 0.05
V4_DIV_THR       = 0.10
V4_EMA_ALPHA     = 0.30
V4_LR_BOOST      = 1.35
V4_LR_COOLDOWN   = 1.50
V4_CLIENT_LR_MIN, V4_CLIENT_LR_MAX = 0.8, 1.2
V4_ROLLBACK_THR  = 0.02         # absolute acc drop
V4_WARMUP_ROUNDS = 2

# ---------------------------
# Seeding helpers
# ---------------------------
GLOBAL_SEED = 42
def seed_worker(worker_id: int):
    import numpy as _np, random as _random
    _np.random.seed(GLOBAL_SEED + worker_id)
    _random.seed(GLOBAL_SEED + worker_id)

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Eval helpers
# ---------------------------
def evaluate(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    n_classes = 100
    hits = np.zeros(n_classes, dtype=np.int64)
    counts = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for c in range(n_classes):
                mask = (y == c)
                if mask.any():
                    counts[c] += mask.sum().item()
                    hits[c] += (pred[mask] == c).sum().item()
    acc = correct / max(1, total)
    per_class_recall = np.array(
        [(hits[c] / counts[c]) if counts[c] > 0 else 0.0 for c in range(n_classes)],
        dtype=np.float32,
    )
    return acc, per_class_recall

def evaluate_loss(model, device, loader):
    model.eval()
    crit = torch.nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(crit(logits, y).item()) * y.size(0)
            n += y.size(0)
    return total_loss / max(1, n)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset_per_client", type=int, default=-1, help="use -1 for all data")
    ap.add_argument("--use_policy", action="store_true", default=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--split_mode", choices=["equal"], default="equal")
    ap.add_argument("--val_size", type=int, default=5000)
    ap.add_argument("--cl_batches", type=int, default=7)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--optimizer", choices=["adam","sgd"], default="adam")
    ap.add_argument("--early_patience", type=int, default=5)
    ap.add_argument("--tag", type=str, default="controller_v4")
    ap.add_argument("--controller", choices=["v4", "mock", "fixed"], default="v4")
    args = ap.parse_args()

    set_seeds(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    global GLOBAL_SEED
    GLOBAL_SEED = args.seed

    g = torch.Generator()
    g.manual_seed(args.seed)

    # Transforms
    tf_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    tf_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # Data
    trainset_full = datasets.CIFAR100(root="./data", train=True,  download=True, transform=tf_train)
    testset       = datasets.CIFAR100(root="./data", train=False, download=True, transform=tf_test)

    total_train = len(trainset_full)  # 50_000
    val_size = args.val_size          # 5_000
    train_size = total_train - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=g
    )
    train_indices = np.array(train_subset.indices, dtype=np.int64)
    val_indices   = np.array(val_subset.indices, dtype=np.int64)

    valset = Subset(trainset_full, val_indices)
    val_loader = DataLoader(valset, batch_size=256, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"[Split] train={len(train_indices)} val={len(val_indices)} test={len(testset)}", flush=True)

    # Split among clients (equal-size)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(train_indices)
    sizes = [len(perm) // args.clients] * args.clients
    for i in range(len(perm) % args.clients):
        sizes[i] += 1
    splits, start = [], 0
    for s in sizes:
        splits.append(perm[start:start+s].tolist())
        start += s

    # Optional subsample per client
    if args.subset_per_client and args.subset_per_client > 0:
        splits = [idxs[:args.subset_per_client] for idxs in splits]

    for i, idxs in enumerate(splits):
        print(f"[Split] client {i}: {len(idxs)} images", flush=True)

    # Build CL schedule: initial ~0.466 + even splits
    def make_cl_batches(indices, num_batches=7, seed=42):
        rng_local = np.random.RandomState(seed)
        idx = np.array(indices, dtype=np.int64)
        rng_local.shuffle(idx)
        init = int(round(0.466 * len(idx)))
        init = max(1, min(len(idx) - (num_batches - 1), init))
        first = idx[:init]
        rem = idx[init:]
        if num_batches <= 1:
            return [idx.tolist()]
        per = len(rem) // (num_batches - 1)
        chunks = [rem[i*per:(i+1)*per] for i in range(num_batches - 2)]
        chunks.append(rem[(num_batches - 2)*per:])
        return [first.tolist()] + [c.tolist() for c in chunks]

    cl_schedule, cl_rows = [], []
    for cid, idxs in enumerate(splits):
        batches = make_cl_batches(idxs, num_batches=args.cl_batches, seed=args.seed + cid)
        cl_schedule.append(batches)
        sizes = [len(b) for b in batches]
        print(f"[CL] client {cid}: {sizes} (sum={sum(sizes)})", flush=True)
        for i, b in enumerate(batches, start=1):
            cl_rows.append({"run_id": "", "client": cid, "cl_batch": i, "size": len(b)})

    # Init clients
    clients = []
    for cid, idx in enumerate(splits):
        subset = Subset(trainset_full, idx)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        model = build_resnet18(100).to(device)
        if args.optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        else:
            opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        replay = ReplayBuffer(capacity=2000)
        clients.append(Client(cid, model, opt, loader, device=device, replay=replay,
                              val_loader=val_loader, early_patience=args.early_patience))

        if cid == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Trainable params: {trainable:,}/{total}", flush=True)

    print("✅ Client 0 model device:", next(clients[0].model.parameters()).device, flush=True)

    # Server / Policy
    server = Server(device=device)
    policy = Policy()

    # Initial global model + metrics
    global_model = server.average([c.model for c in clients])
    acc, per_class = evaluate(global_model, device, test_loader)
    best_recall = per_class.copy()
    forgetting = np.zeros_like(per_class)
    global_loss = evaluate_loss(global_model, device, test_loader)
    ema_loss = global_loss
    div_norm = 0.0

    print(f"[Round -1] acc={acc:.3f}", flush=True)

    # Local best/rollback tracking (no Server.save_state)
    best_state = copy.deepcopy(global_model.state_dict())
    best_global_acc = float(acc)
    best_hp = {"lr": args.lr, "replay_ratio": 0.50, "notes": "init (paper defaults)"}
    best_round = -1
    rollback_flag = False
    rollback_round = -1
    last_acc = acc
    last_hp = copy.deepcopy(best_hp)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    # Fill run_id into CL rows for traceability
    for row in cl_rows:
        row["run_id"] = run_id

    run_logs, round_logs = [], []

    io_root = os.path.join("runs", run_id)
    os.makedirs(io_root, exist_ok=True)

    def _build_state(round_id, acc_global, loss_global, ema_loss, forget_mean, forget_max, divergence, bytes_last_round, client_snapshots):
        return {
            "round_id": int(round_id),
            "global": {
                "acc": float(acc_global),
                "loss": float(loss_global),
                "ema_loss": float(ema_loss),
                "forget_mean": float(forget_mean),
                "forget_max": float(forget_max),
                "divergence": float(divergence),
                "bytes_last_round": int(bytes_last_round),
            },
            "clients": client_snapshots,  # list of dicts with vloss, vacc, last_lr, last_replay_ratio, last_ewc_lambda, etc.
        }

    def _safe_last_lr(c, fallback_lr):
        try:
            return float(c.optimizer.param_groups[0]["lr"])
        except Exception:
            return float(fallback_lr)
    # ---------------------------
    # Training rounds
    # ---------------------------
    bytes_last_round = 0  # carried into the next round's state
    bytes_cum = 0
    aulc_running = 0.0

    for r in range(args.rounds):
        acc_delta = float(acc - last_acc)

        # --- Build and write state JSON (once, at round start) ---
        client_snaps = []
        for c in clients:
            # robust last_lr: if optimizer exists use it, else fallback to last chosen HP or CLI LR
            if hasattr(c, "optimizer") and getattr(c.optimizer, "param_groups", None):
                _lr = float(c.optimizer.param_groups[0]["lr"])
            else:
                _lr = float(last_hp.get("lr", args.lr))

            # new batch size for THIS round for this client
            batches = cl_schedule[c.cid]
            nb = len(batches[r]) if r < len(batches) else len(batches[-1])

            client_snaps.append({
                "id": int(c.cid),
                "vloss": float(getattr(c, "_last_vloss", float("nan"))),
                "vacc": float(getattr(c, "_last_vacc", float("nan"))),
                "new_batch_size": int(nb),
                "replay_capacity": int(getattr(getattr(c, "replay", None), "capacity", 2000)),
                "last_lr": _safe_last_lr(c, hp["lr"] if 'hp' in locals() else args.lr),
                "last_replay_ratio": float(last_hp.get("replay_ratio", 0.50)),
                "last_ewc_lambda": float(getattr(c, "_last_ewc_lambda", 0.0)),
            })

        state = {
            "round_id": int(r),
            "global": {
                "acc": float(acc),
                "loss": float(global_loss),
                "ema_loss": float(ema_loss),
                "forget_mean": float(np.mean(forgetting)) if forgetting is not None else 0.0,
                "forget_max": float(np.max(forgetting)) if forgetting is not None else 0.0,
                "divergence": float(div_norm),
                "bytes_last_round": int(bytes_last_round if 'bytes_last_round' in locals() else 0),
                "bytes_cum": int(bytes_cum),
            },
            "clients": client_snaps,
        }
        write_state_json(io_root, r, state)

        # --- optional: mock controller decides an action for this round ---
        if args.controller == "mock":
            mock_action = mock_decide_action(state, n_clients=len(clients))

            # apply mock decision to this roundif args.controller == "v4":
            hp = {
                "lr": float(args.lr),
                "replay_ratio": float(mock_action["client_params"][0]["replay_ratio"]) if mock_action["client_params"] else 0.50,
                "notes": "mock agent",
            }

            # per-client LR scaling from action
            cid2scale = {p["id"]: float(p.get("lr_scale", 1.0)) for p in mock_action.get("client_params", [])}
            for c in clients:
                scale = cid2scale.get(int(c.cid), 1.0)
                for pg in c.optimizer.param_groups:
                    pg["lr"] = hp["lr"] * scale
                c._last_lr_scale = float(scale)

        # ---- POLICY DECISION (Controller v4) ----
        if args.controller == "v4":
            # rollback branch
            if rollback_flag:
                lr = best_hp["lr"]
                rep = best_hp["replay_ratio"]
                notes = [f"ROLLBACK(r{rollback_round}→best r{best_round})"]
                rollback_flag = False  # consume flag

            elif r < V4_WARMUP_ROUNDS:
                lr, rep = args.lr, 0.50
                notes = ["warmup (fixed defaults)"]

            else:
                # Base on last stable HP
                lr, rep = last_hp["lr"], last_hp["replay_ratio"]
                notes = ["policy_v4"]

                F_t = float(np.mean(forgetting))  # mean forgetting
                dacc = acc_delta
                L_ema = float(ema_loss)
                div = float(div_norm)

                # Deadband on accuracy
                if abs(dacc) < V4_DEADBAND:
                    notes.append(f"deadband(|dacc|<{V4_DEADBAND})")
                else:
                    # Replay scheduling by forgetting/divergence
                    if F_t > V4_FORGET_THR or div > V4_DIV_THR:
                        rep += V4_REP_STEP_HIGH
                        notes.append("replay↑ (forget/div high)")
                    else:
                        rep -= V4_REP_STEP_LOW
                        notes.append("replay↓ (forget low)")

                    # LR by stability/convergence
                    if dacc < -V4_DEADBAND:
                        lr /= V4_LR_COOLDOWN
                        notes.append("lr↓ (dacc<0)")
                    elif dacc > V4_DEADBAND and L_ema > 1.5:
                        lr *= V4_LR_BOOST
                        notes.append("lr↑ (loss high & improving)")

            # Clamp
            lr  = max(V4_LR_MIN,  min(V4_LR_MAX,  lr))
            rep = max(V4_REP_MIN, min(V4_REP_MAX, rep))
            notes.append(f"clamped(lr∈[{V4_LR_MIN},{V4_LR_MAX}], rep∈[{V4_REP_MIN:.2f},{V4_REP_MAX:.2f}])")
            hp = {"lr": lr, "replay_ratio": rep, "notes": " | ".join(notes)}
        else:
            hp = {"lr": args.lr, "replay_ratio": 0.50, "notes": "fixed (paper CL default)"}

        # Log policy line
        F_t_print = float(np.mean(forgetting)) if forgetting is not None else 0.0
        print(
            f"[Policy r={r}] acc={acc:.3f} dacc={acc_delta:+.3f} F_t={F_t_print:.3f} Div={div_norm:.3f} "
            f"-> lr={hp['lr']:.5f}, replay={hp['replay_ratio']:.2f} ({hp['notes']})",
            flush=True,
        )

        # Remember chosen HP
        last_hp = {"lr": hp["lr"], "replay_ratio": hp["replay_ratio"], "notes": hp["notes"]}

        # ---- Broadcast global and set per-client LR scaling (inverted by loss rank) ----
        vlosses = []
        for c in clients:
            v = getattr(c, "_last_vloss", None)
            vlosses.append(float(v) if v is not None and not np.isnan(v) else float(global_loss))
        vl_min, vl_max = float(np.min(vlosses)), float(np.max(vlosses))
        rng_v = max(1e-8, vl_max - vl_min)

        for i, c in enumerate(clients):
            c.load_state_from(global_model)
            rank = (vlosses[i] - vl_min) / rng_v           # 0..1 (higher = worse loss)
            scale = V4_CLIENT_LR_MIN + (1.0 - rank) * (V4_CLIENT_LR_MAX - V4_CLIENT_LR_MIN)
            scale = max(V4_CLIENT_LR_MIN, min(V4_CLIENT_LR_MAX, scale))
            for pg in c.optimizer.param_groups:
                pg["lr"] = hp["lr"] * float(scale)
            c._last_lr_scale = float(scale)

        # --- build canonical Action from the chosen hp + client scales ---
        action = {
            "client_selection_k": len(clients),
            "aggregation": {"method": "FedAvg"},
            "client_params": [
                {
                    "id": int(c.cid),
                    "replay_ratio": float(hp["replay_ratio"]),
                    "lr_scale": float(getattr(c, "_last_lr_scale", 1.0)),
                    "ewc_lambda": float(getattr(c, "_last_ewc_lambda", 0.0)),
                }
                for c in clients
            ],
        }

        # tag by controller type
        source = (
            "Mock" if args.controller == "mock"
            else "ControllerV4" if args.controller == "v4"
            else "Fixed"
        )

        action = validate_action(action, n_clients=len(clients), policy_source=source)
        write_action_json(io_root, r, action, policy_source=source)

        # ---- Local training per client (once) ----
        for c in clients:
            batches = cl_schedule[c.cid]
            if r < len(batches):
                batch_indices = batches[r]
                batch_id = r
            else:
                batch_indices = batches[-1]
                batch_id = len(batches) - 1

            c.loader = DataLoader(
                Subset(trainset_full, batch_indices),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

            print(f"[Round {r}] client {c.cid}: CL batch {batch_id+1}/{len(batches)} "
                  f"(new={len(batch_indices)}; replay≈{hp['replay_ratio']:.2f}, LR_scale={c._last_lr_scale:.2f})",
                  flush=True)

            for e in range(args.epochs):
                avg_loss, epoch_acc, stop = c.train_one_epoch(
                    replay_ratio=hp["replay_ratio"],
                    epoch=e,
                    total_epochs=args.epochs,
                    log_interval=args.log_interval,
                )
                run_logs.append({
                    "run_id": run_id, "tag": args.tag, "round": r, "client": c.cid,
                    "epoch": e + 1,
                    "lr": float(c.optimizer.param_groups[0]["lr"]),
                    "replay_ratio": float(hp["replay_ratio"]),
                    "cl_batch": batch_id + 1,
                    "cl_batch_size": len(batch_indices),
                    "train_loss": float(avg_loss),
                    "train_acc": float(epoch_acc),
                    "val_loss": float(getattr(c, "_last_vloss", float("nan"))),
                    "val_acc": float(getattr(c, "_last_vacc", float("nan"))),
                })
                if stop:
                    print(f"[Client {c.cid}] Early stopping (patience {c.early_patience})", flush=True)
                    break

        # ---- Divergence (before FedAvg) ----
        with torch.no_grad():
            def flat_params(m: torch.nn.Module):
                return torch.cat([p.detach().float().view(-1).to(device) for p in m.parameters()])
            g_flat = flat_params(global_model)
            dists = []
            for c in clients:
                c_flat = flat_params(c.model)
                dists.append(torch.norm(c_flat - g_flat, p=2).item())
            if len(dists) > 1:
                div_norm = float(np.std(dists) / (np.median(dists) + 1e-8))
            else:
                div_norm = 0.0

        # ---- Aggregate & evaluate ----
        global_model = server.average([c.model for c in clients])
        last_acc = float(acc)
        acc, per_class = evaluate(global_model, device, test_loader)
        # running mean AULC up to round r
        aulc_running = ((aulc_running * r) + float(acc)) / max(1, (r + 1))

        # ---- Rollback check ----
        if acc < best_global_acc - V4_ROLLBACK_THR:
            # revert to best weights locally
            global_model.load_state_dict(best_state)
            acc, per_class = evaluate(global_model, device, test_loader)
            forgetting = np.maximum(0.0, best_recall - per_class)
            print(
                f"[🔥 ROLLBACK r{r}] drop detected. Reverted to best (r{best_round}) acc={best_global_acc:.3f}",
                flush=True,
            )
            rollback_flag = True
            rollback_round = r
        else:
            rollback_flag = False

        # ---- Update best state ----
        if acc > best_global_acc:
            best_global_acc = float(acc)
            best_state = copy.deepcopy(global_model.state_dict())
            best_hp = copy.deepcopy(hp)
            best_round = r

        # ---- Update loss/EMA/forgetting ----
        global_loss = evaluate_loss(global_model, device, test_loader)
        ema_loss = V4_EMA_ALPHA * global_loss + (1.0 - V4_EMA_ALPHA) * ema_loss
        forgetting = np.maximum(0.0, best_recall - per_class)
        best_recall = np.maximum(best_recall, per_class)

        # ---- Comm bytes for this round (used next round) ----
        model_size_bytes = sum(p.numel() for p in global_model.parameters()) * 4  # float32
        bytes_last_round = model_size_bytes * 2 * len(clients)  # up + down
        bytes_cum += int(bytes_last_round)

        # ---- Round summary log ----
        round_logs.append({
            "run_id": run_id, "tag": args.tag, "round": r,
            "global_acc": float(acc),
            "lr": float(hp["lr"]), "replay_ratio": float(hp["replay_ratio"]),
            "notes": hp.get("notes", ""),
            "global_loss": float(global_loss), "ema_loss": float(ema_loss),
            "forget_mean": float(np.mean(forgetting)), "divergence": float(div_norm),
            "best_acc_so_far": float(best_global_acc), "was_rollback": bool(rollback_flag),
            "comm_bytes_round": int(bytes_last_round),
            "aulc_running": float(aulc_running),
        })
        print(f"[Round {r}] acc={acc:.3f} (best={best_global_acc:.3f})", flush=True)

    # ---------------------------
    # Write CSVs
    # ---------------------------
    pd.DataFrame(run_logs).to_csv(f"fcl_run_results_{run_id}_{args.tag}.csv", index=False)
    pd.DataFrame(round_logs).to_csv(f"fcl_run_summary_{run_id}_{args.tag}.csv", index=False)
    pd.DataFrame(cl_rows).to_csv(f"fcl_run_cl_batches_{run_id}_{args.tag}.csv", index=False)
    print("✓ Wrote CSVs:",
          f"fcl_run_results_{run_id}_{args.tag}.csv,",
          f"fcl_run_summary_{run_id}_{args.tag}.csv,",
          f"fcl_run_cl_batches_{run_id}_{args.tag}.csv", flush=True)

if __name__ == "__main__":
    main()