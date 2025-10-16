import argparse, numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights
import random
import time
import pandas as pd
import copy

# --- Controller v4: context-aware heuristics (REVISED) ---
V4_LR_MIN, V4_LR_MAX = 1e-4, 2e-3          # Widened MAX LR for faster exploration
V4_REP_MIN, V4_REP_MAX = 0.20, 0.70
V4_DEADBAND = 0.003                       # tiny Δacc → hold
V4_REP_STEP_HIGH = 0.10                   # when forgetting/divergence high
V4_REP_STEP_LOW  = 0.05                   # when forgetting low
V4_FORGET_THR    = 0.05                    # mean forgetting threshold
V4_DIV_THR       = 0.10                    # normalized divergence threshold (normalized STD)
V4_EMA_ALPHA     = 0.3                     # EMA for global loss smoothing
V4_LR_BOOST      = 1.35                    # converge faster when underfitting
V4_LR_COOLDOWN   = 1.5                     # cool when unstable
V4_CLIENT_LR_MIN, V4_CLIENT_LR_MAX = 0.8, 1.2 # per-client LR scaling range
V4_ROLLBACK_THR  = 0.02                    # 2% absolute accuracy drop to trigger rollback
V4_WARMUP_ROUNDS = 2                       # New: 2 rounds of fixed HP before adaptation

# --- Dependencies (assuming these are defined in your environment) ---
from src.model import build_resnet18
from src.fl import Client, Server # NOTE: Server must have save_state/load_state methods
from src.strategies.replay import ReplayBuffer
from src.policy import Policy

# top of file (after imports)
GLOBAL_SEED = 42
def seed_worker(worker_id: int):
    import numpy as _np, random as _random
    _np.random.seed(GLOBAL_SEED + worker_id)
    _random.seed(GLOBAL_SEED + worker_id)

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
            # Track per-class performance (used for forgetting metric)
            for c in range(n_classes):
                mask = (y == c)
                if mask.any():
                    counts[c] += mask.sum().item()
                    hits[c] += (pred[mask] == c).sum().item()
    acc = correct / total
    # Calculate per-class recall (avoiding division by zero)
    per_class_recall = np.array([ (hits[c]/counts[c]) if counts[c] > 0 else 0.0 for c in range(n_classes) ], dtype=np.float32)
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

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=7) # Default to 7 for consistency with CL batches
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset_per_client", type=int, default=-1, help="use -1 for all data")
    ap.add_argument("--use_policy", action="store_true", default=True,
                help="enable LLM-like policy controller (default True here)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=200, help="batches between progress prints")
    ap.add_argument("--split_mode", choices=["equal", "dirichlet"], default="equal",
                help="equal: equal-size random split per client")
    ap.add_argument("--val_size", type=int, default=5000, help="validation holdout from CIFAR100 train")
    ap.add_argument("--cl_batches", type=int, default=7, help="number of continual-learning batches per client")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (paper: 4)")
    ap.add_argument("--optimizer", choices=["adam","sgd"], default="adam",
                help="Paper fine-tuning used Adam; switch to sgd if you want FedSGD baseline.")
    ap.add_argument("--early_patience", type=int, default=5)
    ap.add_argument("--tag", type=str, default="controller_v4",
                help="label for this run (used in CSV filenames)")
    
    args = ap.parse_args()
    set_seeds(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # make seed visible to worker processes
    global GLOBAL_SEED
    GLOBAL_SEED = args.seed

    g = torch.Generator()
    g.manual_seed(args.seed)

    # transforms
    tf_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),  # ImageNet stats
    ])
    tf_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    
    # ===== datasets & paper-faithful split =====
    trainset_full = datasets.CIFAR100(root="./data", train=True, download=True, transform=tf_train)
    testset       = datasets.CIFAR100(root="./data", train=False, download=True, transform=tf_test)

    # --- hold out validation from CIFAR-100 train ---
    total_train = len(trainset_full) 
    val_size = args.val_size         
    train_size = total_train - val_size

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=g
    )

    train_indices = np.array(train_subset.indices, dtype=np.int64)
    val_indices   = np.array(val_subset.indices, dtype=np.int64)

    valset = Subset(trainset_full, val_indices)
    val_loader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"[Split] train={len(train_indices)} val={len(val_indices)} test={len(testset)}", flush=True)

    # --- client splits (equal-size, random) ---
    if args.split_mode == "equal":
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(train_indices)
        sizes = [len(perm) // args.clients] * args.clients
        for i in range(len(perm) % args.clients):
            sizes[i] += 1
        splits = []
        start = 0
        for s in sizes:
            splits.append(perm[start:start+s].tolist())
            start += s
    else:
        raise NotImplementedError("dirichlet over 45k subset comes later; use --split_mode equal for now")

    # optional subsample per client (for speed)
    if args.subset_per_client and args.subset_per_client > 0:
        splits = [idxs[:args.subset_per_client] for idxs in splits]

    for i, idxs in enumerate(splits):
        print(f"[Split] client {i}: {len(idxs)} images", flush=True)

    # === Build CL schedule: 1 initial batch + (cl_batches-1) increments ===
    def make_cl_batches(indices, num_batches=7, seed=42):
        rng = np.random.RandomState(seed)
        idx = np.array(indices, dtype=np.int64)
        rng.shuffle(idx)

        # initial batch ~ 0.466 of data (e.g., 5250/11250 in your example)
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


    # Build a per-client list of batches
    cl_schedule = []
    for cid, idxs in enumerate(splits):
        batches = make_cl_batches(idxs, num_batches=args.cl_batches, seed=args.seed + cid)
        cl_schedule.append(batches)
        sizes = [len(b) for b in batches]
        print(f"[CL] client {cid}: {sizes} (sum={sum(sizes)})", flush=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_logs = []       
    round_logs = []     

    # store CL schedule for export
    cl_rows = []
    for cid, batches in enumerate(cl_schedule):
        for i, b in enumerate(batches, start=1):
            cl_rows.append({
                "run_id": run_id,
                "client": cid,
                "cl_batch": i,
                "size": len(b)
            })
    
    # --- init clients ---
    clients = []
    for cid, idx in enumerate(splits):
        subset = Subset(trainset_full, idx)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker, generator=g,)
        
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


    # --- server, policy, and initial eval ---
    server = Server(device=device)
    policy = Policy()
    
    # Track best state for rollback and initialization
    best_global_acc = -1.0
    best_hp = {"lr": args.lr, "replay_ratio": 0.50, "notes": "init"}
    
    # 1) initial global model (FedAvg of untrained client models)
    global_model = server.average([c.model for c in clients])

    # 2) initial metrics
    acc, per_class = evaluate(global_model, device, test_loader)
    
    # Initialize trackers
    best_recall = per_class.copy()
    forgetting = np.zeros_like(per_class)

    global_loss = evaluate_loss(global_model, device, test_loader)
    ema_loss = global_loss           
    prev_global_loss = global_loss   
    div_norm = 0.0

    print(f"[Round -1] acc={acc:.3f}", flush=True)
    
    # Save the initial (untrained) model as the first 'best' state
    server.save_state(global_model, acc) 
    best_global_acc = acc
    last_acc = acc
    last_hp = copy.deepcopy(best_hp)


    # --- training rounds ---
    for r in range(args.rounds):
        acc_delta = acc - last_acc

        if args.use_policy:
            
            # Reset policy to the best known state if rollback was requested
            if server._rollback_flag:
                lr = best_hp["lr"]
                rep = best_hp["replay_ratio"]
                notes = [f"ROLLBACK (r{server._rollback_round}): Reverted to best HP from r{server._best_round}"]
                server._rollback_flag = False # Clear flag
                
            elif r < V4_WARMUP_ROUNDS:
                # V4 Warm-up: fixed paper defaults for first few rounds
                lr, rep = args.lr, 0.50
                notes = ["warmup (fixed defaults)"]
            
            else:
                # Policy Decision Based on Signals
                
                # Base HP from the *last stable* round
                lr = last_hp["lr"]
                rep = last_hp["replay_ratio"]
                notes = ["policy_v4"]

                F_t = float(np.mean(forgetting))        # mean forgetting signal
                Δacc = float(acc_delta)                 # accuracy change this round
                div  = float(div_norm)                  # divergence signal
                L_ema = float(ema_loss)                 # smoothed loss 

                # 1) Deadband on accuracy (hold small changes)
                if abs(Δacc) < V4_DEADBAND:
                    notes.append(f"deadband(|Δacc|<{V4_DEADBAND})")
                    # No change to lr/rep if within deadband, proceed to clamping
                    
                else:
                    # 2) Forgetting/Divergence-driven Replay Scheduling
                    if F_t > V4_FORGET_THR or div > V4_DIV_THR:
                        rep += V4_REP_STEP_HIGH
                        notes.append("replay↑ (forget/div high)")
                    else:
                        rep -= V4_REP_STEP_LOW
                        notes.append("replay↓ (forget low)")

                    # 3) Convergence-aware LR (EMA of loss + Δacc)
                    if Δacc < -V4_DEADBAND:
                        # instability / overfitting → cool down
                        lr /= V4_LR_COOLDOWN
                        notes.append("lr↓ (Δacc<0)")
                    elif Δacc > V4_DEADBAND and L_ema > 1.5:  # still high loss (approx 1.5) → push
                        lr *= V4_LR_BOOST
                        notes.append("lr↑ (loss high & improving)")
                    # else: keep lr
            
            # 4) Clamp to safe ranges (applies to all scenarios)
            lr  = max(V4_LR_MIN,  min(V4_LR_MAX,  lr))
            rep = max(V4_REP_MIN, min(V4_REP_MAX, rep))
            notes.append(f"clamped(lr∈[{V4_LR_MIN},{V4_LR_MAX}], rep∈[{V4_REP_MIN:.2f},{V4_REP_MAX:.2f}])")

            hp = {"lr": lr, "replay_ratio": rep, "notes": " | ".join(notes)}

        else:
            hp = {"lr": args.lr, "replay_ratio": 0.50, "notes": "fixed (paper CL default)"}

        print(
            f"[Policy r={r}] acc={acc:.3f} Δ={acc_delta:+.3f}, F_t={F_t:.3f}, Div={div_norm:.3f} "
            f"-> lr={hp['lr']:.5f}, replay={hp['replay_ratio']:.2f} ({hp['notes']})",
            flush=True,
        )

        # Update last_hp for the next round's policy base
        last_hp = {"lr": hp["lr"], "replay_ratio": hp["replay_ratio"], "notes": hp["notes"]}


        # --- Per-client LR Scaling (INVERTED LOGIC) ---
        vlosses = []
        for c in clients:
            v = getattr(c, "_last_vloss", None)
            # Use local validation loss or the global loss if validation was skipped/failed
            vlosses.append(float(v) if v is not None and not np.isnan(v) else float(global_loss))
            
        vl_min, vl_max = float(np.min(vlosses)), float(np.max(vlosses))
        rng = max(1e-8, vl_max - vl_min)

        for i, c in enumerate(clients):
            c.load_state_from(global_model)
            
            # Calculate rank [0.0 (min loss) to 1.0 (max loss)]
            rank = (vlosses[i] - vl_min) / rng 
            
            # INVERTED Scaling: High loss (rank=1.0) -> Low scale (V4_CLIENT_LR_MIN=0.8)
            # Low loss (rank=0.0) -> High scale (V4_CLIENT_LR_MAX=1.2)
            # Use (1 - rank) to invert the scaling influence
            scale = V4_CLIENT_LR_MIN + (1.0 - rank) * (V4_CLIENT_LR_MAX - V4_CLIENT_LR_MIN)
            
            scale = max(V4_CLIENT_LR_MIN, min(V4_CLIENT_LR_MAX, scale)) 
            
            for pg in c.optimizer.param_groups:
                pg["lr"] = hp["lr"] * float(scale)
                # Store the actual LR used for logging/debugging
                c._last_lr_scale = float(scale) 
                

        # --- local continual-learning training ---
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
                pin_memory=True, worker_init_fn=seed_worker,
                generator=g,
            )

            print(f"[Round {r}] client {c.cid}: CL batch {batch_id+1}/{len(batches)} "
                f"(new={len(batch_indices)}; replay ratio≈{hp['replay_ratio']:.2f}, LR_scale={c._last_lr_scale:.2f})",
                flush=True)

            # Train for several epochs, mixing replay from previous batches
            for e in range(args.epochs):     
                avg_loss, epoch_acc, stop = c.train_one_epoch(
                    replay_ratio=hp["replay_ratio"],
                    epoch=e,
                    total_epochs=args.epochs,
                    log_interval=args.log_interval,
                )
                
                run_logs.append({
                    "run_id": run_id, "tag": args.tag, "round": r, "client": c.cid,
                    "epoch": e + 1, "lr": c.optimizer.param_groups[0]["lr"], 
                    "replay_ratio": hp["replay_ratio"], "cl_batch": batch_id + 1,
                    "cl_batch_size": len(batch_indices), "train_loss": float(avg_loss), 
                    "train_acc": float(epoch_acc),
                    "val_loss": float(getattr(c, "_last_vloss", float("nan"))),
                    "val_acc": float(getattr(c, "_last_vacc", float("nan"))),
                })

                if stop:
                    print(
                        f"[Client {c.cid}] Early stopping triggered "
                        f"(no val improvement {c.early_patience} epochs).",
                        flush=True,
                    )
                    break


         # --- compute client divergence w.r.t. current global (before FedAvg) ---
        with torch.no_grad():
            def flat_params(m: torch.nn.Module):
                return torch.cat([p.detach().float().view(-1).to(device) for p in m.parameters()])

            g_flat = flat_params(global_model) # the global model before aggregation
            dists = []
            for c in clients:
                c_flat = flat_params(c.model)
                dists.append(torch.norm(c_flat - g_flat, p=2).item())

            if len(dists) > 1: # Requires at least two clients to compute std
                div_norm = float(np.std(dists) / (np.median(dists) + 1e-8)) # Normalized STD
            else:
                div_norm = 0.0

        # --- aggregate & evaluate global model ---
        global_model = server.average([c.model for c in clients])
        last_acc = acc # Store previous round's accuracy
        acc, per_class = evaluate(global_model, device, test_loader)

        # --- V4: ROLLBACK CHECK ---
        if acc < best_global_acc - V4_ROLLBACK_THR:
            server.load_state(global_model) # Load best model parameters back into global_model
            acc = best_global_acc           # Reset reported accuracy to best
            forgetting = np.maximum(0.0, best_recall - per_class) # Recalculate forgetting
            print(f"[🔥 ROLLBACK R{r}] Sharp drop ({last_acc:.3f} -> {acc:.3f} was {best_global_acc:.3f}). "
                  f"Reverting model state to best (r{server._best_round}). Policy will revert to its best HP.", 
                  flush=True)
            # Set flag for the policy check next round (r+1)
            server._rollback_flag = True
            server._rollback_round = r
        
        # --- Update Best State ---
        if acc > best_global_acc:
            best_global_acc = acc
            best_hp = copy.deepcopy(hp)
            server.save_state(global_model, acc)
        
        # --- Update Loss/Forgetting Trackers ---
        prev_global_loss = global_loss
        global_loss = evaluate_loss(global_model, device, test_loader)
        ema_loss = V4_EMA_ALPHA * global_loss + (1.0 - V4_EMA_ALPHA) * ema_loss

        forgetting = np.maximum(0.0, best_recall - per_class)
        best_recall = np.maximum(best_recall, per_class)

        round_logs.append({
            "run_id": run_id, "tag": args.tag, "round": r, "global_acc": float(acc), 
            "lr": hp["lr"], "replay_ratio": hp["replay_ratio"], "notes": hp.get("notes", ""),
            "global_loss": float(global_loss), "ema_loss": float(ema_loss), 
            "forget_mean": float(np.mean(forgetting)), "divergence": float(div_norm),
            "best_acc_so_far": float(best_global_acc), "was_rollback": server._rollback_flag,
        })
        
        print(f"[Round {r}] acc={acc:.3f} (best={best_global_acc:.3f})", flush=True)

    # === write CSVs ===
    pd.DataFrame(run_logs).to_csv(f"fcl_run_results_{run_id}_{args.tag}.csv", index=False)
    pd.DataFrame(round_logs).to_csv(f"fcl_run_summary_{run_id}_{args.tag}.csv", index=False)
    pd.DataFrame(cl_rows).to_csv(f"fcl_run_cl_batches_{run_id}_{args.tag}.csv", index=False)
    print("✓ Wrote CSVs:", f"fcl_run_results_{run_id}_{args.tag}.csv, ...")

if __name__ == "__main__":
    main()