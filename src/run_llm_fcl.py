# src/run_llm_fcl.py
import argparse, numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights
import random
import csv, os

from src.model import build_resnet18
from src.fl import Client, Server
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
            for c in range(n_classes):
                mask = (y == c)
                if mask.any():
                    counts[c] += mask.sum().item()
                    hits[c] += (pred[mask] == c).sum().item()
    acc = correct / total
    per_class_recall = np.array([ (hits[c]/counts[c]) if counts[c] > 0 else 0.0 for c in range(n_classes) ], dtype=np.float32)
    return acc, per_class_recall

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic behavior (safe even on CPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset_per_client", type=int, default=4000, help="for fast CPU demo; use -1 for all")
    ap.add_argument("--use_policy", action="store_true", help="enable LLM-like policy controller")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=200, help="batches between progress prints")
    ap.add_argument("--split_mode", choices=["equal", "dirichlet"], default="equal",
                help="equal: equal-size random split per client; dirichlet: non-iid alpha")
    ap.add_argument("--val_size", type=int, default=5000, help="validation holdout from CIFAR100 train")

    args = ap.parse_args()
    set_seeds(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # make seed visible to worker processes
    global GLOBAL_SEED
    GLOBAL_SEED = args.seed

    # seeded DataLoader workers
    g = torch.Generator()
    g.manual_seed(args.seed)

    # def _seed_worker(worker_id: int):
    #     np.random.seed(args.seed + worker_id)
    #     random.seed(args.seed + worker_id)


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
    total_train = len(trainset_full)  # 50_000
    val_size = args.val_size          # 5_000 (paper)
    train_size = total_train - val_size

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=g
    )

    # absolute indices into trainset_full
    train_indices = np.array(train_subset.indices, dtype=np.int64)
    val_indices   = np.array(val_subset.indices, dtype=np.int64)

    # loaders for val/test (val not yet used; will be in early stopping step)
    valset = Subset(trainset_full, val_indices)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

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

    # --- init clients ---
    # init clients
    clients = []
    for cid, idx in enumerate(splits):
        subset = Subset(trainset_full, idx)  # note: from *full* train with tf_train
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        model = build_resnet18(100).to(device)
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        replay = ReplayBuffer(capacity=2000)
        clients.append(Client(cid, model, opt, loader, device=device, replay=replay))

         ####### for testing: count params
        if cid == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Trainable params: {trainable:,}/{total}", flush=True)

    # ✅ Check device
    print("✅ Client 0 model device:", next(clients[0].model.parameters()).device, flush=True)


    # --- server, policy, and initial eval ---
    server = Server(device=device)
    policy = Policy()
    best_recall = np.zeros(100, dtype=np.float32)
    last_acc = None

    # initial global model (FedAvg of untrained client models)
    global_model = server.average([c.model for c in clients])
    acc, per_class = evaluate(global_model, device, test_loader)
    forgetting = np.maximum(0.0, best_recall - per_class)
    best_recall = np.maximum(best_recall, per_class)
    print(f"[Round -1] acc={acc:.3f}", flush=True)

    # --- training rounds ---
    for r in range(args.rounds):
        acc_delta = 0.0 if last_acc is None else (acc - last_acc)
        if args.use_policy:
            summary = {
                "round": r,
                "accuracy_global": float(acc),
                "acc_delta": float(acc_delta),
                "forgetting_per_class": [float(x) for x in forgetting],
                "non_iid_alpha": float(args.alpha),
            }
            hp = policy.decide(summary)
        else:
            hp = {"lr": args.lr, "replay_ratio": 0.20, "notes": "fixed (baseline)"}
        print(f"[Policy r={r}] acc={acc:.3f} Δ={acc_delta:+.3f} -> lr={hp['lr']:.5f}, replay={hp['replay_ratio']:.2f} ({hp['notes']})", flush=True)

        # broadcast global + set lr
        for c in clients:
            c.load_state_from(global_model)
            for pg in c.optimizer.param_groups:
                pg["lr"] = hp["lr"]

        # local training
        for c in clients:
            print(f"[Round {r}] training client {c.cid} on {next(c.model.parameters()).device}", flush=True)
            for e in range(args.epochs):
                c.train_one_epoch(
                    replay_ratio=hp["replay_ratio"],
                    epoch=e,
                    total_epochs=args.epochs,
                    log_interval=args.log_interval,
                )

        # aggregate & eval
        global_model = server.average([c.model for c in clients])
        last_acc = acc
        acc, per_class = evaluate(global_model, device, test_loader)
        forgetting = np.maximum(0.0, best_recall - per_class)
        best_recall = np.maximum(best_recall, per_class)
        print(f"[Round {r}] acc={acc:.3f}", flush=True)



if __name__ == "__main__":
    main()