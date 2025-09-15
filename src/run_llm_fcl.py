import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import datasets, transforms

from src.data import make_cifar100_splits
from src.model import build_resnet18
from src.fl import Client, Server

def evaluate(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # transforms
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507,0.486,0.440),(0.267,0.256,0.276)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507,0.486,0.440),(0.267,0.256,0.276)),
    ])

    # datasets
    trainset = datasets.CIFAR100(root="./data", train=True, download=False, transform=tf_train)
    testset  = datasets.CIFAR100(root="./data", train=False, download=False, transform=tf_test)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # non-IID client splits
    splits = make_cifar100_splits(trainset.targets, n_clients=args.clients, alpha=args.alpha, seed=0)

    # init clients
    clients = []
    for cid, idx in enumerate(splits):
        subset = Subset(trainset, idx)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        model = build_resnet18(100).to(device)
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        clients.append(Client(cid, model, opt, loader, device=device))

    server = Server(device=device)

    # initial evaluation (randomly initialized clients → we average for a "global" view)
    global_model = server.average([c.model for c in clients])
    acc0 = evaluate(global_model, device, test_loader)
    print(f"[Round -1] global acc = {acc0:.3f}")

    # federated rounds
    for r in range(args.rounds):
        # broadcast current global to clients
        for c in clients:
            c.load_state_from(global_model)

        # local training
        for c in clients:
            for _ in range(args.epochs):
                c.train_one_epoch()

        # aggregate
        global_model = server.average([c.model for c in clients])

        # test
        acc = evaluate(global_model, device, test_loader)
        print(f"[Round {r}] global acc = {acc:.3f}")

if __name__ == "__main__":
    main()