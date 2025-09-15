import numpy as np
from typing import List
from collections import defaultdict

def make_cifar100_splits(targets, n_clients: int = 5, alpha: float = 0.2, seed: int = 0) -> List[np.ndarray]:
    """
    Split CIFAR-100 dataset indices into non-IID client subsets using Dirichlet distribution.
    - targets: list/array of dataset labels
    - n_clients: number of clients
    - alpha: Dirichlet concentration (smaller = more non-IID)
    - seed: random seed for reproducibility
    Returns: list of index arrays (one per client)
    """
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    n_classes = int(targets.max()) + 1

    # collect indices per class
    class_indices = [np.where(targets == c)[0] for c in range(n_classes)]
    for ci in class_indices:
        rng.shuffle(ci)

    # initialize client splits
    idx_splits = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        n_c = len(class_indices[c])
        if n_c == 0:
            continue

        # sample proportions for this class across clients
        proportions = rng.dirichlet(np.ones(n_clients) * alpha)
        counts = (proportions * n_c).astype(int)

        # fix rounding
        while counts.sum() < n_c:
            counts[rng.integers(0, n_clients)] += 1

        start = 0
        for k in range(n_clients):
            take = counts[k]
            if take > 0:
                idx_splits[k].extend(class_indices[c][start:start + take])
                start += take

    return [np.array(sorted(ix)) for ix in idx_splits]


if __name__ == "__main__":
    from torchvision import datasets, transforms

    # load CIFAR-100 targets only
    trainset = datasets.CIFAR100(root="./data", train=True, download=True,
                                 transform=transforms.ToTensor())
    splits = make_cifar100_splits(trainset.targets, n_clients=5, alpha=0.2, seed=42)

    for i, split in enumerate(splits):
        print(f"Client {i}: {len(split)} samples")