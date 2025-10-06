# src/fl.py
from typing import List, Optional
from copy import deepcopy
import torch
from torch import nn
from src.strategies.replay import ReplayBuffer

class Server:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")

    def average(self, models: List[nn.Module]) -> nn.Module:
        # FedAvg on CPU or given device
        avg = deepcopy(models[0]).to(self.device)
        with torch.no_grad():
            for k in avg.state_dict().keys():
                s = sum(m.state_dict()[k].to(self.device) for m in models) / len(models)
                avg.state_dict()[k].copy_(s)
        return avg.to(self.device)

class Client:
    def __init__(self, cid: int, model: nn.Module, optimizer, train_loader,
                 device: Optional[torch.device] = None, replay: Optional[ReplayBuffer] = None):
        self.cid = cid
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)                     # ⬅ ensure model on device
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.loader = train_loader
        self.replay = replay or ReplayBuffer(capacity=2000)

    def load_state_from(self, global_model: nn.Module):
        # load weights then keep model on device
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)                             # ⬅ keep on device

    def train_one_epoch(
        self,
        replay_ratio: float = 0.2,
        epoch: int = 0,
        total_epochs: int = 1,
        log_interval: int = 200,
    ):
        self.model.train()
        num_batches = len(self.loader)
        running_loss = 0.0

        for b, (x, y) in enumerate(self.loader, 1):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            # optional replay: sample on CPU, then move to device
            if self.replay is not None and replay_ratio > 0.0:
                rx, ry = self.replay.sample_like(x.size(0), ratio=replay_ratio)
                if rx is not None:
                    rx = rx.to(self.device, non_blocking=True)
                    ry = ry.to(self.device, non_blocking=True)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry], dim=0)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())

            # store *CPU* copies to avoid VRAM growth
            if self.replay is not None:
                self.replay.add_batch(x.detach().cpu(), y.detach().cpu())

            # periodic log
            if (b % log_interval) == 0 or b == num_batches:
                print(
                    f"[Client {self.cid}] epoch {epoch+1}/{total_epochs} "
                    f"batch {b}/{num_batches} loss={loss.item():.4f}",
                    flush=True,
                )

        avg_loss = running_loss / max(1, num_batches)
        print(
            f"[Client {self.cid}] epoch {epoch+1}/{total_epochs} done, avg_loss={avg_loss:.4f}",
            flush=True,
        )
        return avg_loss