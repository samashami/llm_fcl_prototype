# src/fl.py
from typing import List
from copy import deepcopy
import torch
from torch import nn
from src.strategies.replay import ReplayBuffer

class Server:
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    def average(self, models: List[nn.Module]) -> nn.Module:
        avg = deepcopy(models[0]).to(self.device)
        with torch.no_grad():
            for k in avg.state_dict().keys():
                s = sum(m.state_dict()[k].to(self.device) for m in models) / len(models)
                avg.state_dict()[k].copy_(s)
        return avg

class Client:
    def __init__(self, cid: int, model: nn.Module, optimizer, train_loader, device=None, replay: ReplayBuffer | None = None):
        self.cid = cid
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.loader = train_loader
        self.device = device or next(model.parameters()).device
        self.replay = replay or ReplayBuffer(capacity=2000)

    def load_state_from(self, global_model: nn.Module):
        self.model.load_state_dict(global_model.state_dict())

    def train_one_epoch(self, replay_ratio: float = 0.2):
        self.model.train()
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            # mix replay
            rx, ry = self.replay.sample_like(x.size(0), device=self.device, ratio=replay_ratio)
            if rx is not None:
                x = torch.cat([x, rx], dim=0)
                y = torch.cat([y, ry], dim=0)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # add to replay
            self.replay.add_batch(x.detach(), y.detach())