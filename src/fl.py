from typing import List
from copy import deepcopy
import torch
from torch import nn

class Server:
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    def average(self, models: List[nn.Module]) -> nn.Module:
        """Uniform FedAvg."""
        avg = deepcopy(models[0]).to(self.device)
        with torch.no_grad():
            for k in avg.state_dict().keys():
                s = sum(m.state_dict()[k].to(self.device) for m in models) / len(models)
                avg.state_dict()[k].copy_(s)
        return avg


class Client:
    def __init__(self, cid: int, model: nn.Module, optimizer, train_loader, device=None):
        self.cid = cid
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.loader = train_loader
        self.device = device or next(model.parameters()).device

    def train_one_epoch(self):
        self.model.train()
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def load_state_from(self, global_model: nn.Module):
        self.model.load_state_dict(global_model.state_dict())