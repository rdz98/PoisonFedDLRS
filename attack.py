import torch
import torch.nn as nn
from parse import args
from client import FedRecClient


class BaselineAttackClient(FedRecClient):
    def __init__(self, train_ind, m_item, dim):
        super().__init__(train_ind, [], [], m_item, dim)

    def train_(self, items_emb, linear_layers):
        a, b, c, _ = super().train_(items_emb, linear_layers)
        return a, b, c, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None


class AttackClient(nn.Module):
    def __init__(self, target_items, m_item, dim):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)

    def forward(self, user_emb, items_emb, linear_layers):
        user_emb = user_emb.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_on_user_emb(self, user_emb, items_emb, linear_layers):
        predictions = self.forward(user_emb.requires_grad_(False), items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(args.device))
        return loss

    def train_(self, items_emb, linear_layers):
        target_items_emb = items_emb[self._target_].clone().detach()
        target_linear_layers = [[w.clone().detach(), b.clone().detach()] for w, b in linear_layers]
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        linear_layers = [[w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True)]
                         for (w, b) in linear_layers]
        s = 10
        total_loss = 0
        for _ in range(s):
            nn.init.normal_(self._user_emb.weight, std=0.01)
            if args.attack == 'A-hum':
                for __ in range(30):
                    predictions = self.forward(self._user_emb.weight.requires_grad_(True),
                                               target_items_emb, target_linear_layers)
                    loss = nn.BCELoss()(predictions, torch.zeros(len(self._target_)).to(args.device))

                    self._user_emb.zero_grad()
                    loss.backward()
                    self._user_emb.weight.data.add_(self._user_emb.weight.grad, alpha=-args.lr)
            total_loss += (1 / s) * self.train_on_user_emb(self._user_emb.weight, items_emb, linear_layers)
        total_loss.backward()

        items_emb_grad = items_emb.grad
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._target_, items_emb_grad, linear_layers_grad, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None

