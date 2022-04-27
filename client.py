import torch
import torch.nn as nn
import numpy as np
from parse import args
from evaluate import evaluate_precision, evaluate_recall, evaluate_ndcg


class FedRecClient(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = []
        self.m_item = m_item
        self.dim = dim

        for i in target_ind:
            if i not in train_ind and i not in test_ind:
                self._target_.append(i)

        items, labels = [], []
        for pos_item in train_ind:
            items.append(pos_item)
            labels.append(1.)

            for _ in range(args.num_neg):
                neg_item = np.random.randint(m_item)
                while neg_item in train_ind:
                    neg_item = np.random.randint(m_item)
                items.append(neg_item)
                labels.append(0.)

        self._train_items = torch.Tensor(items).long()
        self._train_labels = torch.Tensor(labels).to(args.device)
        self._user_emb = nn.Embedding(1, dim)
        nn.init.normal_(self._user_emb.weight, std=0.01)

    def forward(self, items_emb, linear_layers):
        user_emb = self._user_emb.weight.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_(self, items_emb, linear_layers):
        items_emb = items_emb[self._train_items].clone().detach().requires_grad_(True)
        linear_layers = [(w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True))
                         for (w, b) in linear_layers]
        self._user_emb.zero_grad()

        predictions = self.forward(items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, self._train_labels)
        loss.backward()

        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        items_emb_grad = items_emb.grad
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._train_items, items_emb_grad, linear_layers_grad, loss.cpu().item()

    def eval_(self, items_emb, linear_layers):
        rating = self.forward(items_emb, linear_layers)
        rating[self._train_] = - (1 << 10)
        if self._test_:
            hr_at_20 = evaluate_recall(rating, self._test_, 20)
            prec_at_20 = evaluate_precision(rating, self._test_, 20)
            ndcg_at_20 = evaluate_ndcg(rating, self._test_, 20)
            test_result = np.array([hr_at_20, prec_at_20, ndcg_at_20])

            rating[self._test_] = - (1 << 10)
        else:
            test_result = None

        if self._target_:
            er_at_5 = evaluate_recall(rating, self._target_, 5)
            er_at_10 = evaluate_recall(rating, self._target_, 10)
            er_at_20 = evaluate_recall(rating, self._target_, 20)
            er_at_30 = evaluate_recall(rating, self._target_, 30)
            target_result = np.array([er_at_5, er_at_10, er_at_20, er_at_30])
        else:
            target_result = None

        return test_result, target_result

