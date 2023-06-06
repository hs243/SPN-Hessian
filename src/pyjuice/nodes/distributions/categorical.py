from __future__ import annotations

import torch

from .distributions import Distribution


class Categorical(Distribution):
    def __init__(self, num_cats: int):
        super(Categorical, self).__init__()

        self.num_cats = num_cats

    def raw2processed_params(self, params: torch.Tensor):
        return params.reshape(-1, self.num_cats)

    def processed2raw_params(self, params: torch.Tensor):
        return params.reshape(-1)

    def init_parameters(self, num_nodes: int, perturbation: float, **kwargs):
        params = torch.exp(torch.rand([num_nodes, self.num_cats]) * -perturbation)
        params /= params.sum(dim = 1, keepdim = True)
        return params.reshape(num_nodes * self.num_cats)

    def __getstate__(self):
        state = {"num_cats": self.num_cats}
        return state

    def __setstate__(self, state):
        self.num_cats = state["num_cats"]