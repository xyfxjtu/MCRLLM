import pickle
from abc import ABC, abstractmethod
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from typing import Tuple, List, Dict
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import copy


class ComplEx(nn.Module):
    def __init__(
            self, rank, nodes
    ):
        super(ComplEx, self).__init__()
        self.rank = rank
        self.nodes = nodes.weight
        self.n_node = nodes.weight.shape[0]
        self.to_score = nn.Parameter(torch.Tensor(self.n_node, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.to_score)

    def forward(self, lhs,rel,rhs,to_score, candidate, score, start, end, queries):

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        to_score = to_score
        # rhs = rhs
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        _to_score = to_score[:, :self.rank], to_score[:, self.rank:]

        if candidate:
            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1) @ to_score[start:end].transpose(0, 1)
        if score:
            return torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * _to_score[0][queries[:, 2]] + (lhs[0] * rel[1] + lhs[1] * rel[0]) *
                _to_score[1][queries[:, 2]], 1, keepdim=True)
        else:
            return (
                (lhs[0] * rel[0] - lhs[1] * rel[1]) @ _to_score[0].transpose(0, 1) + (
                            lhs[0] * rel[1] + lhs[1] * rel[0]) @
                _to_score[1].transpose(0, 1), (
                    lhs[0] ** 2 + lhs[1] ** 2,
                    rel[0] ** 2 + rel[1] ** 2,
                    rhs[0] ** 2 + rhs[1] ** 2,
                )
            )

