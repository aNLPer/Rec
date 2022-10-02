import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

a = torch.tensor([[1],[2],[3]])
b = torch.tensor([[1],[4],[7]])
print(a.mul(b))
print(b.mul(a))