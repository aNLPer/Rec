import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

a = torch.tensor([[[0.9,0.0,0.1],
                   [0.1,0.0,0.9]],

                  [[0.9, 0.0, 0.1],
                   [0.1, 0.0, 0.9]]])
m = Categorical(a)
print(m.sample())