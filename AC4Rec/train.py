import datetime
import pickle
import heapq
import math
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
from AC4Rec.utils import DataPre, Voc, data_split, Policy, BudgetNet, item_split, action_select, action_distribution

# # data prepare
# df = pd.read_csv('../dataset/filtered_data.csv')
# basetime = datetime.datetime.strptime(df['出价时间'].min(), '%Y-%m-%d')
# # 将出价时间设置为与basetime的差
# df['出价时间'] = df['出价时间'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d') - basetime).days)
#
# dp = DataPre(df)
#
# f = open("./dp.pkl", "wb")
# pickle.dump(dp, f)
# f.close()

print("loading dataset...")
with open("./dp.pkl", "rb") as f:
    dp = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '../dataset/filtered_data.csv'
BATCH_SIZE = 32
GAMMA = 0.9
LR = 5e-4
# ITEM_DIM == USER_DIM == BUDGET_DIM
ITEM_DIM = 256
USER_DIM = 256
BUDGET_DIM = 256
GRU_HIDDEN_SIZE = 512
EPOCH = 30
BLOCK_SIZE = 256
BLOCK_NUM = math.ceil(dp.itemVoc.num_words/BLOCK_SIZE)

print("split budgets...")
item_price = list(enumerate(dp.itemPrice))
item_price.sort(key=lambda x: x[1])
budget_blocks = item_split(item_price, BLOCK_SIZE)



class Actor(object):
    def __init__(self,user_num, user_dim, item_num, item_dim, item_price, user_budegt, budget_blocks):
        self.user_num = user_num
        self.user_dim = user_dim
        self.item_num = item_num
        self.item_dim = item_dim
        self.item_price = item_price  # sorted [(item_id, item_price),(),......]
        self.user_budget = user_budegt # user budgets
        self.budget_blocks_action_memory = None
        self.item_action_memory = []
        self.budget_blocks = budget_blocks

        self.budget_policys = None
        self.item_policys = None
        self.genPolicys()

        # 根据state生成budgets
        self.budget_net = BudgetNet(
            user_num=self.user_num,
            user_dim=user_dim,
            item_num = item_num,
            item_dim = item_dim,
            budget_dim=BUDGET_DIM,
            gru_hidden_size=GRU_HIDDEN_SIZE).to(device)

        # 优化器
        self.budget_policy_optim = torch.optim.Adam([{"params":net.parameters()} for net in self.budget_policys], lr=LR, weight_decay=0.05)
        self.budget_net_optim = torch.optim.Adam(self.budget_net.parameters(), lr=LR, weight_decay=0.05)
        self.item_policy_optim = torch.optim.Adam([{"params":net.parameters()} for net in self.item_policys], lr=LR, weight_decay=0.05)

    def genPolicys(self):
        budget_policys = []
        item_policys = []

        # 按照BLOCK_NUM的大小生成budget范围选择策略网络
        for i in range(math.ceil(math.log(BLOCK_NUM, 2))):
            budget_policys.append(Policy(BUDGET_DIM).to(device))

        # 按照BLOCK_SIZE大小生成item选择策略网络
        for i in range(math.ceil(math.log(BLOCK_SIZE, 2))):
            item_policys.append(Policy(BUDGET_DIM).to(device))

        self.budget_policys = budget_policys
        self.item_policys = item_policys

    def choose_action(self, cur_item_id, gold_item_id, user_id):
        # 估计用户的预算  [budget_dim]
        if len(self.item_action_memory) != 0:
            pre_item_id = self.item_action_memory[-1]
        else:
            pre_item_id = None
        budget = self.budget_net(pre_item_id, cur_item_id, user_id)

        # 根据budget选择budget_block
        selected_budget_block_id, selected_budget_block_prob = action_select(budget, len(self.budget_blocks), self.budget_policys)
        # self.budget_blocks_action_memory.append(selected_budget_block_id)

        # 根据budgets选择item
        selected_item_id, selected_item_prob = action_select(budget, len(self.budget_blocks[selected_budget_block_id]), self.item_policys)
        self.item_action_memory.append(selected_item_id)

        # 计算item_action的概率分布
        item_action_dist = action_distribution(budget,len(self.budget_blocks[selected_budget_block_id]),self.item_policys)

        # 根据预测出来item和gold_item的相似度(这里选择的是价格差)设计reward。
        selected_item_price = dp.itemPrice[self.budget_blocks[selected_budget_block_id][selected_item_id][0]]
        user_budget = self.user_budget[user_id]
        if self.budget_blocks[selected_budget_block_id][selected_item_id][0] == gold_item_id:
            reward = 5
        elif selected_item_price < user_budget[1] and selected_item_price>user_budget[0]:
            reward = 1
        else:
            reward = -1

        return budget, item_action_dist, selected_item_id, reward

    def learn(self,item_id, item_dist, td_error):

        # item_dist = item_dist.squeeze()
        # 当前action
        selected_item_id = torch.LongTensor([item_id]).to(device)
        item_dist = torch.concat(item_dist, dim=0)

        l = torch.nn.NLLLoss()
        log_softmax_input = torch.log(item_dist).unsqueeze(dim=0)
        neg_log_prob = l(log_softmax_input, selected_item_id)

        loss_a = -neg_log_prob * td_error

        # 梯度归零
        self.budget_net_optim.zero_grad()
        self.budget_policy_optim.zero_grad()
        self.item_policy_optim.zero_grad()

        # 计算梯度
        loss_a.backward()

        # 梯度裁剪防止梯度爆炸
        nn.utils.clip_grad_norm_(self.budget_net.parameters(), 0.1)
        for p in self.budget_policys:
            nn.utils.clip_grad_norm_(p.parameters(), 0.1)
        for p in self.item_policys:
            nn.utils.clip_grad_norm_(p.parameters(), 0.1)

        # 更新参数
        self.budget_net_optim.step()
        self.budget_policy_optim.step()
        self.item_policy_optim.step()

class QNetwork(nn.Module):
    """
    critic 主干网络，
    输入为state，
    输出为状态值
    """
    def __init__(self, state_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, int(0.5 * state_dim)),
            nn.ReLU(),
            nn.Linear(int(0.5 * state_dim), 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.fc(x)
        return out

class Critic(object):
    # 通过采样数据，学习V(S)
    def __init__(self, input_dim):

        self.input_dim = input_dim
        # 输入S，输出V(S)
        self.network = QNetwork(state_dim=self.input_dim).to(device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def train_Q_network(self, state, reward, next_state):
        # 类似于DQN的5.4，不过这里没有用fixed network，experience relay的机制

        # s, s_ = torch.FloatTensor(state), torch.FloatTensor(next_state)
        # 当前状态，执行了action之后的状态

        v = self.network(state)  # v(s)
        v_ = self.network(next_state)  # v(s')

        # TD
        # r+γV(S') 和V(S) 之间的差距
        loss_q = self.loss_func(reward + GAMMA * v_, v)

        self.optimizer.zero_grad()
        # 反向传播
        loss_q.backward()

        # 梯度裁剪防止梯度爆炸
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

        self.optimizer.step()
        # pytorch老三样

        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v
        # 表示不把相应的梯度传到actor中（actor和critic是独立训练的）

        return td_error

actor = Actor(user_num=dp.userVoc.num_words,
              user_dim=USER_DIM,
              item_num=dp.itemVoc.num_words,
              item_dim=ITEM_DIM,
              item_price=dp.itemPrice,
              user_budegt=dp.userBudgets,
              budget_blocks=budget_blocks)

critic = Critic(input_dim=BUDGET_DIM)

train_data, eval_data = data_split(dp.seq, rate=0.8)


# 训练
for epoch in range(EPOCH):
    # 设置模型为训练状态
    for p in actor.item_policys:
        p.train()
    for p in actor.budget_policys:
        p.train()
    actor.budget_net.train()
    critic.network.train()

    start = time.time()
    for uid, eps in list(train_data.items())[:100]:
        # 清空memory
        actor.item_action_memory = []
        for i in range(len(eps)-1):
            input_item_id = eps[i][0]
            gold_item_id = eps[i+1][0]

            cur_state, item_dist, item_id_pred, reward = actor.choose_action(cur_item_id=input_item_id,
                                                                             gold_item_id=gold_item_id,
                                                                             user_id=uid)
            with torch.no_grad():
                next_state = actor.budget_net(item_id_pred,gold_item_id,uid)

            td_error = critic.train_Q_network(
                cur_state.clone().detach(),
                reward,
                next_state)

            actor.learn(item_id_pred, item_dist, td_error)
            # print("end")
            # true_gradient = grad[logPi(a|s) * td_error]
            # 然后根据前面学到的V（s）值，训练actor，以更好地采样动作
    total_reward = 0
    # 设置模型为训练状态
    for p in actor.item_policys:
        p.eval()
    for p in actor.budget_policys:
        p.eval()
    actor.budget_net.eval()
    critic.network.eval()
    # 取消梯度跟踪
    with torch.no_grad():
        for uid, eps in list(eval_data.items())[:100]:
            for i in range(len(eps) - 1):  # car_id, 是否中标, 出价金额, 出价时间
                input_item_id = eps[i][0]
                gold_item_id = eps[i + 1][0]

                _, _, _, reward = actor.choose_action(cur_item_id=input_item_id,
                                                      gold_item_id=gold_item_id,
                                                      user_id=uid)
                total_reward += reward
    end = time.time()
    print(f"epoch:{epoch}  total reward:{total_reward}  time:{round((end-start)/60, 2)}" )