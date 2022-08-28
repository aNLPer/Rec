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
from AC4Rec.utils_multiUser import DataPre, Voc, data_split, Policy, BudgetNet, item_split, action_select, action_distribution, data_loader, pad_and_cut, BudgetPolicy, category_sampling

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
BATCH_SIZE_ = 32
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
        self.user_budget = user_budegt  # user budgets
        self.budget_blocks_action_memory = None
        self.item_action_memory = []
        self.budget_blocks = budget_blocks

        self.budget_policys = BudgetPolicy(input_dim=BUDGET_DIM, output_dim=BLOCK_NUM).to(device)
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
        # self.budget_policy_optim = torch.optim.Adam([{"params":net.parameters()} for net in self.budget_policys], lr=LR, weight_decay=0.05)
        self.budget_policy_optim = torch.optim.Adam(self.budget_policys.parameters(), lr=LR, weight_decay=0.05)
        self.budget_net_optim = torch.optim.Adam(self.budget_net.parameters(), lr=LR, weight_decay=0.05)
        self.item_policy_optim = torch.optim.Adam([{"params":net.parameters()} for net in self.item_policys], lr=LR, weight_decay=0.05)

    def genPolicys(self):
        # budget_policys = []
        item_policys = []

        # 按照BLOCK_NUM的大小生成budget范围选择策略网络
        # for i in range(math.ceil(math.log(BLOCK_NUM, 2))):
        #     budget_policys.append(Policy(BUDGET_DIM).to(device))

        # 按照BLOCK_SIZE大小生成item选择策略网络
        for i in range(math.ceil(math.log(BLOCK_SIZE, 2))):
            item_policys.append(Policy(BUDGET_DIM).to(device))

        # self.budget_policys = budget_policys
        self.item_policys = item_policys

    def choose_action(self, cur_item_ids, golden_item_ids, user_ids):
        """
        :param cur_item_id: [batch_size]
        :param gold_item_id: [batch_size]
        :param user_id: [batch_size]
        :return:
        """
        # 估计用户的预算  [budget_dim]
        if len(self.item_action_memory) != 0:
            pre_item_ids = self.item_action_memory[-1]
        else:
            pre_item_ids = None

        #[batch_size, budget_dim]
        budgets = self.budget_net(pre_item_ids, cur_item_ids, user_ids)

        # 根据budget选择budget_block
        # [batch_size, block_num]
        budgets_select_time = time.time()
        budget_blocks_dists = self.budget_policys(budgets)
        selected_budget_block_ids = list(map(category_sampling,budget_blocks_dists))
        selected_budget_block_probs =[budget_blocks_dists[key][value] for key, value in enumerate(selected_budget_block_ids)]
        print(f"budget_blocks_select_time:{time.time()-budgets_select_time}\n")
        # 根据budgets选择item
        select_item_time_start = time.time()
        selected_item_ids =[]
        selected_item_probs = []
        for i in range(BATCH_SIZE_):
            selected_item_id, selected_item_prob = action_select(budgets[i],
                                                                 len(self.budget_blocks[selected_budget_block_ids[i]]),
                                                                 self.item_policys)
            selected_item_ids.append(selected_item_id)
            selected_item_probs.append(selected_item_prob)
        select_item_time_end = time.time()
        print(f"select_item_time: {select_item_time_end-select_item_time_start}\n")


        self.item_action_memory.append(selected_item_ids)

        # 计算item_action的概率分布
        cal_item_action_dists_time_start = time.time()
        item_action_dists = []
        for i in range(BATCH_SIZE_):
            item_action_dist = action_distribution(budgets[i], len(self.budget_blocks[selected_budget_block_ids[i]]), self.item_policys, prob=selected_budget_block_probs[i])
            item_action_dist = torch.concat(item_action_dist, dim=0).unsqueeze(dim=0)
            if item_action_dist.shape[1]<BLOCK_SIZE:
                item_action_dist = torch.concat([item_action_dist, torch.zeros([1,BLOCK_SIZE-item_action_dist.shape[1]]).to(device)], dim=1)
            item_action_dists.append(item_action_dist)
        item_action_dists = torch.concat(item_action_dists, dim=0)
        cal_item_action_dists_time_end = time.time()
        print(f"cal_item_action_dists_time: {cal_item_action_dists_time_end-cal_item_action_dists_time_start}\n")
        # 根据预测出来item和gold_item的相似度(这里选择的是价格差)设计reward。
        reward = 0
        selected_item_prices = [self.budget_blocks[selected_budget_block_ids[i]][selected_item_ids[i]][1] for i in range(BATCH_SIZE)]
        user_budgets = [self.user_budget[user_id] for user_id in user_ids]
        for i in range(BATCH_SIZE_):
            if self.budget_blocks[selected_budget_block_ids[i]][selected_item_ids[i]][0] == golden_item_ids[i]:
                reward += 0.1
            elif selected_item_prices[i] < user_budgets[i][1] and selected_item_prices[i]>user_budgets[i][0]:
                reward += 0.05
            else:
                reward -= 0.05

        return budgets, item_action_dists, selected_item_ids, reward

    def learn(self,item_id, item_dist, td_error):

        # item_dist = item_dist.squeeze()
        # 当前action
        selected_item_id = torch.LongTensor(item_id).to(device)
        # item_dist = torch.concat(item_dist, dim=0)

        l = torch.nn.NLLLoss()
        log_softmax_input = torch.log(item_dist)
        neg_log_prob = l(log_softmax_input, selected_item_id)

        loss_a = torch.sum(-neg_log_prob * td_error)

        # 梯度归零
        self.budget_net_optim.zero_grad()
        self.budget_policy_optim.zero_grad()
        self.item_policy_optim.zero_grad()

        # 计算梯度
        loss_a.backward()

        # 梯度裁剪防止梯度爆炸
        nn.utils.clip_grad_norm_(self.budget_net.parameters(), 1)
        nn.utils.clip_grad_norm_(self.budget_policys.parameters(), 1)
        for p in self.item_policys:
            nn.utils.clip_grad_norm_(p.parameters(), 1)

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

train_data, eval_data = data_split(dp.seq, train_rate=0.8)

for epoch in range(EPOCH):
    # 设置模型为训练状态
    start = time.time()
    for p in actor.item_policys:
        p.train()
    actor.budget_policys.train()
    actor.budget_net.train()
    critic.network.train()

    for uids, seqs in data_loader(train_data, BATCH_SIZE):
        batch_time = time.time()
        BATCH_SIZE_ = len(uids)
        # 清空memory
        actor.item_action_memory = []
        # 裁剪seq
        min_length = min([len(s) for s in seqs])
        seqs = pad_and_cut(np.array(seqs), min_length)
        input_item_ids = seqs[:, :-1]
        golden_item_ids = seqs[:, 1:]
        for i in range(min_length-1):
            inputs = input_item_ids[:, i]
            golden = golden_item_ids[:, i]

            action_choose_time_start = time.time()
            budgets, item_action_dists, selected_item_ids, reward = actor.choose_action(cur_item_ids=inputs,
                                                                             golden_item_ids=golden,
                                                                             user_ids=uids)
            actopm_choose_time_end = time.time()
            print(f"action_choose_time{time.time()-action_choose_time_start}\n")

            with torch.no_grad():
                next_budgets = actor.budget_net(selected_item_ids, golden, uids)

            network_update_time = time.time()
            td_error = critic.train_Q_network(
                budgets.clone().detach(),
                reward,
                next_budgets)

            actor.learn(selected_item_ids, item_action_dists, td_error)
            print(f"network_update_time: {time.time()-network_update_time}\n")
            # true_gradient = grad[logPi(a|s) * td_error]
            # 然后根据前面学到的V（s）值，训练actor，以更好地采样动作
        print(f"processed one batch time: {time.time()-batch_time}\n")
    total_reward = 0
    # 设置模型为训练状态
    for p in actor.item_policys:
        p.eval()
    actor.budget_policys.eval()
    actor.budget_net.eval()
    critic.network.eval()
    # 取消梯度跟踪
    with torch.no_grad():
        for uids, seqs in data_loader(eval_data, BATCH_SIZE):
            BATCH_SIZE_ = len(uids)
            # 清空memory
            actor.item_action_memory = []
            # 裁剪seq
            min_length = min([len(s) for s in seqs])
            seqs = pad_and_cut(np.array(seqs), min_length)
            input_item_ids = seqs[:, :-1]
            golden_item_ids = seqs[:, 1:]
            for i in range(min_length - 1):
                inputs = input_item_ids[:, i]
                golden = golden_item_ids[:, i]

                budgets, item_action_dists, selected_item_ids, reward = actor.choose_action(cur_item_ids=inputs,
                                                                                            golden_item_ids=golden,
                                                                                            user_ids=uids)

                total_reward+=reward
                # true_gradient = grad[logPi(a|s) * td_error]
                # 然后根据前面学到的V（s）值，训练actor，以更好地采样动作
                print(total_reward)
    end = time.time()
    print(f"epoch:{epoch}  total reward:{total_reward}  time:{round((end - start) / 60, 2)}")




# # 训练
# for epoch in range(EPOCH):
#     # 设置模型为训练状态
#     for p in actor.item_policys:
#         p.train()
#     for p in actor.budget_policys:
#         p.train()
#     actor.budget_net.train()
#     critic.network.train()
#
#     start = time.time()
#     for uid, eps in list(train_data.items())[:100]:
#         # 清空memory
#         actor.item_action_memory = []
#         for i in range(len(eps)-1):
#             input_item_id = eps[i][0]
#             gold_item_id = eps[i+1][0]
#
#             cur_state, item_dist, item_id_pred, reward = actor.choose_action(cur_item_id=input_item_id,
#                                                                              gold_item_id=gold_item_id,
#                                                                              user_id=uid)
#             with torch.no_grad():
#                 next_state = actor.budget_net(item_id_pred,gold_item_id,uid)
#
#             td_error = critic.train_Q_network(
#                 cur_state.clone().detach(),
#                 reward,
#                 next_state)
#
#             actor.learn(item_id_pred, item_dist, td_error)
#             # print("end")
#             # true_gradient = grad[logPi(a|s) * td_error]
#             # 然后根据前面学到的V（s）值，训练actor，以更好地采样动作
#     total_reward = 0
#     # 设置模型为训练状态
#     for p in actor.item_policys:
#         p.eval()
#     for p in actor.budget_policys:
#         p.eval()
#     # actor.budget_policys.eval()
#     actor.budget_net.eval()
#     critic.network.eval()
#     # 取消梯度跟踪
#     with torch.no_grad():
#         for uid, eps in list(eval_data.items())[:100]:
#             for i in range(len(eps) - 1):  # car_id, 是否中标, 出价金额, 出价时间
#                 input_item_id = eps[i][0]
#                 gold_item_id = eps[i + 1][0]
#
#                 _, _, _, reward = actor.choose_action(cur_item_id=input_item_id,
#                                                       gold_item_id=gold_item_id,
#                                                       user_id=uid)
#                 total_reward += reward
#     end = time.time()
#     print(f"epoch:{epoch}  total reward:{total_reward}  time:{round((end-start)/60, 2)}" )