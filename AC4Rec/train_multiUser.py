import datetime
import pickle
import heapq
import math
import time
import json
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
from AC4Rec.utils_multiUser import DataPre, Voc, data_split, BlockPolicy, ItemPolicy, BudgetNet, item_split, action_select, action_distribution, data_loader, pad_and_cut, BudgetPolicy, category_sampling, mrr, hr, Metrics_map, ndcg
from sklearn.metrics._ranking import label_ranking_average_precision_score


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



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

TOPN = 10
DATA_PATH = '../dataset/filtered_data.csv'
BATCH_SIZE = 128
GAMMA = 0.9
LR = 1e-3
# ITEM_DIM == USER_DIM == BUDGET_DIM
ITEM_DIM = 512
USER_DIM = 512
BUDGET_DIM = 512
BLOCK_DIM = 512
GRU_HIDDEN_SIZE = 512
EPOCH = 400
BLOCK_SIZE = 256
BLOCK_NUM = math.ceil(dp.itemVoc.num_words/BLOCK_SIZE)
TAIL_BLOCK_SIZE = dp.itemVoc.num_words % BLOCK_SIZE

print("split budgets...")
item_price = list(enumerate(dp.itemPrice))
item_price.sort(key=lambda x: x[1])
item_price.extend([(-1, float("inf"))]*(BLOCK_SIZE-TAIL_BLOCK_SIZE))
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

        # 根据state生成budgets
        self.budget_net = BudgetNet(
            user_num=self.user_num,
            user_dim=user_dim,
            item_num=item_num,
            item_dim=item_dim,
            budget_dim=BUDGET_DIM,
            gru_hidden_size=GRU_HIDDEN_SIZE).to(device)

        self.blockPolicy = BlockPolicy(input_dim=BUDGET_DIM, output_dim=BLOCK_NUM).to(device)

        self.itemPolicy = ItemPolicy(BLOCK_DIM, BLOCK_SIZE, BLOCK_NUM).to(device)

        # 优化器
        self.budget_net_optim = torch.optim.Adam(self.budget_net.parameters(), lr=LR)
        self.blockPolicy_optim = torch.optim.Adam(self.blockPolicy.parameters(), lr=LR)
        self.itemPolicy_optim = torch.optim.Adam(self.itemPolicy.parameters(), lr=LR)

    def genPolicys(self):
        # budget_policys = []
        item_policys = []

        # 按照BLOCK_NUM的大小生成budget范围选择策略网络
        # for i in range(math.ceil(math.log(BLOCK_NUM, 2))):
        #     budget_policys.append(Policy(BUDGET_DIM).to(device))

        # 按照BLOCK_SIZE大小生成item选择策略网络
        for i in range(math.ceil(math.log(BLOCK_SIZE, 2))):
            # item_policys.append(Policy(BUDGET_DIM).to(device))
            pass

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
        #[batch_size, budget_dim]
        budgets = self.budget_net(cur_item_ids, user_ids)

        # 根据budget选择budget_block
        # [batch_size, block_num]
        blocks_dists = self.blockPolicy(budgets)
        # for p in self.blockPolicy.parameters():
        #     print(p)
        # print(blocks_dists)
        selected_block_ids = list(map(category_sampling, blocks_dists))
        selected_block_probs =[blocks_dists[key][value].unsqueeze(dim=0) for key, value in enumerate(selected_block_ids)]

        # 根据budgets选择item
        item_dists = self.itemPolicy(budgets, selected_block_ids, BLOCK_NUM, BLOCK_SIZE, TAIL_BLOCK_SIZE)
        # 计算item_action的概率分布
        selected_block_probs = torch.concat(selected_block_probs,dim=0).unsqueeze(dim=1)
        selected_item_ids = list(map(category_sampling, item_dists))
        item_dists = selected_block_probs * item_dists
        # 根据预测出来item和gold_item的相似度(这里选择的是价格差)设计reward。
        reward = 0
        selected_item_prices = [self.budget_blocks[selected_block_ids[i]][selected_item_ids[i]][1] for i in range(len(selected_item_ids))]
        user_budgets = [self.user_budget[user_id] for user_id in user_ids]
        for i in range(len(selected_item_ids)):
            if self.budget_blocks[selected_block_ids[i]][selected_item_ids[i]][0] == golden_item_ids[i]:
                reward += 0.1
            elif selected_item_prices[i] < user_budgets[i][1] and selected_item_prices[i]>user_budgets[i][0]:
                reward += 0.05
            else:
                reward -= 0.05
        return budgets, item_dists, selected_item_ids, reward, selected_block_ids

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
        self.blockPolicy_optim.zero_grad()
        self.itemPolicy_optim.zero_grad()

        # 计算梯度
        loss_a.backward()

        # 梯度裁剪防止梯度爆炸
        nn.utils.clip_grad_norm_(self.budget_net.parameters(), 0.1)
        nn.utils.clip_grad_norm_(self.blockPolicy.parameters(), 0.1)
        nn.utils.clip_grad_norm_(self.itemPolicy.parameters(), 0.1)

        # 更新参数
        self.budget_net_optim.step()
        self.blockPolicy_optim.step()
        self.itemPolicy_optim.step()

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
            nn.Linear(int(0.5 * state_dim), 1),
        )

    def forward(self, x):
        out = self.fc(x)
        return torch.sum(out)

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
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)

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

train_data, valid_data, test_data = data_split(dp.seq)

for epoch in range(EPOCH):
    epoch_time = time.time()
    # 设置模型为训练状态
    actor.budget_net.train()
    actor.blockPolicy.train()
    actor.itemPolicy.train()
    critic.network.train()
    for uids, seqs in data_loader(train_data, BATCH_SIZE):

        # BATCH_SIZE_ = len(uids)
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

            # action_choose_time_start = time.time()
            budgets, item_action_dists, selected_item_ids, reward, selected_block_ids = actor.choose_action(cur_item_ids=inputs,
                                                                             golden_item_ids=golden,
                                                                             user_ids=uids)

            with torch.no_grad():
                next_budgets = actor.budget_net(selected_item_ids, uids)

            # network_update_time = time.time()
            td_error = critic.train_Q_network(
                budgets.clone().detach(),
                reward,
                next_budgets)

            actor.learn(selected_item_ids, item_action_dists, td_error)
            # print(f"network_update_time: {time.time()-network_update_time}\n")
            # true_gradient = grad[logPi(a|s) * td_error]
            # 然后根据前面学到的V（s）值，训练actor，以更好地采样动作
    total_reward = 0
    # 设置模型为训练状态
    actor.budget_net.eval()
    actor.blockPolicy.eval()
    actor.itemPolicy.eval()
    critic.network.eval()
    # 取消梯度跟踪
    with torch.no_grad():
        for uids, seqs in data_loader(valid_data, 8444):
            # 裁剪seq
            min_length = min([len(s) for s in seqs])
            seqs = pad_and_cut(np.array(seqs), min_length)
            input_item_ids = seqs[:, :-1]
            golden_item_ids = seqs[:, 1:]
            for i in range(min_length - 1):
                inputs = input_item_ids[:, i]
                golden = golden_item_ids[:, i]

                # action_choose_time_start = time.time()
                budgets, item_action_dists, selected_item_ids, reward, selected_block_ids = actor.choose_action(cur_item_ids=inputs,
                                                                                            golden_item_ids=golden,
                                                                                            user_ids=uids)
                # 评价指标
                mrr_v = mrr(golden, selected_block_ids, BLOCK_SIZE, item_action_dists, TOPN)
                hr_v = hr(golden, selected_block_ids, BLOCK_SIZE, item_action_dists, TOPN)
                map_v = Metrics_map(golden, selected_block_ids, BLOCK_SIZE, item_action_dists, TOPN)
                ndcg_v = ndcg(golden, selected_block_ids, BLOCK_SIZE, item_action_dists, TOPN)

                total_reward += reward
    print(f"epoch: {epoch}  total-reward: {round(total_reward, 2)}  mrr:{round(mrr_v, 2)}  hr:{round(hr_v, 2)}  map:{round(map_v, 2)}  ndcg:{round(ndcg_v, 2)}"
          f"  time: {round((time.time() - epoch_time) / 60, 2) }min\n")

