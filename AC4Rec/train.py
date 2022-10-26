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
from AC4Rec.utils import DataPre, Voc, data_construction, BlockPolicy, ItemPolicy, ItemNet, item_split, action_select, action_distribution, data_loader, pad_and_cut, category_sampling, evaluate
from sklearn.metrics._ranking import label_ranking_average_precision_score
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup



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
BATCH_SIZE = 1
GAMMA = 0.8
LR = 1e-5
# ITEM_DIM == USER_DIM == BUDGET_DIM
ITEM_DIM = 200
BUDGET_DIM = 512
BLOCK_DIM = 521
GRU_HIDDEN_SIZE = 512
EPOCH = 100
BLOCK_SIZE = 256
BLOCK_NUM = math.ceil(dp.itemVoc.num_words/BLOCK_SIZE)
TAIL_BLOCK_SIZE = dp.itemVoc.num_words % BLOCK_SIZE

# print("split item to block...")
# item_price = list(enumerate(dp.itemPrice))# [(iid, price),...]
# item_price.sort(key=lambda x: x[1])
# item_price.extend([(-1, float("inf"))]*(BLOCK_SIZE-TAIL_BLOCK_SIZE))
# item_blocks = item_split(item_price, BLOCK_SIZE)
# dp.item_blocks = item_blocks

# iid2block = [0]*dp.itemVoc.num_words
# for block_num in range(len(item_blocks)):
#     for iid, _ in item_blocks[block_num]:
#         iid2block[iid] = block_num
# dp.iid2block = iid2block

class Actor(object):
    def __init__(self,item_num, item_dim, item_price, budget_blocks):
        self.item_num = item_num
        self.item_dim = item_dim
        self.item_price = item_price  # sorted [(item_id, item_price),(),......]
        self.budget_blocks = budget_blocks

        # 预测用户budgets
        self.item_net = ItemNet(
            item_num=item_num,
            item_dim=item_dim,
            block_num=BLOCK_NUM,
            block_size=BLOCK_SIZE,
            gru_hidden_size=GRU_HIDDEN_SIZE).to(device)

        # 优化器
        self.item_net_optim = torch.optim.Adam(self.item_net.parameters(), lr=LR)

        self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.item_net_optim,
                                                                            num_warmup_steps=100,
                                                                            num_training_steps=8000,
                                                                            num_cycles=1)

    def choose_action(self, cur_item_id, golden_item_ids, block_ids):
        """
        :param cur_item_id: [batch_size]
        :param gold_item_id: [batch_size]
        :param user_id: [batch_size]
        :return:
        """
        # 计算item分布
        # [seq_len, 179], [seq_len, 256], [seq_len, 512]
        block_dist, item_dist, next_state = self.item_net(cur_item_id)
        # dist = torch.matmul(block_dist.transpose(dim0=1, dim1=2), item_dist)

        # 采样block和item
        # [seq_len]
        selected_block_ids = category_sampling(block_dist)
        # [seq_len]
        selected_item_in_block_ids = category_sampling(item_dist)
        # 计算item分布
        selected_block_prob = torch.concat([block_dist[i][selected_block_ids[i]].unsqueeze(dim=0) for i in range(selected_block_ids.shape[0])]).unsqueeze(dim=1)
        # selected_item_in_block_porb = torch.concat([item_dist[i][selected_item_in_block_ids[i]].unsqueeze(dim=0) for i in range( selected_item_in_block_ids.shape[0])]).unsqueeze(dim=1)
        selected_item_dist = selected_block_prob.mul(item_dist)

        # reward
        rewards = []
        selected_item_ids = []
        for selected_block_id, selected_item_in_block_id in zip(selected_block_ids,selected_item_in_block_ids):
            reward = 0
            selected_item_id = dp.item_blocks[selected_block_id][selected_item_in_block_id][0]
            selected_item_ids.append(selected_item_id)
            if selected_block_id in block_ids:
                reward += 0.1
            if selected_item_id in golden_item_ids:
                reward += 0.1
            rewards.append(reward)

        return next_state, selected_item_in_block_ids, selected_item_dist, rewards, block_dist, selected_item_ids

    #next_state, selected_item_in_block_id, selected_item_dist, reward, selected_item_id

    def learn(self,selected_item_dist, selected_item_in_block_id, td_error, block_dist, golden_block_id):

        # 损失含函数
        l = torch.nn.NLLLoss()
        ce = torch.nn.CrossEntropyLoss()
        log_softmax_input = torch.log(selected_item_dist)
        neg_log_prob = l(log_softmax_input, torch.LongTensor(selected_item_in_block_id))
        loss_a = torch.sum(-neg_log_prob * td_error)
        loss_b = ce(block_dist, torch.tensor(golden_block_id, dtype=torch.long))
        loss = loss_a+loss_b
        # 梯度归零
        self.item_net_optim.zero_grad()

        # 计算梯度
        loss.backward()

        # 梯度裁剪防止梯度爆炸
        nn.utils.clip_grad_norm_(self.item_net.parameters(), 0.1)

        # 更新参数
        self.item_net_optim.step()

        # 更新学习率
        self.scheduler.step()

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
        self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
                                                                       num_warmup_steps=100,
                                                                       num_training_steps=8000,
                                                                       num_cycles=1)

    def train_Q_network(self, state, reward, next_state):
        # 类似于DQN的5.4，不过这里没有用fixed network，experience relay的机制

        s, s_ = torch.FloatTensor(state), torch.FloatTensor(next_state)
        # 当前状态，执行了action之后的状态

        v = self.network(s)  # v(s)
        v_ = self.network(s_)  # v(s')

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

        # 更新学习率
        self.scheduler.step()

        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v
        # 表示不把相应的梯度传到actor中（actor和critic是独立训练的）

        return td_error

actor = Actor(item_num=dp.itemVoc.num_words,
              item_dim=ITEM_DIM,
              item_price=dp.itemPrice,
              budget_blocks=dp.item_blocks)

critic = Critic(input_dim=BUDGET_DIM)

train_data, valid_data, test_data = data_construction(dp)



for epoch in range(EPOCH):
    print(f"epoch {epoch} :")
    epoch_time = time.time()

    # 设置模型为训练状态
    actor.item_net.train()
    critic.network.train()

    train_rec_count = 0
    train_total_reward = 0.0

    for seqs, blocks in data_loader(train_data, dp.iid2block, BATCH_SIZE): # 426435
        # 初始化hidden_state
        actor.item_net.init_hidden = torch.zeros(size=(1, GRU_HIDDEN_SIZE),dtype=torch.float32)
        # # 裁剪seq
        # min_length = min([len(s) for s in seqs])
        # train_rec_count+=min_length
        # seqs = pad_and_cut(np.array(seqs), min_length)
        # blocks = pad_and_cut(np.array(blocks), min_length)
        # 模型输入x_t
        input_item_ids = seqs[:-1]
        # 监督输出x_t+1
        golden_item_ids = seqs[1:]
        # 输出所在的block
        golden_item_block = blocks[1:]

        # for i in range(min_length-1):
        # input_iid = input_item_ids[:, i]
        # golden_iids = golden_item_ids[:, i:]
        # block_ids = golden_item_block[:, i:]
        # cur_state
        init_state = actor.item_net.init_hidden
        # action_choose_time_start = time.time()
        next_state, selected_item_in_block_ids, selected_item_dist, rewards, block_dist, _ = actor.choose_action(cur_item_id=input_item_ids,
                                                                                                golden_item_ids=golden_item_ids,
                                                                                                block_ids = golden_item_block)
        state_list = torch.concat([init_state, next_state], dim=0).tolist()
        train_total_reward+=sum(rewards)
        # input_iid = selected_item_id
        # network_update_time = time.time()
        td_errors = []
        for i in range(2):#len(state_list)-1
            cur_state = state_list[i]
            next_state = state_list[i+1]
            td_error = critic.train_Q_network(
                cur_state,
                rewards[i],
                next_state)
            td_errors.append(td_error)
        actor.learn(selected_item_dist, selected_item_in_block_ids, sum(td_errors), block_dist, golden_item_block)
        # print(f"network_update_time: {time.time()-network_update_time}\n")
        # true_gradient = grad[logPi(a|s) * td_error]
        # 然后根据前面学到的V（s）值，训练actor，以更好地采样动作
    # 评价模型
    # print(f"training-reward:{train_total_reward}")
    # hr, map_, mrr = evaluate(goldens, selected_blocks, item_dists, dp.item_blocks, TOPN=TOPN)
    # print(f"train_total-reward: {round(train_total_reword, 2)}  train_mrr:{round(mrr, 2)}  train_hr:{round(hr, 2)}  train_map:{round(map_, 2)}  train_block_acc: {selected_block_num/train_rec_count}")

    valid_total_reward = 0
    # 设置模型为训练状态
    actor.item_net.eval()
    # 取消梯度跟踪
    with torch.no_grad():
        topn_rec = 0
        rec_count = 0
        for seqs, blocks in data_loader(valid_data, dp.iid2block, BATCH_SIZE):  # 426435
            rec_count+=1
            # 初始化hidden_state
            actor.item_net.init_hidden = torch.zeros(size=(1, GRU_HIDDEN_SIZE), dtype=torch.float32)
            # # 裁剪seq
            # min_length = min([len(s) for s in seqs])
            # train_rec_count+=min_length
            # seqs = pad_and_cut(np.array(seqs), min_length)
            # blocks = pad_and_cut(np.array(blocks), min_length)
            # 模型输入x_t
            input_item_ids = seqs[:-1]
            # 监督输出x_t+1
            golden_item_ids = seqs[1:]
            # 输出所在的block
            golden_item_block = blocks[1:]

            init_state = actor.item_net.init_hidden
            # action_choose_time_start = time.time()
            for _ in range(TOPN):
                next_state, selected_item_in_block_ids, selected_item_dist, rewards, block_dist, selected_item_ids = actor.choose_action(
                                                                                                        cur_item_id=input_item_ids,
                                                                                                        golden_item_ids=golden_item_ids,
                                                                                                        block_ids=golden_item_block)
                if selected_item_ids[-1] in golden_item_ids:
                    topn_rec+=1
                valid_total_reward += sum(rewards)
                # input_iid = selected_item_id

    # 评价模型
    print(f"topn_hit: {topn_rec/rec_count}")
    # hr, map_, mrr = evaluate(goldens, selected_blocks, item_dists, dp.item_blocks, TOPN=TOPN)
    # print(f"valid_total-reward: {round(valid_total_reward, 2)} valid_mrr:{round(mrr, 2)}  valid_hr:{round(hr, 2)}  valid_map:{round(map_, 2)} "
    #       f"valid_block_acc: {round(selected_block_num/valid_rec_count, 2)} \ntime: {round((time.time() - epoch_time) / 60, 2) }min\n")

