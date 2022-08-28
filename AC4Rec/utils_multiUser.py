import numpy as np
import torch.nn as nn
import torch
import math
import heapq
import random
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Voc:
    def __init__(self, sentence=False):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0
        if type(sentence) != type(False):
            self.addSentence(sentence)

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

class DataPre:
    def __init__(self, data):
        self.data = data
        self.itemPrice = None  # item 平均出价金额作为item的price
        self.userBudgets = None  # 根据用户交互的item的平均price作为用户的budget
        self.userVoc = Voc(data['商户ID'])
        self.itemVoc = Voc(data['车ID'])
        self.actionVoc = Voc(data['是否中标'])
        self.item_freq = {} # {item_id: frequence}
        self.seq = {}  # item_seq {uid: {[iid, action, value],[...]}}
        self._toSeq()
        self._getItemPrice()
        self._getUserBudgets()

    def _toSeq(self):
        for index, row in self.data.iterrows():
            uid = self.userVoc.word2index[int(row['商户ID'])]
            iid = self.itemVoc.word2index[int(row['车ID'])]
            aid = self.actionVoc.word2index[int(row['是否中标'])]
            self.seq.setdefault(uid, [])
            self.item_freq.setdefault(iid, 0)
            self.seq[uid].append([iid, aid, round(float(row['出价金额']), 2), int(row['出价时间'])])
            self.item_freq[iid] += 1

            # making order meanful and thus can remove time_stamp
        for uid, items in self.seq.items():
            self.seq[uid] = sorted(items, key=lambda x: x[-1])

    def _getItemPrice(self):
        itemPrice = [0]*self.itemVoc.num_words
        for word in self.itemVoc.word2index.keys():
            temp = self.data[self.data["车ID"] == word]
            if temp.count()['车ID']>3:
                bid_mean =(temp["出价金额"].sum() - (temp["出价金额"].max() + temp["出价金额"].min())) / (temp.count()['车ID'] - 2)
            else:
                bid_mean = temp["出价金额"].mean()
            item_idx = self.itemVoc.word2index[word]
            itemPrice[item_idx] = int(bid_mean)
        self.itemPrice = itemPrice

    def _getUserBudgets(self):
        userBudgets = [0] * self.userVoc.num_words
        for key, values in self.seq.items():
            provide_prices = [p[2] for p in values]
            provide_prices.remove(max(provide_prices))
            provide_prices.remove(min(provide_prices))
            userBudgets[key] = (min(provide_prices), max(provide_prices))
            # if len(values) > 3:
            #     userBudgets[key] = int((sum(provide_prices)-max(provide_prices)-min(provide_prices))/(len(provide_prices)-2))
            # else:
            #     userBudgets[key] = int(sum(provide_prices)/len(provide_prices))
        self.userBudgets = userBudgets

class BudgetPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BudgetPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softmax()
        )
    def forward(self, input):
        """
        根据当前状态
        :return: [0~1]
        """
        out = self.fc(input)
        return out

class Policy(nn.Module):
    def __init__(self, input_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, int(0.5 * input_dim)),
            nn.ReLU(),
            nn.Linear(int(0.5 * input_dim), 2),
            nn.Softmax()
        )
    def forward(self, input):
        """
        根据当前状态
        :return: [0~1]
        """
        out = self.fc(input)
        return out
# 预测用户的budget
class BudgetNet(nn.Module):
    # 以用户embedding初始化h_0
    # 以上一时刻的item embeddin作为输入
    # 预测下一时刻的budget
    def __init__(self, user_num, user_dim, item_num, item_dim, budget_dim, gru_hidden_size):
        super(BudgetNet, self).__init__()
        # item嵌入矩阵
        self.item_em = nn.Embedding(item_num, item_dim).to(device)

        # user 嵌入矩阵item
        self.user_em = nn.Embedding(user_num, user_dim).to(device)

        self.gru = nn.GRUCell(input_size=item_dim, hidden_size=gru_hidden_size)

        # self.gru = nn.GRU(input_size=item_dim, hidden_size=gru_hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, int(0.5*gru_hidden_size)),
            nn.Linear(int(0.5*gru_hidden_size), budget_dim)
        )

    def forward(self, pre_item_id, cur_item_id, user_id):
        """
        """
        # [batch_size, user_dim]
        user_em = self.user_em(torch.LongTensor(user_id).to(device))
        # [batch_size, item_dim]
        cur_item_em = self.item_em(torch.LongTensor(cur_item_id).to(device))

        if pre_item_id is not None:
            # [batch_size, item_dim]
            pre_item_em = self.item_em(torch.LongTensor(pre_item_id).to(device))
            item_em = cur_item_em + pre_item_em
        else:
            # [batch_size, item_dim]
            item_em = cur_item_em
        # 初始化隐藏状态
        # [batch_size, gru_hidden_size]
        out = self.gru(item_em)
        # [batch_size, budget_dim]
        budget_pred = self.fc(out)
        # [batch_size, budget_dim]
        return budget_pred+user_em

def item_split(items_price, step):
    num_items = len(items_price)
    indices = list(range(num_items))
    splited_item = []
    for i in range(0, num_items, step):
        ids = indices[i: min(i + step, num_items)]
        splited_item.append([items_price[i] for i in ids])
    return splited_item

def data_split(data, train_rate=0.6):
    train_data = {}
    valid_data = {}
    for key, value in data.items():
        if len(value)*train_rate < 2 or len(value)*(1-train_rate) < 2:
            pass
        train_data[key] = value[:int(len(value)*train_rate)]
        valid_data[key] = value[int(len(value)*train_rate):]
    return train_data, valid_data

# 选择 action 计算对应的概率
def action_select(state, action_num, policys):
    directs = [p(state).squeeze() for p in policys]  # 计算policy的选择概率
    # print(f"action_select: {directs}")
    p = 1
    left = 0
    right = action_num - 1
    mid = int((left + right) / 2)
    for i in range(len(directs)):
        max_idx = category_sampling(directs[i])
        # max_idx = torch.argmax(directs[i]).item()
        p = p * directs[i][max_idx]
        if max_idx == 1:  # 向右
            left = mid + 1
        else:  # 向左
            right = mid
        mid = int((left + right) / 2)

        if left == right:  # 搜索到
            return left, p  # 返回选择的action以及对应的概率

def action_distribution(state, action_num, policys, prob = None):
    """
    :param budget: budget状态向量
    :param item_num: 选择的budget_block对应的item_num
    :param item_polocys: 选择策略
    :return:
    """
    directs = [p(state).squeeze() for p in policys]  # 计算policy的选择概率
    # print(f"action_distribution: {directs}")
    item_in_block_dist = [0] * action_num
    for target in range(action_num):
        if prob is None:
            p = 1
        else:
            p = prob
        left = 0
        right = action_num - 1
        mid = int((left + right) / 2)
        layer = 0
        while 1:
            if target <= mid:  # 向左
                right = mid
                p = p * directs[layer][0]
            else:  # 向右
                left = mid + 1
                p = p * directs[layer][1]

            mid = int((left + right) / 2)

            if left == target and right == target:
                item_in_block_dist[target] = p.unsqueeze(dim=0)
                break
            else:
                layer += 1  # 开始下层搜索

    return item_in_block_dist

def data_loader(data, batch_size):
    """
    :param data: dict
    :param batch_size: batch_size
    :return:
    """
    # tuple:[8444]
    sorted_data = sorted(data.items(), key=lambda x: len(x[1]))
    uids = []
    seqs = []
    for uid, s in sorted_data:
        uids.append(uid)
        seqs.append([item[0] for item in s])
    num_examples = len(seqs)
    indices = list(range(num_examples))
    for i in range(0, num_examples, batch_size):
        ids = indices[i: min(i + batch_size, num_examples)]
        # 最后⼀次可能不⾜⼀个batch
        yield [uids[j] for j in ids], [seqs[j] for j in ids]

def category_sampling(prob):
    m = Categorical(prob)
    return m.sample().item()

def pad_and_cut(data, length):
    """填充或截二维维numpy到固定的长度"""
    # 将2维ndarray填充和截断到固定长度
    n = len(data)
    for i in range(n):
        if len(data[i]) < length:
            # 进行填充
            data[i] = np.pad(data[i], pad_width=(0,length-len(data[i])))
        if len(data[i]) > length:
            # 进行截断
            data[i] = data[i][:length]
    # 转化为np.array()形式
    new_data = np.array(data.tolist())
    return new_data


if __name__=="__main__":
    a = [1,3,2]
    provide_prices = [1,3,2]
    provide_prices.remove(max(provide_prices))
    provide_prices.remove(min(provide_prices))
    print(provide_prices)
