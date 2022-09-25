import numpy as np
import torch.nn as nn
import torch
import math
import heapq
import random
from torch.distributions import Categorical

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
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
        self.userPreference = {} # {uid:[itemid]}

        self._toSeq()
        self._getItemPrice()
        self._getUserBudgets()
        self._getUserPreference()

    def _toSeq(self):
        # making user interaction order
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
        # making item2price
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
        # making user2budget
        userBudgets = [0] * self.userVoc.num_words
        for key, values in self.seq.items():
            # key：uid
            provide_prices = [p[2] for p in values]
            provide_prices.remove(max(provide_prices))
            provide_prices.remove(min(provide_prices))
            userBudgets[key] = (min(provide_prices), max(provide_prices))
            # if len(values) > 3:
            #     userBudgets[key] = int((sum(provide_prices)-max(provide_prices)-min(provide_prices))/(len(provide_prices)-2))
            # else:
            #     userBudgets[key] = int(sum(provide_prices)/len(provide_prices))
        self.userBudgets = userBudgets

    def _getUserPreference(self):
        # 获取用户交互item集合
        for key, values in self.seq.items():
            items = [item[0] for item in values]
            if key not in self.userPreference:
                self.userPreference[key] = items

class BudgetPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BudgetPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Softmax()
        )
    def forward(self, input):
        """
        根据当前状态
        :return: [0~1]
        """
        out = self.fc(input)
        return out

class ItemPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, block_num):
        super(ItemPolicy, self).__init__()
        self.block_em = nn.Embedding(block_num, input_dim).to(device)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    def forward(self, input, selected_block_ids, block_num, block_size, tail_block_size):
        """
        :param input: [batch_size, block_dim]
        :param selected_block_ids: [batch_size]
        :param block_num: [BLOCK_NUM]
        :param block_size: [BLOCK_SIZE]
        :param tail_block_size: [TAIL_BLOCK_SIZE]
        :return:
        """
        input = input + self.block_em(torch.tensor(selected_block_ids)).to(device)
        out = self.fc(input)
        # mask = [[0]*block_size for _ in range(len(selected_block_ids))]
        # for idx,id in enumerate(selected_block_ids):
        #     if id == block_num-1: # 选中的是最后一个block
        #         mask[idx] = [0 if i < tail_block_size else float("-inf") for i in range(block_size)]
        # mask = torch.tensor(mask, dtype=torch.float32)
        # out = mask + out
        out = torch.softmax(out, dim=1)
        return out

class BlockPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BlockPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.BatchNorm1d(output_dim),
            nn.ReLU(),
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
    def __init__(self,item_num, item_dim, budget_dim, gru_hidden_size):
        super(BudgetNet, self).__init__()
        # item嵌入矩阵
        self.item_em = nn.Embedding(item_num, item_dim).to(device)

        self.gru = nn.GRUCell(input_size=item_dim, hidden_size=gru_hidden_size)

        # self.gru = nn.GRU(input_size=item_dim, hidden_size=gru_hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, int(0.5*gru_hidden_size)),
            nn.Linear(int(0.5*gru_hidden_size), budget_dim)
        )

    def forward(self, cur_item_id):
        """
        """
        # [batch_size, user_dim]
        # user_em = self.user_em(torch.LongTensor(user_id).to(device))
        # [batch_size, item_dim]
        cur_item_em = self.item_em(torch.LongTensor(cur_item_id).to(device))

        # 初始化隐藏状态
        # [batch_size, gru_hidden_size]
        out = self.gru(cur_item_em)
        # [batch_size, budget_dim]
        budget_pred = self.fc(out)
        # [batch_size, budget_dim]
        return budget_pred

def item_split(items_price, step):
    num_items = len(items_price)
    indices = list(range(num_items))
    splited_item = []
    for i in range(0, num_items, step):
        ids = indices[i: min(i + step, num_items)]
        splited_item.append([items_price[i] for i in ids])
    return splited_item

def data_split(data, shuffe=False):
    train_data = []
    valid_data = []
    test_data = []
    data = list(data.items())
    data.sort(key=lambda x:len(x[1]), reverse=False)
    indices = list(range(len(data)))
    if shuffe:# 打乱数据
        np.random.shuffle(indices)
    # 划分数据
    train_indices = indices[:int(len(data)*0.6)]
    valid_indices = indices[int(len(data)*0.6):int(len(data)*0.8)]
    test_indices = indices[int(len(data) * 0.8):]
    train_data_ = [data[i] for i in train_indices]
    valid_data_ = [data[i] for i in valid_indices]
    test_data_ = [data[i] for i in test_indices]

    for sample in train_data_:
        item = [sample[0]]
        item.append([item[0] for item in sample[1]])
        train_data.append(item)

    for sample in valid_data_:
        item = [sample[0]]
        item.append([item[0] for item in sample[1]])
        valid_data.append(item)

    for sample in test_data_:
        item = [sample[0]]
        item.append([item[0] for item in sample[1]])
        test_data.append(item)

    return train_data, valid_data, test_data

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
    # tuple:[8444,]
    # sorted_data = sorted(data.items(), key=lambda x: len(x[1]))
    # uids = []
    # seqs = []
    # for uid, s in sorted_data:
    #     uids.append(uid)
    #     seqs.append([item[0] for item in s])
    num_examples = len(data)
    indices = list(range(num_examples))
    for i in range(0, num_examples, batch_size):
        ids = indices[i: min(i + batch_size, num_examples)]
        # 最后⼀次可能不⾜⼀个batch
        yield [data[j][0] for j in ids], [data[j][1] for j in ids]

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

def evaluate(goldens, selected_block_ids, selected_item_dist, budget_blocks, TOPN=200):
    """
    golden:[rec_count]
    selected_block_ids:[rec_count]
    selected_item_dist:[rec_count, block_size]
    """
    # 处理golden不在selected_block的情况
    # for i in range(len(goldens)):
    #     if goldens[i] in [item[0] for item in budget_blocks[selected_block_ids[i]]]:
    #         goldens[i] = 1024 # 设置一个大与block_size=256的值
    #     else:
    #         goldens[i] = goldens[i] % block_size
    # 计算 hit ratio
    h_count = 0
    for i in range(len(goldens)):
        selected_block = [item[0] for item in budget_blocks[selected_block_ids[i]]]
        if goldens[i] not in selected_block:
            continue
        # 推荐排序
        dist = list(enumerate(selected_item_dist[i]))
        dist.sort(key=lambda x: x[1], reverse=True)

        for j in range(TOPN):
            if goldens[i] == selected_block[dist[j][0]]:
                h_count += 1
    print(h_count)
    print(len(goldens))
    hr = round(h_count/len(goldens), 5)

    # 计算 map
    map_= 0.0
    aps = []
    for i in range(len(goldens)):
        selected_block = [item[0] for item in budget_blocks[selected_block_ids[i]]]
        if goldens[i] not in selected_block:
            continue
        # 排序
        dist = list(enumerate(selected_item_dist[i]))
        dist.sort(key=lambda x: x[1], reverse=True)
        # 计算每个用户ap之和(存在一个问题，测试集中用户只有一个item)
        ap = 0.
        for j in range(TOPN):
            if goldens[i] == selected_block[dist[j][0]]:
                ap += (1.0 / (j + 1)) / TOPN
        aps.append(ap)
    map_ = sum(aps) / len(aps)

    # 计算 mrr
    mrr = 0.0
    for i in range(len(goldens)):
        selected_block = [item[0] for item in budget_blocks[selected_block_ids[i]]]
        if goldens[i] not in selected_block:
            continue
        # 排序
        dist = list(enumerate(selected_item_dist[i]))
        dist.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [item[0] for item in dist]

        for j in range(TOPN):
            if goldens[i] == selected_block[dist[j][0]]:
                mrr += 1. / (1 + j)
    mrr = mrr/len(goldens)

    # 计算ndcg
    # dcg
    # dcgs = []
    # for i in range(len(goldens)):
    #
    #     dist = list(enumerate(selected_item_dist[i]))
    #     dist.sort(key=lambda x: x[1], reverse=True)
    #     sorted_indices = [item[0] for item in dist]
    #     d = 0.0
    #     for j in range(TOPN):
    #         if goldens[i] == sorted_indices[j]:
    #             d += 1.0/(math.log(j+1,2)+0.0001)
    #     dcgs.append(d)
    #
    # idcgs = [1.0] * len(goldens)
    # for i in range(len(goldens)):
    #     dist = list(enumerate(selected_item_dist[i]))
    #     dist.sort(key=lambda x: x[1], reverse=True)
    #     sorted_indices = [item[0] for item in dist]
    #     d = 0.0
    #     for j in range(TOPN):
    #         if goldens[i] == sorted_indices[j]:
    #             d += 1.0 / math.log(1 + 1, 2)
    #     dcgs.append(d)

    # ndcg = sum(np.array(dcgs)/np.array(idcgs))/len(goldens)


    return hr, map_, mrr

if __name__=="__main__":
    a = [(-1, float("inf"))]*10
    print(a)

