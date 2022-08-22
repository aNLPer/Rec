import sys
import csv
import math
import random
import copy
import argparse
import logging
import ast
import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Voc:
    def __init__(self, sentence=False):
        PAD_token = 0  # Used for padding short sentences
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD"}
        self.num_words = 1
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
        self.userVoc = Voc(data['商户ID'])
        self.itemVoc = Voc(data['车ID'])
        self.actionVoc = Voc(data['是否中标'])
        self.seq = {}  ##item_seq {uid: {[iid, action, value],[...]}}

        self._toSeq()

    def _toSeq(self):
        for index, row in self.data.iterrows():
            uid = self.userVoc.word2index[int(row['商户ID'])]
            iid = self.itemVoc.word2index[int(row['车ID'])]
            aid = self.actionVoc.word2index[int(row['是否中标'])]
            self.seq.setdefault(uid, [])
            self.seq[uid].append([iid, aid, int(row['出价金额']), int(row['出价时间'])])

            # making order meanful and thus can remove time_stamp
        for uid, items in self.seq.items():
            self.seq[uid] = sorted(items, key=lambda x: x[-1])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_seq):
        super().__init__()
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]

def collate_fn(data):
    dataset = [list(map(lambda x: x[0], seq)) for seq in data]  # x[0] stores itemID
    dataset.sort(key=lambda x: len(x), reverse=True)

    input_len = [len(seq) - 1 for seq in dataset]
    input_seq = [seq[:-1] if len(seq[:-1]) == input_len[0] else seq[:-1] + [0] * (input_len[0] - len(seq[:-1]))
                 for seq in dataset]
    output_seq = [seq[1:] if len(seq[1:]) == input_len[0] else seq[1:] + [0] * (input_len[0] - len(seq[1:]))
                  for seq in dataset]

    output_seq = [seq[1:] if len(seq[1:]) == input_len[0] else seq[1:] + [0] * (input_len[0] - len(seq[1:]))
                  for seq in dataset]

    return input_seq, output_seq, input_len


df = pd.read_csv('dataset/filtered_data.csv')
basetime = datetime.datetime.strptime(df['出价时间'].min(), '%Y-%m-%d')
# 将出价时间设置为与basetime的差
df['出价时间'] = df['出价时间'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d') - basetime).days)
dp = DataPre(df)

from torch.utils.data import DataLoader
from torch.utils.data import random_split


mydata = Dataset(list(map(lambda x: x[1][-6:], list(dp.seq.items())))) #last 5 items of each user

#split dataset
train_size = int(dp.userVoc.num_words * 0.8)
val_size = int(dp.userVoc.num_words * 0.05)
test_size = dp.userVoc.num_words - train_size - val_size - 1

train, val, test = random_split(mydata, [train_size, val_size, test_size]) #不重合


class GRU4Rec(nn.Module):
    def __init__(self, Ni, K, emb_reg=False):
        super(GRU4Rec, self).__init__()
        # torch.manual_seed(999)
        # voc_size
        self.Ni = Ni
        # emb_size
        self.K = K
        self.emb_reg = emb_reg

        self.i_em = nn.Embedding(Ni, K)
        self.i_em.weight.data.normal_(0, .1)
        # emb_size = hidden_size = K
        self.gru = nn.GRU(K, K, 1, batch_first=True)
        '''
        input_size, hidden_size, num_layers
        '''
        self.hid2voc = nn.Linear(K, Ni)

        self.ce = nn.CrossEntropyLoss()

    def forward(self, seq, seq_len):
        ems = self.i_em(torch.tensor(seq))  # [batch, token, K]
        # padded
        ems = pack_padded_sequence(ems, seq_len, batch_first=True)

        out, h = self.gru(ems)
        return self.hid2voc(out.data), self.hid2voc(h)


def evaluation(model, input_x, output_y, seq_len, K):
    mrr = 0.
    hit_n = [0] * K

    out, h = model(input_x, seq_len)

    _, indices = torch.sort(h, descending=True)
    indices = indices.squeeze(0)

    for i in range(indices.shape[0]):
        # if val_y[i].item() in indices[i].tolist():
        mrr += 1. / (1 + indices[i].tolist().index(output_y[i][-1]))
        for topn in range(K):
            if output_y[i][-1] in indices[i][:topn + 1].tolist():
                hit_n[topn] += 1

    return np.array(hit_n) * 1. / indices.shape[0], mrr / indices.shape[0]

# model = GRU4Rec(dp.itemVoc.num_words, 3).to(device)
# out, h = model(batch_x, batch_x_len)
# model.ce(out, pack_padded_sequence(torch.tensor(batch_y), batch_x_len, batch_first=True).data)


epoch_num = 20
batch_size = 500
hidden_size = 50
lr = 0.05

model = GRU4Rec(dp.itemVoc.num_words, hidden_size).to(device)
optimizer = optim.Adagrad(model.parameters(), lr)

val_data = DataLoader(val, batch_size=len(val), shuffle=False, collate_fn=collate_fn)
val_x, val_y, val_seq_len = iter(val_data).next()

test_data = DataLoader(test, batch_size=len(test), shuffle=False, collate_fn=collate_fn)
test_x, test_y, test_seq_len = iter(test_data).next()

for step in range(epoch_num):
    data_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    for i, min_batch in enumerate(data_loader):
        model.train()

        batch_x, batch_y, batch_x_len = min_batch
        out, h = model(batch_x, batch_x_len)

        loss = model.ce(out, pack_padded_sequence(torch.tensor(batch_y), batch_x_len, batch_first=True).data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item(), i, step)
        if (i + 1) % 10 == 0:
            model.eval()
            print(evaluation(model, val_x, val_y, val_seq_len, 5))
model.eval()
print(evaluation(model, test_x, test_y, test_seq_len, 10))


