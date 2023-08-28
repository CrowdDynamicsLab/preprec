import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pdb
import math
from scipy.stats import rankdata, percentileofscore


# train/val/test data generation
def data_partition(fname, maxlen=None, augfulllen=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_dict = {}
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index start from 0, assume user interactions are sorted by time
    f = open(f"../data/{fname}_int2.csv", "r")
    for line in f:
        u, i, t, t2 = line.rstrip().split(",")
        u = int(u) + 1
        i = int(i) + 1
        t = int(t)
        t2 = int(t2)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append((i, t, t2))

    if maxlen is None:
        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = [User[user][-2]]
                user_test[user] = [User[user][-1]]
        return [user_train, user_valid, user_test, usernum, itemnum]

    newuser = usernum + 1
    for user in User:
        user_dict[user] = user
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            if augfulllen is not None:
                User[user] = User[user][-augfulllen - 2 :]
            numiter = math.ceil((nfeedback - maxlen - 2) / maxlen)
            if numiter > 0:
                left = (nfeedback - maxlen - 2) % maxlen
                if left > 2:
                    user_train[newuser] = User[user][:left]
                    user_dict[newuser] = user
                    newuser += 1
                for i in range(numiter - 1):
                    user_train[newuser] = User[user][
                        maxlen * i + left : maxlen * i + maxlen + left
                    ]
                    newuser += 1
                user_train[user] = User[user][maxlen * (numiter - 1) + left : -2]
            else:
                user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    return [user_train, user_valid, user_test, usernum, itemnum, user_dict]


def data_partition2(fname, wrong_num):
    add = 0 if wrong_num else 1
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(f"../data/{fname}_int2.csv", "r")
    for line in f:
        u, i = line.rstrip().split(",")[0:2]
        u = int(u) + add
        i = int(i) + add
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    return [user_train, user_valid, user_test, usernum, itemnum]


def data_partition3(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(f"../data/{fname}_int2.csv", "r")
    for line in f:
        u, i = line.rstrip().split(",")[0:2]
        u = int(u) + 1
        i = int(i) + 1
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            validi = np.random.choice(nfeedback)
            testi = np.random.choice(np.delete(np.arange(nfeedback, dtype=int), validi))
            if max(validi, testi) > len(User[user]) - 1:
                pdb.set_trace()
            user_valid[user] = [User[user][validi]]
            user_test[user] = [User[user][testi]]
            user_train[user] = np.delete(User[user], [validi, testi])
    return [user_train, user_valid, user_test, usernum, itemnum]
