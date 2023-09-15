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


def data_partition_wtime(fname, maxlen, augment=False, augfulllen=None):
    usernum = 0
    itemnum = 0
    User = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    user_dict = {}
    user_train = ({}, {}, {}, {})
    user_valid = ({}, {}, {}, {})
    user_test = ({}, {}, {}, {})
    # assume user/item index start from 0, assume user interactions are sorted by time
    f = open(f"../data/{fname}_intwtime.csv", "r")
    for line in f:
        u, i, t, t2, te = line.rstrip().split(",")
        u = int(u) + 1
        i = int(i) + 1
        t = int(t)
        t2 = int(t2)
        te = int(float(te))
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[0][u].append(i)
        User[1][u].append(t)
        User[2][u].append(t2)
        User[3][u].append(te)
        # User[u].append((i, t, t2))

    if not augment:
        for user in User[0]:
            nfeedback = len(User[0][user])
            uselen = min(maxlen+2, len(User[3][user]))
            temp = np.array(User[3][user][-uselen+1:]) - np.array(User[3][user][-uselen:-1])
            user_train[0][user] = User[0][user][-maxlen-3:-2]
            user_train[1][user] = User[1][user][-maxlen-3:-2]
            user_train[2][user] = User[2][user][-maxlen-3:-2]
            user_train[3][user] = list(np.argsort(temp[:-2][-maxlen:]) + 1)
            user_valid[0][user] = User[0][user][-2]
            user_valid[1][user] = User[1][user][-2]
            user_valid[2][user] = User[2][user][-2]
            user_valid[3][user] = list(np.argsort(temp[:-1][-maxlen:]) + 1)
            user_test[0][user] = User[0][user][-1]
            user_test[1][user] = User[1][user][-1]
            user_test[2][user] = User[2][user][-1]
            user_test[3][user] = list(np.argsort(temp[-maxlen:]) + 1)
            user_train[0][user] = list(np.zeros(1 + maxlen - len(user_train[0][user])).astype(int)) + user_train[0][user]
            user_train[1][user] = list(np.zeros(1 + maxlen - len(user_train[1][user]))) + user_train[1][user]
            user_train[2][user] = list(np.zeros(1 + maxlen - len(user_train[2][user]))) + user_train[2][user]
            user_train[3][user] = list(np.zeros(maxlen - len(user_train[3][user]))) + user_train[3][user]
            user_valid[3][user] = list(np.zeros(maxlen - len(user_valid[3][user]))) + user_valid[3][user]
            user_test[3][user] = list(np.zeros(maxlen - len(user_test[3][user]))) + user_test[3][user]
        return [user_train, user_valid, user_test, usernum, itemnum]


# train/val/test data generation
def data_partition(fname, maxlen, augment=False, augfulllen=None):
    usernum = 0
    itemnum = 0
    User = [defaultdict(list), defaultdict(list), defaultdict(list)]
    user_dict = {}
    user_train = ({}, {}, {})
    user_valid = ({}, {}, {})
    user_test = ({}, {}, {})
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
        User[0][u].append(i)
        User[1][u].append(t)
        User[2][u].append(t2)
        # User[u].append((i, t, t2))

    if not augment:
        for user in User[0]:
            nfeedback = len(User[0][user])
            if nfeedback < 3:
                user_train[0][user] = User[0][user]
                user_train[1][user] = User[1][user]
                user_train[2][user] = User[2][user]
                user_valid[0][user] = 0
                user_valid[1][user] = 0
                user_valid[2][user] = 0
                user_test[0][user] = 0
                user_test[1][user] = 0
                user_test[2][user] = 0
            else:
                user_train[0][user] = User[0][user][-maxlen-3:-2]
                user_train[1][user] = User[1][user][-maxlen-3:-2]
                user_train[2][user] = User[2][user][-maxlen-3:-2]
                user_valid[0][user] = User[0][user][-2]
                user_valid[1][user] = User[1][user][-2]
                user_valid[2][user] = User[2][user][-2]
                user_test[0][user] = User[0][user][-1]
                user_test[1][user] = User[1][user][-1]
                user_test[2][user] = User[2][user][-1]
            user_train[0][user] = list(np.zeros(1 + maxlen - len(user_train[0][user])).astype(int)) + user_train[0][user]
            user_train[1][user] = list(np.zeros(1 + maxlen - len(user_train[1][user]))) + user_train[1][user]
            user_train[2][user] = list(np.zeros(1 + maxlen - len(user_train[2][user]))) + user_train[2][user]
        return [user_train, user_valid, user_test, usernum, itemnum]

    newuser = usernum + 1
    for user in User[0]:
        user_dict[user] = user
        nfeedback = len(User[0][user])
        if nfeedback < 3:
            user_train[0][user] = User[0][user]
            user_train[1][user] = User[1][user]
            user_train[2][user] = User[2][user]
            user_valid[0][user] = 0
            user_valid[1][user] = 0
            user_valid[2][user] = 0
            user_test[0][user] = 0
            user_test[1][user] = 0
            user_test[2][user] = 0

        else:
            if augfulllen is not None:
                User[0][user] = User[0][user][-augfulllen - 2 :]
                User[1][user] = User[1][user][-augfulllen - 2 :]
                User[2][user] = User[2][user][-augfulllen - 2 :]
            numiter = math.ceil((nfeedback - maxlen - 2) / maxlen)
            if numiter > 0:
                left = (nfeedback - maxlen - 2) % maxlen
                if left > 2:
                    user_train[0][newuser] = User[0][user][:left]
                    user_train[1][newuser] = User[1][user][:left]
                    user_train[2][newuser] = User[2][user][:left]
                    user_dict[newuser] = user
                    newuser += 1
                for i in range(numiter - 1):
                    user_train[0][newuser] = User[0][user][
                        -1 + maxlen * i + left : maxlen * i + maxlen + left
                    ]
                    user_train[1][newuser] = User[1][user][
                        -1 + maxlen * i + left : maxlen * i + maxlen + left
                    ]
                    user_train[2][newuser] = User[2][user][
                        -1 + maxlen * i + left : maxlen * i + maxlen + left
                    ]
                    newuser += 1
                user_train[0][user] = User[0][user][maxlen * (numiter - 1) + left : -2][-1 - maxlen:]
                user_train[1][user] = User[1][user][maxlen * (numiter - 1) + left : -2][-1 - maxlen:]
                user_train[2][user] = User[2][user][maxlen * (numiter - 1) + left : -2][-1 - maxlen:]
            else:
                user_train[0][user] = User[0][user][:-2][-1 - maxlen:]
                user_train[1][user] = User[1][user][:-2][-1 - maxlen:]
                user_train[2][user] = User[2][user][:-2][-1 - maxlen:]
            user_valid[0][user] = User[0][user][-2]
            user_valid[1][user] = User[1][user][-2]
            user_valid[2][user] = User[2][user][-2]
            user_test[0][user] = User[0][user][-1]
            user_test[1][user] = User[1][user][-1]
            user_test[2][user] = User[2][user][-1]

        user_train[0][user] = list(np.zeros(1 + maxlen - len(user_train[0][user]))) + user_train[0][user]
        user_train[1][user] = list(np.zeros(1 + maxlen - len(user_train[1][user]))) + user_train[1][user]
        user_train[2][user] = list(np.zeros(1 + maxlen - len(user_train[2][user]))) + user_train[2][user]
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
