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



def data_partition_wtime(fname, maxlen, sparse_name = '', override_sparse=False, mod=''):
    """
    dataset pre-processing that uses coarse time index, fine time index, and relative time embedding via exact timestamp
    refer to data/data.py for dataset formatting
    """

    usernum = 0
    itemnum = 0
    User = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    user_dict = {}
    user_train = ({}, {}, {}, {})
    user_valid = ({}, {}, {}, {})
    user_test = ({}, {}, {}, {})
    f = open(f"./data/{fname}_{sparse_name}intwtime{mod}.csv", "r")
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

    for user in User[0]:
        nfeedback = len(User[0][user])
        uselen = min(maxlen+2, len(User[3][user]))
        temp = np.array(User[3][user][-uselen+1:]) - np.array(User[3][user][-uselen:-1])
        if sparse_name == '' or override_sparse:
            user_train[0][user] = User[0][user][-maxlen-3:-2]
            user_train[1][user] = User[1][user][-maxlen-3:-2]
            user_train[2][user] = User[2][user][-maxlen-3:-2]
            user_train[3][user] = list(np.argsort(temp[:-2][-maxlen:]) + 1)
            user_valid[0][user] = User[0][user][-2]
            user_valid[1][user] = User[1][user][-2]
            user_valid[2][user] = User[2][user][-2]
            user_valid[3][user] = list(np.argsort(temp[:-1][-maxlen:]) + 1)
            user_valid[3][user] = list(np.zeros(maxlen - len(user_valid[3][user]))) + user_valid[3][user]
        else:
            user_train[0][user] = User[0][user][-maxlen-2:-1]
            user_train[1][user] = User[1][user][-maxlen-2:-1]
            user_train[2][user] = User[2][user][-maxlen-2:-1]
            user_train[3][user] = list(np.argsort(temp[:-1][-maxlen:]) + 1)
            user_valid[0][user] = 0
            user_valid[1][user] = 0
            user_valid[2][user] = 0
            user_valid[3][user] = []
        user_test[0][user] = User[0][user][-1]
        user_test[1][user] = User[1][user][-1]
        user_test[2][user] = User[2][user][-1]
        user_test[3][user] = list(np.argsort(temp[-maxlen:]) + 1)
        user_train[0][user] = list(np.zeros(1 + maxlen - len(user_train[0][user])).astype(int)) + user_train[0][user]
        user_train[1][user] = list(np.zeros(1 + maxlen - len(user_train[1][user]))) + user_train[1][user]
        user_train[2][user] = list(np.zeros(1 + maxlen - len(user_train[2][user]))) + user_train[2][user]
        user_train[3][user] = list(np.zeros(maxlen - len(user_train[3][user]))) + user_train[3][user]
        user_test[3][user] = list(np.zeros(maxlen - len(user_test[3][user]))) + user_test[3][user]
    return [user_train, user_valid, user_test, usernum, itemnum]



def data_partition(fname, maxlen, sparse_name = '', override_sparse=False, mod=''):
    """
    dataset pre-processing that uses coarse time index and fine time index
    refer to data/data.py for dataset formatting
    """

    usernum = 0
    itemnum = 0
    User = [defaultdict(list), defaultdict(list), defaultdict(list)]
    user_dict = {}
    user_train = ({}, {}, {})
    user_valid = ({}, {}, {})
    user_test = ({}, {}, {})
    if sparse_name != '':
        f = open(f"./data/{fname}_{sparse_name}intwtime{mod}.csv", "r")
    else:
        f = open(f"./data/{fname}_int2{mod}.csv", "r")

    for line in f:
        if sparse_name != '':
            u, i, t, t2, _ = line.rstrip().split(",")
        else:
            u, i, t, t2 = line.rstrip().split(",")
        u = int(u) + 1
        if u >= 40000:
            continue
        i = int(i) + 1
        t = int(t)
        t2 = int(t2)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[0][u].append(i)
        User[1][u].append(t)
        User[2][u].append(t2)

    for user in User[0]:
        if sparse_name != '' and not override_sparse:
            user_train[0][user] = User[0][user][-maxlen-3:-1]
            user_train[1][user] = User[1][user][-maxlen-3:-1]
            user_train[2][user] = User[2][user][-maxlen-3:-1]
            user_valid[0][user] = 0 
            user_valid[1][user] = 0 
            user_valid[2][user] = 0 
        else:
            user_train[0][user] = User[0][user][-maxlen-3:-2]
            user_train[1][user] = User[1][user][-maxlen-3:-2]
            user_train[2][user] = User[2][user][-maxlen-3:-2]
            user_train[0][user] = list(np.zeros(1 + maxlen - len(user_train[0][user])).astype(int)) + user_train[0][user]
            user_train[1][user] = list(np.zeros(1 + maxlen - len(user_train[1][user]))) + user_train[1][user]
            user_train[2][user] = list(np.zeros(1 + maxlen - len(user_train[2][user]))) + user_train[2][user]
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



def data_partition2(fname, sparse_name, override_sparse, mod=''):
    """
    dataset pre-processing without time 
    refer to data/data.py for dataset formatting
    """

    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    if sparse_name != '':
        f = open(f"./data/{fname}_{sparse_name}intwtime{mod}.csv", "r")
    else:
        f = open(f"./data/{fname}_int2{mod}.csv", "r")
    for line in f:
        u, i = line.rstrip().split(",")[0:2]
        u = int(u) + 1
        i = int(i) + 1
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    min_list_key = min(User, key=lambda k: len(User[k]))
    min_length = len(User[min_list_key])
    if min_length < 5 and not override_sparse:
        sparse = True
    else:
        sparse = False

    for user in User:
        # nfeedback = len(User[user])
        if sparse:
            user_train[user] = User[user][:-1]
            user_valid[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
        user_test[user] = [User[user][-1]]
    return [user_train, user_valid, user_test, usernum, itemnum]


def data_partition3(fname, maxlen, sparse_name, override_sparse, mod=''):
    """
    dataset pre-processing without time but with sequence lengths
    refer to data/data.py for dataset formatting
    """

    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    if sparse_name != '':
        f = open(f"./data/{fname}_{sparse_name}intwtime{mod}.csv", "r")
    else:
        f = open(f"./data/{fname}_int2{mod}.csv", "r")
    for line in f:
        u, i = line.rstrip().split(",")[0:2]
        u = int(u) + 1
        i = int(i) + 1
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    min_list_key = min(User, key=lambda k: len(User[k]))
    min_length = len(User[min_list_key])
    if min_length < 5 and not override_sparse:
        sparse = True
    else:
        sparse = False

    userlens = np.zeros(usernum+1, dtype=int)
    for user in User:
        if sparse:
            user_train[user] = User[user][:-1]
            user_valid[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
        user_test[user] = [User[user][-1]]
        userlens[user] = min(maxlen, len(user_train[user])-1)
    return [user_train, user_valid, user_test, usernum, itemnum, userlens]
