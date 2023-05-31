import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pdb
import math

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, model, batch_size, maxlen, result_queue, SEED):
    if model == 'our':
        def sample():

            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum+1)

            seq = np.zeros([maxlen], dtype=np.int32)
            time1 = np.zeros([maxlen], dtype=np.int32)
            time2 = np.zeros([maxlen], dtype=np.int32)
            pos = np.zeros([maxlen], dtype=np.int32)
            neg = np.zeros([maxlen], dtype=np.int32)
            nxt = user_train[user][-1][0]
            idx = maxlen - 1

            ts = set(map(lambda x: x[0],user_train[user]))
            for i in reversed(user_train[user][:-1]):
                seq[idx] = i[0]
                time1[idx] = i[1]
                time2[idx] = i[2]
                pos[idx] = nxt
                if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
                nxt = i[0]
                idx -= 1
                if idx == -1: break

            return (user, seq, time1, time2, pos, neg)


    elif model == 'sasrec':
        def sample():

            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

            seq = np.zeros([maxlen], dtype=np.int32)
            pos = np.zeros([maxlen], dtype=np.int32)
            neg = np.zeros([maxlen], dtype=np.int32)
            nxt = user_train[user][-1]
            idx = maxlen - 1

            ts = set(user_train[user])
            for i in reversed(user_train[user][:-1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1: break

            return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, model, batch_size=64, maxlen=10, n_workers=1, augment=False):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        if augment:
            usernum = len(User)
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      model,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname, maxlen = None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_dict = {}
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index start from 0, assume user interactions are sorted by time
    f = open(f'../data/{fname}_int.csv', 'r')
    for line in f:
        u, i, t, t2 = line.rstrip().split(',')
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
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        return [user_train, user_valid, user_test, usernum, itemnum]

    newuser = usernum + 1
    for user in User:
        user_dict[user] = user
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[ user]
            user_valid[user] = []
            user_test[user] = []
        else:
            numiter = math.ceil((nfeedback - maxlen - 2)/maxlen)
            if numiter > 0:
                left = (nfeedback - maxlen - 2) % maxlen 
                if left > 1:
                    user_train[newuser] = User[user][:left]
                    user_dict[newuser] = user
                    newuser += 1
                for i in range(numiter-1):
                    user_train[newuser] = User[user][maxlen*i + left:maxlen*i + maxlen + left]
                    newuser += 1
                user_train[user] = User[user][maxlen*(numiter-1) + left:-2]
            else:
                user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum, user_dict] 


def data_partition2(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(f'../data/{fname}_int.csv', 'r')
    for line in f:
        u, i = line.rstrip().split(',')[0:2]
        u = int(u)
        i = int(i)
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
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args):
    if args.augment:
        [train, valid, test, usernum, itemnum, userdict] = copy.deepcopy(dataset)
    else:
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    if args.model == 'mostpop':
        raw_pop = np.loadtxt(f'../data/{args.dataset}_{args.rawpop}.txt')

    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)
        t2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]
        t1[idx] = valid[u][0][1]
        t2[idx] = valid[u][0][2]
        idx -= 1
        for t in reversed(train[u]):
            seq[idx] = t[0]
            t1[idx] = t[1]
            t2[idx] = t[2]
            idx -= 1
            if idx == -1: break

        rated = set(seq)
        rated.add(0)
        rated.add(test[u][0][0])
        item_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

        if args.model == 'mostpop':
            predictions = -raw_pop[t1[-1],item_idx]
        else:
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [t1], [t2], item_idx]])
            predictions = predictions[:, 0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    if args.augment:
        [train, valid, test, usernum, itemnum, userdict] = copy.deepcopy(dataset)
    else:
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)
        t2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for t in reversed(train[u]):
            seq[idx] = t[0]
            t1[idx] = t[1]
            t2[idx] = t[2]
            idx -= 1
            if idx == -1: break

        rated = set(seq)
        rated.add(0)
        rated.add(valid[u][0][0])
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [t1], [t2], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user



def evaluate2(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid2(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user