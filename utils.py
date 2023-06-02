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


def sample_function_newrec(user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED):
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

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


def sample_function_sasrec(user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED):
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


def sample_function_bert4rec(user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        tokens = []
        labels = []
        for s in user_train[user]:
            prob = np.random.random()
            if prob < mask_prob:
                prob /= mask_prob

                if prob < 0.8:
                    tokens.append(0)
                elif prob < 0.9:
                    tokens.append(np.random.randint(1, itemnum + 1))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-maxlen:]
        labels = labels[-maxlen:]

        mask_len = maxlen - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return tokens, labels

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, model, batch_size=64, maxlen=10, n_workers=1, mask_prob=0, augment=False):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        if augment:
            usernum = len(User)

        if model == 'newrec':
            func = sample_function_newrec
        elif model == 'sasrec':
            func = sample_function_sasrec
        elif model == 'bert4rec':
            func = sample_function_bert4rec

        for i in range(n_workers):
            self.processors.append(Process(target=func, args=(
                User, usernum, itemnum, batch_size, maxlen, mask_prob, self.result_queue, np.random.randint(2e9))))
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
    f = open(f'../data/{fname}_int2.csv', 'r')
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
            numiter = math.ceil((nfeedback - maxlen - 2)/maxlen)
            if numiter > 0:
                left = (nfeedback - maxlen - 2) % maxlen 
                if left > 2:
                    user_train[newuser] = User[user][:left]
                    user_dict[newuser] = user
                    newuser += 1
                for i in range(numiter-1):
                    user_train[newuser] = User[user][maxlen*i + left:maxlen*i + maxlen + left]
                    newuser += 1
                user_train[user] = User[user][maxlen*(numiter-1) + left:-2]
            else:
                user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    return [user_train, user_valid, user_test, usernum, itemnum, user_dict] 


def data_partition2(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(f'../data/{fname}_int2.csv', 'r')
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
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args):
    if args.model == 'newrec':
        return evaluate_newrec(model, dataset, args)
    if args.model == 'sasrec':
        return evaluate_sasrec(model, dataset, args)
    if args.model == 'bert4rec':
        return evaluate_bert4rec(model, dataset, args)
    if args.model == 'mostpop':
        return evaluate_mostpop(model, dataset, args)


def evaluate_newrec(model, dataset, args):
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

    evaluate = test if args.mode == 'test' else valid

    for u in users:
        if len(train[u]) < 1 or len(evaluate[u]) < 1: continue

        seq = [x[0] for x in train[u]]
        t1 = [x[1] for x in train[u]]
        t2 = [x[2] for x in train[u]]
        if args.mode == 'test':
            seq.append(valid[u][0][0])
            t1.append(valid[u][0][1])
            t2.append(valid[u][0][2])
        seq = seq[-args.maxlen:]
        padding_len = args.maxlen - len(seq)
        seq = [0] * padding_len + seq
        t1 = t1[-args.maxlen:]
        t1 = [0] * padding_len + t1
        t2 = t2[-args.maxlen:]
        t2 = [0] * padding_len + t2

        rated = set([x[0] for x in train[u]])
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        item_idx = [evaluate[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

        predictions = -model.predict(*[np.array(l) for l in [[seq], [t1], [t2], item_idx]])
        predictions = torch.flatten(predictions)
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < args.topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_mostpop(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    raw_pop = np.loadtxt(f'../data/{args.dataset}_{args.rawpop}.txt')

    evaluate = test if args.mode == 'test' else valid

    for u in users:
        if len(train[u]) < 1 or len(evaluate[u]) < 1: continue

        if args.mode == 'test':
            t1 = valid[u][-1][1]
        else:
            t1 = train[u][-1][1]
        rated = set([x[0] for x in train[u]])
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        item_idx = [evaluate[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

        predictions = -raw_pop[t1,item_idx]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < args.topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_sasrec(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    evaluate = test if args.mode == 'test' else valid

    for u in users:
        if len(train[u]) < 1 or len(evaluate[u]) < 1: continue

        seq = train[u]
        if args.mode == 'test':
            seq.append(valid[u][0])
        seq = seq[-args.maxlen:]
        padding_len = args.maxlen - len(seq)
        seq = [0] * padding_len + seq

        rated = set(train[u])
        rated.add(valid[u][0])
        rated.add(test[u][0])
        item_idx = [evaluate[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[seq], item_idx]])
        predictions = torch.flatten(predictions) # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < args.topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_bert4rec(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    evaluate = test if args.mode == 'test' else valid

    for u in users:
        if len(train[u]) < 1 or len(evaluate[u]) < 1: continue

        if args.mode == 'test':
            seq = train[u] + valid[u] + [0]
        else:
            seq = train[u] + [0]
        seq = seq[-args.maxlen:]
        padding_len = args.maxlen - len(seq)
        seq = [0] * padding_len + seq

        rated = set(train[u])
        rated.add(valid[u][0])
        rated.add(test[u][0])
        item_idx = [evaluate[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[torch.LongTensor([seq]), torch.LongTensor(item_idx)])
        predictions = torch.flatten(predictions) # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < args.topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user