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


def sample_function_newb4rec(user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        tokens = []
        t1 = []
        t2 = []
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
                    tokens.append(s[0])
                labels.append(s[0])
            else:
                tokens.append(s[0])
                labels.append(0)
            t1.append(s[1])
            t2.append(s[2])

        tokens = tokens[-maxlen:]
        labels = labels[-maxlen:]
        t1 = t1[-maxlen:]
        t2 = t2[-maxlen:]
        mask_len = maxlen - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        t1 = [0] * mask_len + t1
        t2 = [0] * mask_len + t2

        return tokens, labels, t1, t2

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


def sample_function_bprmf(user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
        pos = np.pad(np.random.permutation(user_train[user]), (0, maxlen - len(user_train[user])), 'constant')
        neg = np.zeros([maxlen], dtype=np.int32)
        ts = set(user_train[user])
        for i in range(len(user_train[user])):
            neg[i] = random_neq(1, itemnum + 1, ts)
        return user, pos, neg

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
        elif model == 'newb4rec':
            func = sample_function_newb4rec
        elif model == 'bert4rec':
            func = sample_function_bert4rec
        elif model == 'bprmf':
            func = sample_function_bprmf
            maxlen = max([len(x) for x in User.values()])

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
def data_partition(fname, maxlen = None, augfulllen = None):
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
            if augfulllen is not None:
                User[user] = User[user][-augfulllen - 2:]
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


def data_partition2(fname, wrong_num):
    add = 0 if wrong_num else 1
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
    f = open(f'../data/{fname}_int2.csv', 'r')
    for line in f:
        u, i = line.rstrip().split(',')[0:2]
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
            if max(validi, testi) > len(User[user])-1:
                pdb.set_trace()
            user_valid[user] = [User[user][validi]]
            user_test[user] = [User[user][testi]]
            user_train[user] = np.delete(User[user], [validi, testi])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args):
    if args.model == 'newrec':
        predict = predict_newrec
    elif args.model == 'newb4rec':
        predict = predict_newb4rec
    elif args.model == 'mostpop':
        predict = predict_mostpop
    elif args.model == 'bprmf':
        predict = predict_bprmf
    elif args.model == 'sasrec':
        predict = predict_sasrec
    elif args.model == 'bert4rec':
        predict = predict_bert4rec

    if args.eval_quality:
        userpop = np.loadtxt(f'../data/{args.dataset}_{args.userpop}.txt')
        userpop = 100*rankdata(userpop)/len(userpop)
        userpop[userpop > 99] = 99
        numgroups = int(100//args.quality_size)
        metrics = [([0.0 for _ in range(numgroups)], [0.0 for _ in range(numgroups)]) for _ in args.topk]
        counts = [0 for _ in range(numgroups)]
    else:
        metrics = [[0.0, 0.0] for _ in args.topk]

    valid_user = 0.0

    if args.augment:
        [train, valid, test, usernum, itemnum, userdict] = copy.deepcopy(dataset)
    else:
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    users = range(1, usernum + 1)
    evaluate = test if args.mode == 'test' else valid

    if args.model == "mostpop":
        misc = np.loadtxt(f'../data/{args.dataset}_{args.rawpop}.txt')
    else:
        misc = None

    for u in users:
        if len(train[u]) < 1 or len(evaluate[u]) < 1: continue
        if args.model == "bprmf":
            misc = u
        rank = predict(model, evaluate[u], train[u], valid[u], test[u], itemnum, args, misc)
        valid_user += 1

        if args.eval_quality:
            loc = int(userpop[u]//args.quality_size)
            counts[loc] += 1

        if args.eval_method != 3:
            for i, k in enumerate(args.topk):
                if rank < k:
                    if args.eval_quality:
                        metrics[i][0][loc] += 1 / np.log2(rank+2)
                        metrics[i][1][loc] += 1
                    else:
                        metrics[i][0] += 1 / np.log2(rank+2)
                        metrics[i][1] += 1
        else:
            for i, k in enumerate(args.topk):
                if rank < k:
                    metrics[i][0] += 1/np.log2(rank+2)
                    metrics[i][1] += rank
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    if args.eval_quality:
        metrics = [[[metrics[i][j][k]/counts[k] for k in range(numgroups) if counts[k] != 0] for j in range(2)] for i in range(len(args.topk))]
    else:
        metrics = [[metrics[i][j]/valid_user for j in range(2)] for i in range(len(args.topk))]
    if args.mode == 'test':
        print(metrics)
    return metrics


def predict_newrec(model, evaluate, train, valid, test, itemnum, args, _):
    seq = [x[0] for x in train]
    t1 = [x[1] for x in train]
    t2 = [x[2] for x in train]
    if args.mode == 'test':
        seq.append(valid[0][0])
        t1.append(valid[0][1])
        t2.append(valid[0][2])
    seq = seq[-args.maxlen:]
    padding_len = args.maxlen - len(seq)
    seq = [0] * padding_len + seq
    t1 = t1[-args.maxlen:]
    t1 = [0] * padding_len + t1
    t2 = t2[-args.maxlen:]
    t2 = [0] * padding_len + t2

    rated = set([x[0] for x in train])
    rated.add(valid[0][0])
    rated.add(test[0][0])

    if args.eval_method == 1:
        item_idx = [evaluate[0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

    elif args.eval_method == 2:
        pass
    
    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum+1)).difference(rated))
        item_idx.insert(0, evaluate[0][0])

    predictions = -model.predict(*[np.array(l) for l in [[seq], [t1], [t2], item_idx]])
    predictions = torch.flatten(predictions)
    rank = predictions.argsort().argsort()[0].item()
    return rank


def predict_newb4rec(model, evaluate, train, valid, test, itemnum, args, _):
    seq = [x[0] for x in train]
    t1 = [x[1] for x in train]
    t2 = [x[2] for x in train]
    if args.mode == 'test':
        seq.append(valid[0][0])
        t1.append(valid[0][1])
        t2.append(valid[0][2])

    seq = seq[-args.maxlen:]
    t1 = t1[-args.maxlen:]
    t2 = t2[-args.maxlen:]
    padding_len = args.maxlen - len(seq)
    seq = [0] * padding_len + seq
    t1 = [0] * padding_len + t1
    t2 = [0] * padding_len + t2

    rated = set([x[0] for x in train])
    rated.add(valid[0][0])
    rated.add(test[0][0])

    if args.eval_method == 1:
        item_idx = [evaluate[0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

    elif args.eval_method == 2:
        pass
    
    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum+1)).difference(rated))
        item_idx.insert(0, evaluate[0][0])

    predictions = -model.predict(*[np.array(l) for l in [[seq], [t1], [t2], item_idx]])
    predictions = torch.flatten(predictions) # - for 1st argsort DESC
    rank = predictions.argsort().argsort()[0].item()
    return rank


def predict_mostpop(model, evaluate, train, valid, test, itemnum, args, rawpop):
    if args.mode == 'test':
        t1 = valid[-1][1]
    else:
        t1 = train[-1][1]
    rated = set([x[0] for x in train])
    rated.add(valid[0][0])
    rated.add(test[0][0])

    if args.eval_method == 1:
        item_idx = [evaluate[0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

    elif args.eval_method == 2:
        pass
    
    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum+1)).difference(rated))
        item_idx.insert(0, evaluate[0][0])

    predictions = -rawpop[t1,item_idx]
    rank = predictions.argsort().argsort()[0].item()
    return rank


def predict_bprmf(model, evaluate, train, valid, test, itemnum, args, u):
    rated = set(train)
    rated.add(valid[0])
    rated.add(test[0])

    if args.eval_method == 1:
        item_idx = [evaluate[0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

    elif args.eval_method == 2:
        pass
    
    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum+1)).difference(rated))
        item_idx.insert(0, evaluate[0])

    predictions = -model.predict(*[np.array(l) for l in [[u], item_idx]])
    predictions = torch.flatten(predictions) # - for 1st argsort DESC

    rank = predictions.argsort().argsort()[0].item()
    return rank


def predict_sasrec(model, evaluate, train, valid, test, itemnum, args, _):
    seq = train
    if args.mode == 'test':
        seq.append(valid[0])
    seq = seq[-args.maxlen:]
    padding_len = args.maxlen - len(seq)
    seq = [0] * padding_len + seq

    rated = set(train)
    rated.add(valid[0])
    rated.add(test[0])

    if args.eval_method == 1:
        item_idx = [evaluate[0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

    elif args.eval_method == 2:
        pass
    
    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum+1)).difference(rated))
        item_idx.insert(0, evaluate[0])

    predictions = -model.predict(*[np.array(l) for l in [[seq], item_idx]])
    predictions = torch.flatten(predictions) # - for 1st argsort DESC

    rank = predictions.argsort().argsort()[0].item()
    return rank


def predict_bert4rec(model, evaluate, train, valid, test, itemnum, args, _):
    if args.mode == 'test':
        seq = train + valid + [0]
    else:
        seq = train + [0]
    seq = seq[-args.maxlen:]
    padding_len = args.maxlen - len(seq)
    seq = [0] * padding_len + seq

    rated = set(train)
    rated.add(valid[0])
    rated.add(test[0])

    if args.eval_method == 1:
        item_idx = [evaluate[0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            rated.add(t)

    elif args.eval_method == 2:
        pass
    
    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum+1)).difference(rated))
        item_idx.insert(0, evaluate[0])

    pdb.set_trace()
    predictions = -model.predict(*[torch.LongTensor([seq]), torch.LongTensor(item_idx)])
    predictions = torch.flatten(predictions) # - for 1st argsort DESC

    rank = predictions.argsort().argsort()[0].item()
    return rank