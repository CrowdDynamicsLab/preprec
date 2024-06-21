import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Process, Queue
import pdb
import math
from scipy.stats import rankdata, percentileofscore
import pickle
from operator import itemgetter
import psutil


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function_newrec(
    user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED
):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while np.sum(np.array(user_train[0][user]) != 0) <= 1:
            user = np.random.randint(1, usernum + 1)

        # seq = np.zeros([maxlen], dtype=np.int32)
        # time1 = np.zeros([maxlen], dtype=np.int32)
        # time2 = np.zeros([maxlen], dtype=np.int32)
        seq = np.array(user_train[0][user][:maxlen], dtype=np.int32)
        time1 = np.array(user_train[1][user], dtype=np.int32)
        time2 = np.array(user_train[2][user], dtype=np.int32)
        if len(user_train) > 3:
            time_embed = np.array(user_train[3][user], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[0][user][-1]
        idx = maxlen - 1

        ts = set(user_train[0][user])
        for i in reversed(user_train[0][user][:-1]):
            # seq[idx] = i[0]
            # time1[idx] = i[1]
            # time2[idx] = i[2]
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if len(user_train) > 3:
            return (user, seq, time1, time2, time_embed, pos, neg)
        return (user, seq, time1, time2, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


def sample_function_newrec_rfo(
    user_comb, usernum, itemnum, batch_size, maxlen, result_queue, userpop,
):
    user_train, user_valid, user_test = user_comb
    def sample(user):
        if np.sum(np.array(user_train[0][user]) != 0) <= 1:
            raise ValueError("Must have at least 2 items.")
        seq = np.array(user_train[0][user] + [user_valid[0][user]] + [user_test[0][user]], dtype=np.int32)
        time1 = np.array(user_train[1][user] + [user_valid[1][user]] + [user_test[1][user]], dtype=np.int32)
        time2 = np.array(user_train[2][user] + [user_valid[2][user]] + [user_test[2][user]], dtype=np.int32)
        return (user, seq, time1, time2)

    one_batch = []
    for i in range(batch_size):
        one_batch.append(sample(userpop[i]))
    result_queue.put(zip(*one_batch))


def sample_function_sasrec(
    user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED
):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


def sample_function_newb4rec(
    user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED
):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

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


def sample_function_bert4rec(
    user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED
):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

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


def sample_function_bprmf(
    user_train, usernum, itemnum, batch_size, maxlen, mask_prob, result_queue, SEED
):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)
        pos = np.pad(
            np.random.permutation(user_train[user]),
            (0, maxlen - len(user_train[user])),
            "constant",
        )
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

def sample_function_cl4srec(
    user_train, usernum, itemnum, batch_size, maxlen, seq_lens, result_queue, SEED
):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return (seq, seq_lens[user], pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    # @profile
    def __init__(
        self,
        User,
        usernum,
        itemnum,
        model,
        batch_size=64,
        maxlen=10,
        n_workers=1,
        mask_prob=0,
        augment=False,
        raw_feature_only=False,
        misc=None,
    ):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        if raw_feature_only:
            def extract_subset(user_comb, keys):
                user_train, user_valid, user_test = user_comb
                train_subset = tuple({key: d[key] for key in keys if key in d} for d in user_train)
                valid_subset = tuple({key: d[key] for key in keys if key in d} for d in user_valid)
                test_subset = tuple({key: d[key] for key in keys if key in d} for d in user_test)
                return (train_subset, valid_subset, test_subset)

            userint = misc
            userpop = np.argsort(userint) + 1
            print("USER INT LENGTH", len(userpop))
            nworkers = int((len(userpop)) // batch_size)
            self.result_queue = Queue(maxsize=nworkers)
            chunks = [userpop[i * batch_size:(i + 1) * batch_size] for i in range(nworkers)]
            func = sample_function_newrec_rfo
            for i in range(nworkers):
                print(f"starting worker {i}")
                user_subset = extract_subset(User, chunks[i])
                self.processors.append(
                    Process(
                        target=func,
                        args=(
                            user_subset,
                            usernum,
                            itemnum,
                            batch_size,
                            maxlen,
                            self.result_queue,
                            chunks[i],
                        ),
                    )
                )
                self.processors[-1].daemon = True
                self.processors[-1].start()
                print(f"finished worker {i}")
            return

        if augment:
            usernum = len(User)
        if model == "newrec":
            func = sample_function_newrec
        elif model == "sasrec":
            func = sample_function_sasrec
        elif model == "newb4rec":
            func = sample_function_newb4rec
        elif model == "bert4rec":
            func = sample_function_bert4rec
        elif model == "bprmf":
            func = sample_function_bprmf
            maxlen = max([len(x) for x in User.values()])
        elif model == "cl4srec":
            func = sample_function_cl4srec
            user_lens = misc
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=func,
                    args=(
                        User,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        mask_prob if model not in ["cl4srec"] else user_lens,
                        self.result_queue,
                        np.random.randint(2e9),
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def setup_negatives(dataset, dataname, mod, args):
    train, valid, test, usernum, itemnum = dataset
    print(usernum, itemnum)
    users = range(1, usernum + 1)
    usersneg = {}
    if args.userneg == "userneg":
        for u in users:
            rated = set(train[0][u])
            rated.add(valid[0][u])
            rated.add(test[0][u])
            negs = []
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                rated.add(t)
                negs.append(t)
            usersneg[u] = negs
    else:
        lastpop = np.loadtxt(f"../data/{args.dataset}_{mod}{args.rawpop}.txt")
        if lastpop.ndim == 2:
            lastpop = lastpop[-1]
        lastprob = lastpop/np.sum(lastpop)
        for u in users:
            rated = set(train[0][u])
            rated.add(valid[0][u])
            rated.add(test[0][u])
            negs = []
            for _ in range(100):
                t = np.random.choice(np.arange(1, itemnum+1), p=lastprob)
                while t in rated:
                    t = np.random.choice(np.arange(1, itemnum+1), p=lastprob)
                rated.add(t)
                negs.append(t)
            usersneg[u] = negs
    with open(f"../data/{dataname}_{mod}{args.userneg}.pickle", 'wb') as handle:
        pickle.dump(usersneg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("finished negative setup for " + dataname)


def evaluate(model, dataset, args, mode, usernegs, second=False):
    if args.model == "newrec":
        if args.save_emb:
            predict = newrec_user
        else:
            predict = newpredict_newrec
    elif args.model == "newb4rec":
        predict = predict_newb4rec
    elif args.model == "mostpop":
        predict = predict_mostpop
    elif args.model == "bprmf":
        predict = newpredict_bprmf
    elif args.model == "sasrec":
        predict = newpredict_sasrec
    elif args.model == "bert4rec":
        predict = newpredict_bert4rec
    elif args.model == "cl4srec":
        predict = newpredict_cl4srec

    load = args.dataset2 if second else args.dataset
    if args.eval_quality:
        userpop = np.loadtxt(f"../data/{load}_{args.userpop}.txt")
        if load == "amazon/amazon_office":
            change_inds = np.random.choice(np.where(userpop==5)[0], userpop[userpop==5].size//2, replace=False)
            userpop[change_inds] = 5.5
        userpop = 100 * rankdata(userpop) / len(userpop)
        userpop[userpop > 99] = 99
        numgroups = int(100 // args.quality_size)
        metrics = [([0.0 for _ in range(numgroups)], [0.0 for _ in range(numgroups)]) for _ in args.topk]
    else:
        metrics = [[0.0, 0.0] for _ in args.topk]

    valid_user = 0.0

    if args.augment:
        [train, valid, test, usernum, itemnum, userdict] = dataset
    elif args.model not in ["cl4srec"]:
        [train, valid, test, usernum, itemnum] = dataset
    else:
        [train, valid, test, usernum, itemnum, userlens] = dataset
    users = np.arange(1, usernum + 1, dtype=np.int32)
    evaluate = test if mode == "test" else valid

    if args.model == "mostpop":
        misc = np.loadtxt(f"../data/{load}_{args.rawpop}.txt")
    else:
        misc = None

    if args.model in ["newrec", "bert4rec", "sasrec", "bprmf", "cl4srec"]:
        ranks, predusers = predict(model, evaluate, train, valid, test, itemnum, args, mode, usernegs, users)
        if args.pause:
            np.savetxt(f"../data/{args.dataset}_{args.model}_ranks_{args.time_df_mod}.txt", ranks)
            sys.exit()

        if args.eval_quality:
            locs = Counter((userpop[predusers] // args.quality_size).astype(int))

        if args.use_scores:
            for j in range(len(args.alphas)):
                valid_user = len(ranks[j])
                tempranks = ranks[j]
                if not args.eval_quality:
                    metrics = [[0.0, 0.0] for _ in args.topk]
                    for i, k in enumerate(args.topk):
                        modranks = tempranks[tempranks < k]
                        metrics[i][0] = np.sum(1 / np.log2(modranks + 2))
                        metrics[i][1] = len(modranks) 
                    metrics = [
                        [metrics[i][j] / valid_user for j in range(2)]
                        for i in range(len(args.topk))
                    ]
                else:
                    metrics = [([0.0 for _ in range(numgroups)], [0.0 for _ in range(numgroups)]) for _ in args.topk]
                    fullmetrics = [[0.0, 0.0] for _ in args.topk]
                    for key in locs:
                        selranks = tempranks[(userpop[predusers] // args.quality_size).astype(int) == key]
                        for i, k in enumerate(args.topk):
                            modranks = selranks[selranks < k]
                            metrics[i][0][key] = np.sum(1 / np.log2(modranks + 2))
                            metrics[i][1][key] = len(modranks)
                            fullmetrics[i][0] += metrics[i][0][key]
                            fullmetrics[i][1] += metrics[i][1][key]
                    metrics = [[[metrics[m][n][k] / locs[k] for k in range(numgroups) if locs[k] != 0] for n in range(2)] for m in range(len(args.topk))]
                    fullmetrics = [
                        [fullmetrics[i][j] / valid_user for j in range(2)]
                        for i in range(len(args.topk))
                    ]
                    print(fullmetrics)
                print(metrics)
            sys.exit()

        valid_user = len(ranks)

        if args.eval_method != 3:
            for i, k in enumerate(args.topk):
                if args.eval_quality:
                    fullmetrics = [[0.0, 0.0] for _ in args.topk]
                    for key in locs:
                        selranks = ranks[(userpop[predusers] // args.quality_size).astype(int) == key]
                        modranks = selranks[selranks < k]
                        metrics[i][0][key] = np.sum(1 / np.log2(modranks + 2))
                        metrics[i][1][key] = len(modranks)
                        fullmetrics[i][0] += metrics[i][0][key]
                        fullmetrics[i][1] += metrics[i][1][key]
                else:
                    modranks = ranks[ranks < k]
                    metrics[i][0] = np.sum(1 / np.log2(modranks + 2))
                    metrics[i][1] = len(modranks)
        else:
            for i, k in enumerate(args.topk):
                modranks = ranks[ranks < k]
                metrics[i][0] = np.sum(1 / np.log2(modranks + 2))
                metrics[i][1] = len(modranks)

    else:
        ranks = []
        for u in users:
            if args.model == "mostpop":
                rank = predict(
                    model, [evaluate[0][u], evaluate[1][u], evaluate[2][u]], [train[0][u], train[1][u], train[2][u]], [valid[0][u], valid[1][u], valid[2][u]], [test[0][u], test[1][u], test[2][u]], itemnum, args, mode, usernegs[u], misc
                )
            elif args.model == "bprmf":
                misc = u
                rank = predict(
                    model, evaluate[u], train[u], valid[u], test[u], itemnum, args, mode, usernegs[u], misc
                )
            valid_user += 1

            if args.eval_quality:
                loc = int(userpop[u] // args.quality_size)
                counts[loc] += 1

            if args.eval_method != 3:
                for i, k in enumerate(args.topk):
                    if rank < k:
                        if args.eval_quality:
                            metrics[i][0][loc] += 1 / np.log2(rank + 2)
                            metrics[i][1][loc] += 1
                        else:
                            metrics[i][0] += 1 / np.log2(rank + 2)
                            metrics[i][1] += 1
            else:
                for i, k in enumerate(args.topk):
                    if rank < k:
                        metrics[i][0] += 1 / np.log2(rank + 2)
                        metrics[i][1] += rank
            # if valid_user % 100 == 0:
                # print(".", end="")
                # sys.stdout.flush()
            ranks.append(rank)
        if args.pause:
            np.savetxt(f"../data/{args.dataset}_{args.model}_ranks_{args.time_df_mod}.txt", np.array(ranks))
            sys.exit()

    if args.eval_quality:
        metrics = [[[round(metrics[i][j][k] / locs[k], 3) for k in range(numgroups) if locs[k] != 0] for j in range(2)] for i in range(len(args.topk))]
        fullmetrics = [[round(fullmetrics[i][j] / valid_user, 3) for j in range(2)] for i in range(len(args.topk))]
        print(fullmetrics)
    else:
        metrics = [[round(metrics[i][j] / valid_user, 3) for j in range(2)] for i in range(len(args.topk))]
    print(metrics)
    return metrics

# predict(
#             model, evaluate[u], train[u], valid[u], test[u], itemnum, args, mode, usernegs[u], misc
#         )
def newpredict_newrec(model, evaluate, train, valid, test, itemnum, args, mode, usernegs, users):
    seqs = np.array(list(itemgetter(*users)(train[0])))
    t1s = np.array(list(itemgetter(*users)(train[1])))
    t2s = np.array(list(itemgetter(*users)(train[2])))
    if not args.sparse or args.override_sparse:
        seqs_valid = np.array(list(itemgetter(*users)(valid[0])))
        t1s_valid = np.array(list(itemgetter(*users)(valid[1])))
        t2s_valid = np.array(list(itemgetter(*users)(valid[2])))
    seqs_test = np.array(list(itemgetter(*users)(test[0])))
    t1s_test = np.array(list(itemgetter(*users)(test[1])))
    t2s_test = np.array(list(itemgetter(*users)(test[2])))
    if mode == "test":
        if not args.no_valid_in_test and (not args.sparse or args.override_sparse):
            seqs = np.concatenate((seqs, np.expand_dims(seqs_valid, -1)), axis=1)
            t1s = np.concatenate((t1s, np.expand_dims(t1s_valid, -1)), axis=1)
            t2s = np.concatenate((t2s, np.expand_dims(t2s_valid, -1)), axis=1)
        item_idxs = seqs_test
        item_t1s = t1s_test
        item_t2s = t2s_test
        if args.time_embed:
            tes = np.array(list(itemgetter(*users)(test[3])))
    else:
        item_idxs = seqs_valid
        item_t1s = t1s_valid
        item_t2s = t2s_valid
        if args.time_embed:
            tes = np.array(list(itemgetter(*users)(valid[3])))
    seqs = seqs[:, -args.maxlen:]
    t1s = t1s[:, -args.maxlen:]
    t2s = t2s[:, -args.maxlen:]
    if args.time_embed:
        tes = tes[:, -args.maxlen:]

    if args.eval_method == 1:
        partitions = int(len(users)*101/(7*10**8)+1)
    elif args.eval_method == 3:
        partitions = int(len(users)*itemnum/(7*10**8)+1)
    subset = len(users)//partitions + 1
    for i in range(partitions):
        user_subset = users[subset*i:subset*(i+1)]
        if args.eval_method == 1:
            negs = np.array(list(itemgetter(*user_subset)(usernegs)))
        elif args.eval_method == 3:
            negs = (np.arange(1, itemnum+1) + np.zeros((len(user_subset), 1))).astype(int) 

        cond = np.isin(users, user_subset)
        seqs_curr, t1s_curr, t2s_curr, item_idxs_curr, item_t1s_curr, item_t2s_curr, users_curr = seqs[cond], t1s[cond], t2s[cond], item_idxs[cond], item_t1s[cond], item_t2s[cond], np.array(users)[cond]
        if args.time_embed:
            tes_curr = tes[cond]
        item_idxs_curr = np.concatenate((np.expand_dims(item_idxs_curr, -1), negs), axis=1)
        t1s_curr, t2s_curr, item_t1s_curr, item_t2s_curr = np.maximum(0, t1s_curr - 1 - args.lag//4), np.maximum(0, t2s_curr - args.lag), np.maximum(0, item_t1s_curr - 1 - args.lag//4), np.maximum(0, item_t2s_curr - args.lag)
        if args.prev_time:
            item_t1s_curr = np.repeat(np.expand_dims(t1s_curr[:,-1], -1), item_idxs_curr.shape[1], axis=1)
            item_t2s_curr = np.repeat(np.expand_dims(t2s_curr[:,-1], -1), item_idxs_curr.shape[1], axis=1)
        else:
            item_t1s_curr = np.repeat(np.expand_dims(item_t1s_curr, -1), item_idxs_curr.shape[1], axis=1)
            item_t2s_curr = np.repeat(np.expand_dims(item_t2s_curr, -1), item_idxs_curr.shape[1], axis=1)
        if args.use_scores:
            use_scores = np.loadtxt(args.use_score_dir)
            fullranks = [np.zeros(seqs_curr.shape[0]) for _ in args.alphas]
        else:
            fullranks = np.zeros(seqs_curr.shape[0])

        if args.save_scores:
            writescores = np.zeros((seqs_curr.shape[0], item_idxs_curr.shape[1]))

        for j in range(seqs_curr.shape[0]//125+1):
            inds = np.arange(125*j, min(125*(j+1), seqs_curr.shape[0]))
            if inds.size == 0:
                continue
            if args.time_embed:
                send = tes_curr[inds]
            else:
                send = None
            for item_loc in range((item_idxs_curr.shape[1]-1)//101 + 1):
                item_use = item_idxs_curr[inds, 101*item_loc:101*(item_loc+1)]
                if item_use.size == 0:
                    continue
                t1_use = item_t1s_curr[inds, 101*item_loc:101*(item_loc+1)]
                t2_use = item_t2s_curr[inds, 101*item_loc:101*(item_loc+1)]
                temp_predictions = -model.predict(
                    *[np.array(l) for l in [seqs_curr[inds], t1s_curr[inds], t2s_curr[inds], send, item_use, t1_use, t2_use, np.array(user_subset)[inds]]]
                )
                if item_loc == 0:
                    predictions = temp_predictions.detach().cpu().numpy()
                else:
                    predictions = np.concatenate([predictions, temp_predictions.detach().cpu().numpy()], axis=1)
            if args.save_scores:
                writescores[inds] = -predictions
            if args.use_scores:
                for k, alpha in enumerate(args.alphas):
                    total = -alpha*predictions + use_scores[inds]*(1-alpha)
                    fullranks[k][inds] = (-total).argsort(axis=1).argsort(axis=1)[:, 0] #.to('cpu')
            elif not args.not_rank_scores:
                random_keys = np.random.rand(predictions.shape[0], predictions.shape[1])
                structured_predictions = np.empty(predictions.shape, dtype=[('predictions', predictions.dtype), ('random_keys', random_keys.dtype)])
                structured_predictions['predictions'] = predictions
                structured_predictions['random_keys'] = random_keys
                sorted_indices = np.argsort(structured_predictions, order=('predictions', 'random_keys'))
                fullranks[inds] = sorted_indices.argsort(axis=1)[:, 0] #.to('cpu')
        if args.pause:
            pdb.set_trace()
        if args.save_scores:
            add = ''
            if args.eval_method == 3:
                add = '_global'
            if args.transfer:
                add += '_transf'
            np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/preds{add}_{i}.txt", writescores)

    if args.not_rank_scores:
        sys.exit()
    if args.save_ranks:
        np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/{args.ranks_name}.txt", fullranks)
    return fullranks, np.array(list(usernegs.keys()))[cond]



def newrec_user(model, evaluate, train, valid, test, itemnum, args, mode, usernegs, users):
    seqs = np.array(list(itemgetter(*users)(train[0])))
    t1s = np.array(list(itemgetter(*users)(train[1])))
    t2s = np.array(list(itemgetter(*users)(train[2])))
    if not args.sparse:
        seqs_valid = np.array(list(itemgetter(*users)(valid[0])))
        t1s_valid = np.array(list(itemgetter(*users)(valid[1])))
        t2s_valid = np.array(list(itemgetter(*users)(valid[2])))
    seqs_test = np.array(list(itemgetter(*users)(test[0])))
    t1s_test = np.array(list(itemgetter(*users)(test[1])))
    t2s_test = np.array(list(itemgetter(*users)(test[2])))
    if mode == "test":
        if not args.no_valid_in_test:
            seqs = np.concatenate((seqs, np.expand_dims(seqs_valid, -1)), axis=1)
            t1s = np.concatenate((t1s, np.expand_dims(t1s_valid, -1)), axis=1)
            t2s = np.concatenate((t2s, np.expand_dims(t2s_valid, -1)), axis=1)
        if args.time_embed:
            tes = np.array(list(itemgetter(*users)(test[3])))
    else:
        if args.time_embed:
            tes = np.array(list(itemgetter(*users)(valid[3])))
    seqs = seqs[:, -args.maxlen:]
    t1s = t1s[:, -args.maxlen:]
    t2s = t2s[:, -args.maxlen:]
    if args.time_embed:
        tes = tes[:, -args.maxlen:]

    partitions = int(len(users)/(1*10**3)+1)
    subset = len(users)//partitions + 1
    totranks = np.zeros(users.size)
    for i in range(partitions): 
        user_subset = users[subset*i:subset*(i+1)]
        cond = np.isin(users, user_subset)
        seqs_curr, t1s_curr, t2s_curr, users_curr = seqs[cond], t1s[cond], t2s[cond], np.array(users)[cond]
        if args.time_embed:
            tes_curr = tes[cond]
        t1s_curr, t2s_curr = np.maximum(0, t1s_curr - 1 - args.lag//4), np.maximum(0, t2s_curr - args.lag)

        user_embed = model.user_score(*[np.array(l) for l in [seqs_curr, t1s_curr, t2s_curr, tes_curr, np.array(user_subset)]]).detach().cpu().numpy()
        if i == 0:
            total_user_embed = user_embed
        else:
            total_user_embed = np.concatenate([total_user_embed, user_embed], axis=0)
    np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/user_embed_{args.label}.txt", total_user_embed)
    print("done!")
    sys.exit()
    return



def newpredict_sasrec(model, evaluate, train, valid, test, itemnum, args, mode, usernegs, users):
    listseqs = list(itemgetter(*users)(train))
    length = max(map(len, listseqs))
    seqs = np.array([[0]*(length-len(xi))+xi for xi in listseqs])
    seqs_valid = np.array(list(itemgetter(*users)(valid)))
    seqs_test = np.array(list(itemgetter(*users)(test)))

    if mode == "test":
        if not args.no_valid_in_test and (not args.sparse or args.override_sparse):
            seqs = np.concatenate((seqs, seqs_valid), axis=1)
        item_idxs = seqs_test
    else:
        item_idxs = seqs_valid
    seqs = seqs[:, -args.maxlen:]

    if args.eval_method == 1:
        partitions = int(len(users)*101/(7*10**8)+1)
    elif args.eval_method == 3:
        partitions = int(len(users)*itemnum/(7*10**8)+1)
    subset = len(users)//partitions + 1
    for i in range(partitions):
        user_subset = users[subset*i:subset*(i+1)]
        if args.eval_method == 1:
            negs = np.array(list(itemgetter(*user_subset)(usernegs)))
        elif args.eval_method == 3:
            negs = (np.arange(1, itemnum+1) + np.zeros((len(user_subset), 1))).astype(int) 

        cond = np.isin(users, user_subset)
        seqs_curr, item_idxs_curr, users_curr = seqs[cond], item_idxs[cond], np.array(users)[cond]
        item_idxs_curr = np.concatenate((item_idxs_curr, negs), axis=1)

        if args.save_scores:
            writescores = np.zeros((seqs_curr.shape[0], item_idxs_curr.shape[1]))
        if args.use_scores:
            use_scores = np.loadtxt(args.use_score_dir)
            fullranks = [np.zeros(seqs_curr.shape[0]) for _ in args.alphas]
        else:
            fullranks = np.zeros(seqs_curr.shape[0])

        for j in range(seqs_curr.shape[0]//1000+1):
            inds = np.arange(1000*j, min(1000*(j+1), seqs_curr.shape[0]))
            if inds.size == 0:
                continue
            for item_loc in range((item_idxs_curr.shape[1]-1)//200 + 1):
                item_use = item_idxs_curr[inds, 200*item_loc:200*(item_loc+1)]
                if item_use.size == 0:
                    continue
                temp_predictions = -model.predict(*[np.array(l) for l in [seqs_curr[inds], item_use]])
                if item_loc == 0:
                    predictions = temp_predictions.detach().cpu().numpy()
                else:
                    predictions = np.concatenate([predictions, temp_predictions.detach().cpu().numpy()], axis=1)
            if args.save_scores:
                writescores[inds] = -predictions
            if args.use_scores:
                for k, alpha in enumerate(args.alphas):
                    total = -alpha*predictions + use_scores[inds]*(1-alpha)
                    fullranks[k][inds] = (-total).argsort(axis=1).argsort(axis=1)[:, 0]
            elif not args.not_rank_scores:
                fullranks[inds] = predictions.argsort(axis=1).argsort(axis=1)[:, 0]
        if args.save_scores:
            add = ''
            if args.eval_method == 3:
                add = '_global'
            np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/preds{add}_{i}.txt", writescores)

    if args.not_rank_scores:
        sys.exit()
    if args.save_ranks:
        np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/{args.ranks_name}.txt", fullranks)
    return fullranks, np.array(list(usernegs.keys()))[cond]


def newpredict_bprmf(model, evaluate, train, valid, test, itemnum, args, mode, usernegs, users):
    seqs_valid = np.array(list(itemgetter(*users)(valid)))
    seqs_test = np.array(list(itemgetter(*users)(test)))

    if mode == "test":
        item_idxs = seqs_test
    else:
        item_idxs = seqs_valid

    if args.eval_method == 1:
        negs = np.array(list(itemgetter(*users)(usernegs)))
    elif args.eval_method == 3:
        negs = (np.arange(1, itemnum+1) + np.zeros((len(users), 1))).astype(int) 
    item_idxs = np.concatenate((item_idxs, negs), axis=1)
    
    if args.save_scores:
        writescores = np.zeros_like(item_idxs)
    if args.use_scores:
        use_scores = np.loadtxt(args.use_score_dir)
        fullranks = [np.zeros(item_idxs.shape[0]) for _ in args.alphas]
    else:
        fullranks = np.zeros(item_idxs.shape[0])

    for j in range(item_idxs.shape[0]//1000+1):
        inds = np.arange(1000*j, min(1000*(j+1), item_idxs.shape[0]))
        if inds.size == 0:
            continue
        users_curr = np.repeat(users[inds], item_idxs.shape[1])
        item_curr = item_idxs[inds].flatten()
        predictions = -model.predict(*[users_curr, item_curr]).detach().cpu().numpy()
        predictions = np.reshape(predictions, (inds.size, item_idxs.shape[1]))
        if args.save_scores:
            writescores[inds] = -predictions
        if args.use_scores:
            for k, alpha in enumerate(args.alphas):
                total = -alpha*predictions + use_scores[inds]*(1-alpha)
                fullranks[k][inds] = (-total).argsort(axis=1).argsort(axis=1)[:, 0]
        elif not args.not_rank_scores:
            fullranks[inds] = predictions.argsort(axis=1).argsort(axis=1)[:, 0]
    if args.save_scores:
        np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/preds.txt", writescores)

    if args.not_rank_scores:
        sys.exit()
    if args.save_ranks:
        if args.use_scores:
            for k, alpha in enumerate(args.alphas):
                np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/{args.ranks_name}_{alpha}.txt", fullranks[k])
        else:
            np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/{args.ranks_name}.txt", fullranks)
    return fullranks, np.array(list(usernegs.keys()))


def newpredict_bert4rec(model, evaluate, train, valid, test, itemnum, args, mode, usernegs, users):
    listseqs = list(itemgetter(*users)(train))
    length = max(map(len, listseqs))
    seqs = np.array([[0]*(length-len(xi))+xi for xi in listseqs])
    seqs_valid = np.array(list(itemgetter(*users)(valid)))
    seqs_test = np.array(list(itemgetter(*users)(test)))

    if mode == "test":
        if not args.no_valid_in_test:
            seqs = np.concatenate((seqs, seqs_valid), axis=1)
        item_idxs = seqs_test
    else:
        item_idxs = seqs_valid
    seqs = seqs[:, -args.maxlen:]

    negs = np.array(list(itemgetter(*usernegs.keys())(usernegs)))
    cond = np.isin(users, list(usernegs.keys())) # (item_idxs != 0) & ()
    seqs, item_idxs, users = seqs[cond], item_idxs[cond], np.array(users)[cond]
    cond = np.squeeze(item_idxs, 1) != 0
    seqs, item_idxs, users = seqs[cond], item_idxs[cond], np.array(users)[cond]
    item_idxs = np.concatenate((item_idxs, negs), axis=1)

    fullranks = np.zeros(seqs.shape[0])
    for i in range(seqs.shape[0]//100+1):
        inds = np.arange(100*i, min(100*(i+1), seqs.shape[0]))
        if inds.size == 0:
            continue
        predictions = -model.predict(
            *[torch.LongTensor(seqs[inds]), torch.LongTensor(item_idxs[inds])]
        )
        fullranks[inds] = predictions.argsort(axis=1).argsort(axis=1)[:, 0].to('cpu')
    return fullranks, np.array(list(usernegs.keys()))[cond]


def newpredict_cl4srec(model, evaluate, train, valid, test, itemnum, args, mode, usernegs, users):
    listseqs = list(itemgetter(*users)(train))
    length = max(map(len, listseqs))
    seqs = np.array([[0]*(length-len(xi))+xi for xi in listseqs])
    seqs_valid = np.array(list(itemgetter(*users)(valid)))
    seqs_test = np.array(list(itemgetter(*users)(test)))

    if mode == "test":
        if not args.no_valid_in_test and (not args.sparse or args.override_sparse):
            seqs = np.concatenate((seqs, seqs_valid), axis=1)
        item_idxs = seqs_test
    else:
        item_idxs = seqs_valid
    seqs = seqs[:, -args.maxlen:]

    if args.eval_method == 1:
        partitions = int(len(users)*101/(7*10**8)+1)
    elif args.eval_method == 3:
        partitions = int(len(users)*itemnum/(7*10**8)+1)
    subset = len(users)//partitions + 1
    for i in range(partitions):
        user_subset = users[subset*i:subset*(i+1)]
        if args.eval_method == 1:
            negs = np.array(list(itemgetter(*user_subset)(usernegs)))
        elif args.eval_method == 3:
            negs = (np.arange(1, itemnum+1) + np.zeros((len(user_subset), 1))).astype(int)

        cond = np.isin(users, user_subset)
        seqs_curr, item_idxs_curr, users_curr = seqs[cond], item_idxs[cond], np.array(users)[cond]
        item_idxs_curr = np.concatenate((item_idxs_curr, negs), axis=1)

        if args.save_scores:
            writescores = np.zeros((seqs_curr.shape[0], item_idxs_curr.shape[1]))
        if args.use_scores:
            use_scores = np.loadtxt(args.use_score_dir)
            fullranks = [np.zeros(seqs_curr.shape[0]) for _ in args.alphas]
        else:
            fullranks = np.zeros(seqs_curr.shape[0])

        for j in range(seqs_curr.shape[0]//1000+1):
            inds = np.arange(1000*j, min(1000*(j+1), seqs_curr.shape[0]))
            if inds.size == 0:
                continue
            for item_loc in range((item_idxs_curr.shape[1]-1)//200 + 1):
                item_use = item_idxs_curr[inds, 200*item_loc:200*(item_loc+1)]
                if item_use.size == 0:
                    continue
                temp_predictions = -model.predict(*[np.array(l) for l in [seqs_curr[inds], item_use]])
                if item_loc == 0:
                    predictions = temp_predictions.detach().cpu().numpy()
                else:
                    predictions = np.concatenate([predictions, temp_predictions.detach().cpu().numpy()], axis=1)
            if args.save_scores:
                writescores[inds] = -predictions
            if args.use_scores:
                for k, alpha in enumerate(args.alphas):
                    total = -alpha*predictions + use_scores[inds]*(1-alpha)
                    fullranks[k][inds] = (-total).argsort(axis=1).argsort(axis=1)[:, 0]
            elif not args.not_rank_scores:
                fullranks[inds] = predictions.argsort(axis=1).argsort(axis=1)[:, 0]
        if args.save_scores:
            add = ''
            if args.eval_method == 3:
                add = '_global'
            np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/preds{add}_{i}.txt", writescores)

    if args.not_rank_scores:
        sys.exit()
    if args.save_ranks:
        np.savetxt('/'.join(args.state_dict_path.split('/')[:-1]) + f"/{args.ranks_name}.txt", fullranks)
    return fullranks, np.array(list(usernegs.keys()))[cond]


# def newpredict_recbole(model, evaluate, train, valid, test, itemnum, args, mode, usernegs, users, userlens):
#     listseqs = list(itemgetter(*users)(train))
#     length = max(map(len, listseqs))
#     seqs = np.array([[0]*(length-len(xi))+xi for xi in listseqs])
#     seqs_valid = np.array(list(itemgetter(*users)(valid)))
#     seqs_test = np.array(list(itemgetter(*users)(test)))
#
#     if mode == "test":
#         if not args.no_valid_in_test:
#             seqs = np.concatenate((seqs, seqs_valid), axis=1)
#         item_idxs = seqs_test
#         use_userlens = np.minimum(userlens+2, args.maxlen)
#     else:
#         item_idxs = seqs_valid
#         use_userlens = np.minimum(userlens+1, args.maxlen)
#     seqs = seqs[:, -args.maxlen:]
#
#     negs = np.array(list(itemgetter(*usernegs.keys())(usernegs)))
#     cond = np.isin(users, list(usernegs.keys())) # (item_idxs != 0) & ()
#     seqs, item_idxs, users = seqs[cond], item_idxs[cond], np.array(users)[cond]
#     cond = np.squeeze(item_idxs, 1) != 0
#     seqs, item_idxs, users = seqs[cond], item_idxs[cond], np.array(users)[cond]
#     item_idxs = np.concatenate((item_idxs, negs), axis=1)
#
#     fullranks = np.zeros(seqs.shape[0])
#     for i in range(seqs.shape[0]//100+1):
#         inds = np.arange(100*i, min(100*(i+1), seqs.shape[0]))
#         if inds.size == 0:
#             continue
#         predictions = -model.predict(
#             *[torch.LongTensor(seqs[inds]).to(args.device), torch.LongTensor(use_userlens[inds]).to(args.device), torch.LongTensor(item_idxs[inds]).to(args.device)]
#         )
#         fullranks[inds] = predictions.argsort(axis=1).argsort(axis=1)[:, 0].to('cpu')
#     return fullranks, np.array(list(usernegs.keys()))[cond]


def predict_newrec(model, evaluate, train, valid, test, itemnum, args, mode, negs, user):
    seq = [x[0] for x in train]
    t1 = [x[1] for x in train]
    t2 = [x[2] for x in train]
    if mode == "test":
        seq.append(valid[0][0])
        t1.append(valid[0][1])
        t2.append(valid[0][2])

    seq = seq[-args.maxlen :]
    padding_len = args.maxlen - len(seq)
    seq = [0] * padding_len + seq
    t1 = t1[-args.maxlen :]
    t1 = [0] * padding_len + t1
    t2 = t2[-args.maxlen :]
    t2 = [0] * padding_len + t2

    if args.eval_method == 1:
        item_idx = [evaluate[0][0]]
        item_idx.extend(negs)

    elif args.eval_method == 2:
        pass

    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum + 1)).difference(rated))
        item_idx.insert(0, evaluate[0][0])

    predictions = -model.predict(
        *[np.array(l) for l in [[seq], [t1], [t2], item_idx, user]]
    )
    predictions = torch.flatten(predictions)
    rank = predictions.argsort().argsort()[0].item()
    return rank


def predict_newb4rec(model, evaluate, train, valid, test, itemnum, args, mode, negs, _):
    seq = [x[0] for x in train]
    t1 = [x[1] for x in train]
    t2 = [x[2] for x in train]
    if mode == "test":
        seq.append(valid[0][0])
        t1.append(valid[0][1])
        t2.append(valid[0][2])

    seq = seq[-args.maxlen :]
    t1 = t1[-args.maxlen :]
    t2 = t2[-args.maxlen :]
    padding_len = args.maxlen - len(seq)
    seq = [0] * padding_len + seq
    t1 = [0] * padding_len + t1
    t2 = [0] * padding_len + t2

    if args.eval_method == 1:
        item_idx = [evaluate[0][0]]
        item_idx.extend(negs)

    elif args.eval_method == 2:
        pass

    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum + 1)).difference(rated))
        item_idx.insert(0, evaluate[0][0])

    predictions = -model.predict(*[np.array(l) for l in [[seq], [t1], [t2], item_idx]])
    predictions = torch.flatten(predictions)  # - for 1st argsort DESC
    rank = predictions.argsort().argsort()[0].item()
    return rank


def predict_mostpop(model, evaluate, train, valid, test, itemnum, args, mode, negs, rawpop):
    if args.eval_method == 1:
        item_idx = [evaluate[0]]
        item_idx.extend(negs)

    elif args.eval_method == 2:
        pass

    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum + 1)).difference(rated))
        item_idx.insert(0, evaluate[0])

    if len(rawpop.shape) == 2:
        if mode == "test":
            t1 = test[1] - 1
        else:
            t1 = valid[1] - 1
        predictions = -rawpop[-1, np.array(item_idx)-1]
        # predictions = -rawpop[t1, np.array(item_idx)-1]
    else:
        predictions = -rawpop[np.array(item_idx)-1]
    b = np.random.random(predictions.size)
    rank = np.lexsort((b, predictions)).argsort()[0].item()
    return rank




def predict_bprmf(model, evaluate, train, valid, test, itemnum, args, mode, negs, u):
    if args.eval_method == 1:
        item_idx = [evaluate[0]]
        item_idx.extend(negs)

    elif args.eval_method == 2:
        pass

    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum + 1)).difference(rated))
        item_idx.insert(0, evaluate[0])

    predictions = -model.predict(*[np.array(l) for l in [[u], item_idx]])
    predictions = torch.flatten(predictions)  # - for 1st argsort DESC

    rank = predictions.argsort().argsort()[0].item()
    return rank


def predict_sasrec(model, evaluate, train, valid, test, itemnum, args, mode, negs, _):
    seq = train
    if mode == "test":
        seq.append(valid[0])
    seq = seq[-args.maxlen :]
    padding_len = args.maxlen - len(seq)
    seq = [0] * padding_len + seq

    if args.eval_method == 1:
        item_idx = [evaluate[0]]
        item_idx.extend(negs)
    elif args.eval_method == 2:
        pass

    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum + 1)).difference(rated))
        item_idx.insert(0, evaluate[0])

    predictions = -model.predict(*[np.array(l) for l in [[seq], item_idx]])
    predictions = torch.flatten(predictions)  # - for 1st argsort DESC

    rank = predictions.argsort().argsort()[0].item()
    return rank


def predict_bert4rec(model, evaluate, train, valid, test, itemnum, args, mode, negs, _):
    if mode == "test":
        seq = train + valid + [0]
    else:
        seq = train + [0]
    seq = seq[-args.maxlen :]
    padding_len = args.maxlen - len(seq)
    seq = [0] * padding_len + seq

    if args.eval_method == 1:
        item_idx = [evaluate[0]]
        item_idx.extend(negs)

    elif args.eval_method == 2:
        pass

    elif args.eval_method == 3:
        item_idx = list(set(range(1, itemnum + 1)).difference(rated))
        item_idx.insert(0, evaluate[0])

    predictions = -model.predict(*[torch.LongTensor([seq]), torch.LongTensor(item_idx)])
    predictions = torch.flatten(predictions)  # - for 1st argsort DESC

    rank = predictions.argsort().argsort()[0].item()
    return rank

