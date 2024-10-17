import numpy as np
import pandas as pd
from datetime import *
from collections import Counter
from scipy.stats import rankdata
import argparse
import pickle
from operator import itemgetter
from multiprocessing import Pool

from data import pop_embed

def uniform_negs_per_user(group):
    user, items, itemnum = group
    valid_items = set(range(1, itemnum + 1)) - set(items)
    selected_items = np.random.choice(list(valid_items), size=100, replace=False)
    return user + 1, list(selected_items)

def popularity_negs_per_user(group):
    user, items, itemnum, lastpop = group
    valid_items = set(range(1, itemnum + 1)) - set(items)
    filtered_pop = lastpop[~np.isin(np.arange(len(lastpop)), np.array(list(valid_items)) - 1)]
    selected_items = np.random.choice(list(valid_items), size=100, replace=False, p=filtered_pop/np.sum(filtered_pop))
    return user + 1, list(selected_items)

def process_user(user_data):
    last_u, userneg, otmpw, ao, t2_size = user_data
    counter = Counter(ao[(ao['time6'] == last_u['time6']) & (ao['time'] < last_u['time'])]['item'])
    arr = np.array(userneg) - 1
    arr = np.insert(arr, 0, last_u['item'])
    counts = np.array(itemgetter(*arr)(counter))
    urow = otmpw[int(last_u['time6']) - 1].copy()
    urow[arr] += counts
    percs = 100 * rankdata(urow, "average") / len(urow)
    return np.array([pop_embed(perc, t2_size) for perc in percs[arr]]).T

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./douban/douban_music', type=str)
    parser.add_argument('--mode', default="uniform", type=str, choices=["uniform", "popularity"], help='uniform or popularity negative sampling')
    parser.add_argument('--t2_size', default=5, type=int, help='number of fine periods')
    parser.add_argument('--userneg',  action='store_true', help='conduct user negative sampling for evaluation')
    parser.add_argument('--week_adj',  action='store_true', help='get most recent fine popularities (prior to interaction period) for evaluation')
    args = parser.parse_args()
    dataset = args.dataset

    ao = pd.read_csv(f"{dataset}_intwtime.csv", header=None)
    ao.columns = ['user', 'item', 'time4', 'time6', 'time']

    if args.userneg:
        usernum = ao.user.nunique()
        itemnum = ao.item.nunique()
        usersneg = {}
        grouped = ao.groupby('user')
        with Pool(processes=8) as pool:
            if args.mode == "uniform":
                groups = [(u, group.item.values, itemnum) for u, group in grouped]
                usersneg = dict(pool.map(uniform_negs_per_user, groups))
            elif args.mode == "popularity":
                lastpop = np.loadtxt(f"{dataset}_rawpop.txt")
                groups = [(u, group.item.values, itemnum, lastpop) for u, group in grouped]
                usersneg = dict(pool.map(popularity_negs_per_user, groups))
        name = "userneg" if args.mode == "uniform" else "popneg"
        with open(f"{dataset}_{name}.pickle", 'wb') as handle:
            pickle.dump(usersneg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.week_adj:
        users = sorted(ao.user.unique())
        with open(f"{dataset}_userneg.pickle", 'rb') as handle:
            usernegs = pickle.load(handle)
        last = ao.groupby('user').last()
        otmpw = np.loadtxt(f"{dataset}_week_curr_raw.txt")
        with Pool(processes=8) as pool:
            user_data = [(last.iloc[u], usernegs[u + 1], otmpw, ao, args.t2_size) for u in users]
            results = pool.map(process_user, user_data)
        df = np.concatenate(results)
        np.savetxt(f"{dataset}_week_wt_embed_adj.txt", df)
