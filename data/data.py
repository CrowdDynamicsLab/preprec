import numpy as np
import pandas as pd
import time as t
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import *
from collections import Counter
import json
from scipy.stats import percentileofscore, rankdata
import itertools
import argparse
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import sys

def filter_g_k_one(data,k=10,u_name='user_id',i_name='business_id',y_name='stars'):
    item_group = data.groupby(i_name).agg({y_name:'count'})
    item_g10 = item_group[item_group[y_name]>=k].index
    data_new = data[data[i_name].isin(item_g10)]
    user_group = data_new.groupby(u_name).agg({y_name:'count'})
    user_g10 = user_group[user_group[y_name]>=k].index
    data_new = data_new[data_new[u_name].isin(user_g10)]
    return data_new

def filter_tot(data,k=10,u_name='user_id',i_name='business_id',y_name='stars'):
    data_new=data
    while True:
        data_new = filter_g_k_one(data_new,k=k,u_name=u_name,i_name=i_name,y_name=y_name)
        m1 = data_new.groupby(i_name).agg({y_name:'count'})
        m2 = data_new.groupby(u_name).agg({y_name:'count'})
        num1 = m1[y_name].min()
        num2 = m2[y_name].min()
        print('item min:',num1,'user min:',num2)
        if num1>=k and num2>=k:
            break
    return data_new

def pop_embed(perc):
    if perc == 0:
        return [0]*11
    loc = int(perc//10)
    if perc % 10 == 0:
        return [0]*loc + [1] + [0]*(10 - loc)
    return [0]*loc + [1 - (perc%10) / 10] + [(perc%10) / 10] + [0]*(9 - loc)

def pop_embed2(perc):
    if perc == 0:
        return [0]*6
    loc = int(perc//20)
    if perc % 20 == 0:
        return [0]*loc + [1] + [0]*(5 - loc)
    return [0]*loc + [1 - (perc%20) / 20] + [(perc%20) / 20] + [0]*(4 - loc)

def position_encoding(perc):
    position_enc = np.array([perc / np.power(10000, 2 * (j // 2) / 10) for j in range(10)])
    position_enc[0::2] = np.sin(position_enc[0::2]) # dim 2i
    position_enc[1::2] = np.cos(position_enc[1::2]) # dim 2i+1
    return position_enc

basis_setup = np.insert(np.repeat(np.arange(1,6), 2),0,0)/100
basis_setup2 = np.insert(np.repeat(np.arange(1,4), 2),0,0)/100

def position_encoding_basis(perc):
    position_enc = perc*basis_setup
    position_enc[0::2] = np.sin(position_enc[0::2])
    position_enc[1::2] = np.cos(position_enc[1::2])
    return position_enc

def position_encoding_basis2(perc):
    position_enc = perc*basis_setup2
    position_enc[0::2] = np.sin(position_enc[0::2])
    position_enc[1::2] = np.cos(position_enc[1::2])
    return position_enc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='../../data/amazon/amazon_office', type=str)
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--sparse_val', default=100, type=int)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--name', default='wt', type=str)
args = parser.parse_args()
dataset = args.dataset

# each row must have item, user, interaction/rating, time (as unix timestamp) in that order
ao = pd.read_csv(f'{dataset}.csv')
ao.columns=["item", "user", "rate", "time"]
ao = ao.drop_duplicates(['item', 'user'])
# k-core filtering
ao = filter_tot(ao,k=5,u_name='user',i_name='item',y_name='rate')
# sparse scenario
sparse = ''
if args.sparse:
    ao.sort_values(['time'], inplace=True)
    train_filt = ao.groupby('user').apply(lambda x: x.iloc[-max(3, int(args.sparse_val/100.0*(len(x)-1))):])
    test = ao.groupby('user').last()
    ao = pd.concat([train_filt.reset_index(drop=True), test.reset_index()], axis=0)
    sparse = f"_sparse_{args.sparse_val}"
# user, item ids
item_map = dict(zip(sorted(ao.item.unique()), range(len(ao.item.unique()))))
ao.item = ao.item.apply(lambda x: item_map[x])
user_map = dict(zip(sorted(ao.user.unique()), range(len(ao.user.unique()))))
ao.user = ao.user.apply(lambda x: user_map[x])

# month and week ids, these horizons can be changed based on dataset
ao['time2'] = ao.time.apply(lambda x: datetime.fromtimestamp(x))
ao['time3'] = ao.time2.dt.year*10000 + ao.time2.dt.month*100
var_map = dict(zip(sorted(ao['time3'].unique()), range(len(ao['time3'].unique()))))
ao['time4'] = ao['time3'].apply(lambda x: var_map[x])
ao['time5'] = ao.time2.dt.year*10000 + ao.time2.dt.month*100 + ao.time2.dt.isocalendar().week
var_map = dict(zip(sorted(ao['time5'].unique()), range(len(ao['time5'].unique()))))
ao['time6'] = ao['time5'].apply(lambda x: var_map[x])
# interaction matrix processed by model with time embedding
ao.sort_values(['time2'])[['user', 'item', 'time4', 'time6', 'time']].drop_duplicates().to_csv(f'{dataset}{sparse}_intwtime.csv', header=False, index=False)
# interaction matrix processed by model without time embedding
ao.sort_values(['time2'])[['user', 'item', 'time4', 'time6']].drop_duplicates().to_csv(f'{dataset}{sparse}_int2.csv', header=False, index=False)
print("saved interaction matrix")

# 3 potential ways to compute popularity over time: just current period, cumulative over periods, exponential weighted average over periods
# uncomment below sections to run the current period and cumulative periods approaches
items = sorted(ao.item.unique())
grouped = ao.groupby('time4')

# ototaldft = pd.DataFrame(columns=["time4", "item", "perc"])
# for i, ints in grouped:
    # counter = Counter(ints.item)
    # vals = list(counter.values())
    # percs = 100 * rankdata(vals, "average") / len(vals)
    # item_orders = list(counter.keys())
    # left = list(set(items) - set(item_orders))
    # df = pd.DataFrame({"time4": [i for _ in range(len(items))], "item": item_orders + left, "perc": np.concatenate((percs, np.zeros(len(left))))})
    # ototaldft = pd.concat([ototaldft, df])
    
# ototaldft2 = pd.DataFrame(columns=["time4", "item", "perc"])
# counter = Counter()
# for i, ints in grouped:
    # counter.update(ints.item)
    # vals = list(counter.values())
    # percs = 100 * rankdata(vals, "average") / len(vals)
    # item_orders = list(counter.keys())
    # left = list(set(items) - set(item_orders))
    # df = pd.DataFrame({"time4": [i for _ in range(len(items))], "item": item_orders + left, "perc": np.concatenate((percs, np.zeros(len(left))))})
    # ototaldft2 = pd.concat([ototaldft2, df])
    
ototaldft3 = pd.DataFrame(columns=["time4", "item", "perc"])
counter = Counter()
for i, ints in grouped:
    counter = Counter({k:args.weight*v for k,v in counter.items()})
    counter.update(ints.item)
    vals = list(counter.values())
    percs = 100 * rankdata(vals, "average") / len(vals)
    item_orders = list(counter.keys())
    left = list(set(items) - set(item_orders))
    df = pd.DataFrame({"time4": [i for _ in range(len(items))], "item": item_orders + left, "perc": np.concatenate((percs, np.zeros(len(left))))})
    ototaldft3 = pd.concat([ototaldft3, df])
    
# np.savetxt(f"{dataset}{sparse}_currpop.txt", ototaldft)
# np.savetxt(f"{dataset}{sparse}_cumpop.txt", ototaldft2)
np.savetxt(f"{dataset}{sparse}_{args.name}pop.txt", ototaldft3)
print("saved monthly popularity percentiles")

# construct simple popularity feature based on each of 3 methods

# otmp = ototaldft.pivot(index = 'time4', columns = 'item', values='perc')
# otmp_ = otmp.apply(lambda x: list(itertools.chain.from_iterable([pop_embed(p) for p in x])))
# np.savetxt(f"{dataset}{sparse}_currembed.txt", otmp_.values)
# otmp2 = ototaldft2.pivot(index = 'time4', columns = 'item', values='perc')
# np.savetxt(f"{dataset}{sparse}_rawpop.txt", otmp2)
# otmp2_ = otmp2.apply(lambda x: list(itertools.chain.from_iterable([pop_embed(p) for p in x])))
# np.savetxt(f"{dataset}{sparse}_cumembed.txt", otmp2_.values)
otmp3 = ototaldft3.pivot(index = 'time4', columns = 'item', values='perc')
otmp3_ = otmp3.apply(lambda x: list(itertools.chain.from_iterable([pop_embed(p) for p in x])))
np.savetxt(f"{dataset}{sparse}_{args.name}embed.txt", otmp3_.values)

# uncomment to test sinusoidal popularity features

# otmp3_p = otmp3.apply(lambda x: list(itertools.chain.from_iterable([position_encoding(p) for p in x])))
# np.savetxt(f"{dataset}{sparse}_wtembed_pos.txt", otmp3_p.values)
# otmp3_pb = otmp3.apply(lambda x: list(itertools.chain.from_iterable([position_encoding_basis(p) for p in x])))
# np.savetxt(f"{dataset}{sparse}_wtembed_pos2.txt", otmp3_pb.values)
print("saved coarse popularity embeddings")

# capture previous 4 weeks popularity (if we're at January 30th don't want to lose January 1-January 28 data)
ototaldftw = pd.DataFrame(columns=["time6", "item", "perc"])
grouped = ao.groupby('time6')
counter = Counter()
for i, ints in grouped:
    if i >= 4:
        counter.subtract(prev4)
    counter.update(ints.item)
    vals = list(counter.values())
    percs = 100 * rankdata(vals, "average") / len(vals)
    item_orders = list(counter.keys())
    left = list(set(items) - set(item_orders))
    df = pd.DataFrame({"time6": [i for _ in range(len(items))], "item": item_orders + left, "perc": np.concatenate((percs, np.zeros(len(left))))})
    ototaldftw = pd.concat([ototaldftw, df])
    if i >= 3:
        prev4 = prev3
    if i >= 2:
        prev3 = prev2
    if i >= 1:
        prev2 = prev1
    prev1 = ints.item
# simple popularity feature w/ lower dimension to reduce time/space
otmpw = ototaldftw.pivot(index = 'time6', columns = 'item', values='perc')
otmpw_ = otmpw.apply(lambda x: list(itertools.chain.from_iterable([pop_embed2(p) for p in x])))
np.savetxt(f"{dataset}{sparse}_week_embed2.txt", otmpw_.values)
# uncomment to test sinusoidal popularity features (w/ lower dimension for 2nd)
# otmpw_p = otmpw.apply(lambda x: list(itertools.chain.from_iterable([position_encoding(p) for p in x])))
# np.savetxt(f"{dataset}{sparse}_weekembed_pos.txt", otmp3_p.values)
# otmpw_pb = otmpw.apply(lambda x: list(itertools.chain.from_iterable([position_encoding_basis2(p) for p in x])))
# np.savetxt(f"{dataset}{sparse}_weekembed_pos2.txt", otmp3_pb.values)
print("saved fine popularity embeddings")

# uncomment for user activity features used in regularization loss

# users = sorted(ao.user.unique())
# grouped = ao[['user', 'item', 'time4']].groupby(['time4'])
# ototaldftd = pd.DataFrame(columns=["time4", "user", "perc"])
# counter = Counter()
# for i, ints in grouped:
#     counter.update(ints.user)
#     vals = list(counter.values())
#     percs = 100 * rankdata(vals, "average") / len(vals)
#     user_orders = list(counter.keys())
#     left = list(set(users) - set(user_orders))
#     df = pd.DataFrame({"time4": [i for _ in range(len(users))], "user": user_orders + left, "perc": np.concatenate((percs, np.zeros(len(left))))})
#     ototaldftd = pd.concat([ototaldftd, df])
# otmpd = ototaldftd.pivot(index = 'time4', columns = 'user', values='perc')
# np.savetxt(f"{dataset}{sparse}_userhist.txt", otmpd.values)
# print("saved user activity features")

# uncomment for item-cooccurrence based feature

# get count for all consecutive item-cooccurrences (wrt user, symmetric between before and after)
# ints = pd.read_csv(f"{dataset}{sparse}_int2.csv", header=None)
# ints.columns = ["user", "item", "t1", "t2"]
# num_items = len(pd.unique(ints['item']))
# counter = Counter()
# ints.groupby('user').apply(lambda x: counter.update(list(zip(x['item'], x['item'].loc[1:]))))
# # obtain low dimensional vector via randomized svd
# rows = [x[0] for x in counter.keys()]
# cols = [x[1] for x in counter.keys()]
# vals = list(counter.values())
# finalrows = rows + cols
# finalcols = rows + cols
# vals = vals + vals

# csr = csr_matrix((vals, (finalrows, finalcols)), shape = (num_items, num_items))
# norm_csr = normalize(csr)
# random_state = 2023
# u, s, v = randomized_svd(norm_csr, n_components=50, n_oversamples=50, random_state=random_state)
# final = np.zeros(((u*s).shape[0]+1, (u*s).shape[1]))
# final[0, :] = 0
# final[1:, :] = (u*s)
# np.savetxt(f"{dataset}{sparse}_copca.txt", final)
# print("saved item coocurrence features")

print("done!")