import numpy as np
import pandas as pd
from datetime import *
from collections import Counter
from scipy.stats import rankdata
import itertools
import argparse
import sys
import pickle
from operator import itemgetter

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

def pop_embed(perc, num=10):
    if perc == 0:
        return [0] * (num + 1)
    rev = 100 // num
    loc = int(perc // rev)
    if loc >= num:
        loc = num  # Ensure that for 100% the index is set to the last element
    res = [0] * (num + 1)
    if perc % rev == 0 and loc <= num:
        res[loc] = 1
    else:
        if loc < num:  # Check to prevent out-of-bounds access
            res[loc] = 1 - (perc % rev) / rev
            res[loc + 1] = (perc % rev) / rev
    return res

def pop_embed_old(perc):
    if perc == 0:
        return [0]*11
    loc = int(perc//10)
    if perc % 10 == 0:
        return [0]*loc + [1] + [0]*(10 - loc)
    return [0]*loc + [1 - (perc%10) / 10] + [(perc%10) / 10] + [0]*(9 - loc)

def pop_embed2_old(perc):
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
parser.add_argument('--dataset', default='../data/douban/douban_music', type=str)
parser.add_argument('--mode', default='', type=str, help='sparse,fs,temp_fs')
parser.add_argument('--mode2', default='orig', type=str, help='orig,sin,perc')
parser.add_argument('--extra', default='', type=str)
parser.add_argument('--sparse_val', default=100, type=int)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--name', default='wt', type=str)
parser.add_argument('--extra2', default='', type=str)
parser.add_argument('--t1_cutoff', default=366/12, type=float)
parser.add_argument('--t1_size', default=10, type=int)
parser.add_argument('--t2_cutoff', default=366/62, type=float)
parser.add_argument('--t2_size', default=5, type=int)
parser.add_argument('--stop_early',  action='store_true')
parser.add_argument('--week_adj',  action='store_true')
parser.add_argument('--last_pop',  action='store_true')
parser.add_argument('--use_ref',  action='store_true')
parser.add_argument('--not_coarse',  action='store_true')
parser.add_argument('--not_fine',  action='store_true')
parser.add_argument('--reference', default='../data/amazon/amazon_tool_intwtime.csv', type=str)
parser.add_argument('--ref_frac', default=10, type=int)
parser.add_argument('--noise_std', default=0, type=int)
parser.add_argument('--noise_p', default=1, type=float)
parser.add_argument('--noise_prop_t', default=0, type=float)
parser.add_argument('--use_perc',  action='store_true')
parser.add_argument('--reverse',  action='store_true')
parser.add_argument('--day_shift',  action='store_true')
parser.add_argument('--hour_shift',  action='store_true')
args = parser.parse_args()
dataset = args.dataset

# each row must have item, user, interaction/rating, time (as unix timestamp) in that order
ao = pd.read_csv(f'{dataset}.csv')
ao.columns=["item", "user", "rate", "time"]
ao = ao.drop_duplicates(['item', 'user'])
# sparse scenario
sparse = ''
if args.mode == 'sparse':
    # k-core filtering
    ao = filter_tot(ao,k=5,u_name='user',i_name='item',y_name='rate')
    ao.sort_values(['time'], inplace=True)
    train_filt = ao.groupby('user').apply(lambda x: x.iloc[-max(3, int(args.sparse_val/100.0*(len(x)-1)))-1:])
    test = ao.groupby('user').last()
    ao = pd.concat([train_filt.reset_index(drop=True), test.reset_index()], axis=0)
    sparse = f"_sparse_{args.sparse_val}{args.extra}"
elif args.mode == 'fs':
    # k-core filtering
    ao = filter_tot(ao,k=5,u_name='user',i_name='item',y_name='rate')
    ao.sort_values(['time'], inplace=True)
    ao = ao.groupby('user').apply(lambda x: x.iloc[:max(3, int(args.sparse_val/100.0*(len(x)-1)))+1]).reset_index(drop=True)
    sparse = f"_fs_{args.sparse_val}{args.extra}"
elif args.mode == 'temp_fs':
    ao.sort_values(['time'], inplace=True)
    if args.use_ref:
        temp = pd.read_csv(args.reference)
        size = temp.shape[0] * args.ref_frac / 100.0
    else:
        size = ao.shape[0] * args.sparse_val / 100.0
    ao = ao.iloc[:int(size)]
    # k-core filtering
    ao = filter_tot(ao,k=5,u_name='user',i_name='item',y_name='rate')
    sparse = f"_temp_fs_{args.sparse_val}{args.extra}"
else:
    ao = filter_tot(ao,k=5,u_name='user',i_name='item',y_name='rate')
# user, item ids
item_map = dict(zip(sorted(ao.item.unique()), range(len(ao.item.unique()))))
ao.item = ao.item.apply(lambda x: item_map[x])
user_map = dict(zip(sorted(ao.user.unique()), range(len(ao.user.unique()))))
ao.user = ao.user.apply(lambda x: user_map[x])

arr = np.array([ao.groupby('item').apply(lambda x: len(x)).values])
np.savetxt(f'{dataset}{sparse}_rawpop.txt', arr)

ao = ao[ao.time > 12]
if args.noise_prop_t > 0:
    time_std = (ao.time.max() - ao.time.min()) * args.noise_prop_t
    shrink = 0.5 * (ao.time.max() - ao.time.min()) / (0.5 * (ao.time.max() - ao.time.min()) + 3 * time_std)
    mean_time = ao.time.mean()
    ao.time = (ao.time - mean_time) * shrink + mean_time
    ao.time = (ao.time + np.random.normal(0, time_std, ao.shape[0])).astype(int)

# month and week ids, these horizons can be changed based on dataset
try:
    ao['time2'] = ao.time.apply(lambda x: datetime.fromtimestamp(x))
except:
    ao['time2'] = ao.time.apply(lambda x: datetime.fromtimestamp(x/1000))

if args.day_shift:
    ao['time3'] = np.ceil(ao.time2.dt.dayofyear * 100 + ao.time2.dt.hour / args.t1_cutoff)
elif args.hour_shift:
    ao['time3'] = np.ceil(ao.time2.dt.dayofyear * 100 + ao.time2.dt.hour * 100 + ao.time2.dt.minute / args.t1_cutoff)
else:
    ao['time3'] = np.ceil(ao.time2.dt.year * 1000 + ao.time2.dt.dayofyear / args.t1_cutoff)
var_map = dict(zip(sorted(ao['time3'].unique()), range(len(ao['time3'].unique()))))
ao['time4'] = ao['time3'].apply(lambda x: var_map[x])
if args.day_shift:
    ao['time5'] = np.ceil(ao.time2.dt.dayofyear * 100 + ao.time2.dt.hour / args.t2_cutoff)
elif args.hour_shift:
    ao['time5'] = np.ceil(ao.time2.dt.dayofyear * 100 + ao.time2.dt.hour * 100 + ao.time2.dt.minute / args.t2_cutoff)
else:
    ao['time5'] = np.ceil(ao.time2.dt.year * 1000 + ao.time2.dt.dayofyear / args.t2_cutoff)
var_map = dict(zip(sorted(ao['time5'].unique()), range(len(ao['time5'].unique()))))
ao['time6'] = ao['time5'].apply(lambda x: var_map[x])
if args.stop_early:
    print(args.dataset.split('/')[-1], args.mode, args.sparse_val, ao.time4.min(), ao.time4.max(), ao.time6.min(), ao.time6.max())
    sys.exit()
# interaction matrix processed by model with time embedding
ao.sort_values(['time2'])[['user', 'item', 'time4', 'time6', 'time']].drop_duplicates().to_csv(f'{dataset}{sparse}_intwtime{args.extra2}.csv', header=False, index=False)
# interaction matrix processed by model without time embedding
ao.sort_values(['time2'])[['user', 'item', 'time4', 'time6']].drop_duplicates().to_csv(f'{dataset}{sparse}_int2{args.extra2}.csv', header=False, index=False)
print("saved interaction matrix")

if args.last_pop:
    df = pd.read_csv(f'{dataset}{sparse}_int2{args.extra2}.csv', header=None, index_col=False)
    df.columns=['user', 'item', 'time', 'time1']
    ctr = Counter(df['user']+1)
    res = [0] + list(itemgetter(*np.arange(1, df['user'].max()+2))(ctr))
    np.savetxt(f"{dataset}_lastuserpop.txt", res)
    ctr2 = Counter(df['item'])
    res_dict = df.groupby('user')['item'].last()
    final_dict = {u: ctr2[v]+1 for u, v in res_dict.items()}
    res2 = [0] + list(itemgetter(*np.arange(0, df['user'].max()+1))(final_dict))
    np.savetxt(f"{dataset}_lastitempop.txt", res2)
    sys.exit()

if args.week_adj:
    otmpw = np.loadtxt(f"{dataset}{sparse}_week_curr_raw{args.extra2}.txt")
    with open(f"{dataset}_userneg.pickle", 'rb') as handle:
        usernegs = pickle.load(handle)
    last = ao.groupby('user').last()
    users = sorted(ao.user.unique())
    num = 6 if args.mode2 == "orig" else 7
    df = np.zeros((num * len(users), 101))
    last_values = last.to_dict('index')
    start = datetime.now()
    for u in users:
        last_u = last_values[u]
        counter = Counter(ao[(ao['time6'] == last_u['time6']) & (ao['time'] < last_u['time'])]['item'])
        arr = np.array(usernegs[u + 1]) - 1
        arr = np.insert(arr, 0, last.iloc[u]['item'])
        counts = np.array(itemgetter(*arr)(counter))
        urow = otmpw[last_u['time6'] - 1]
        urow[arr] += counts
        percs = 100 * rankdata(urow, "average") / len(urow)
        if args.mode2 == "orig":
            df[(args.t2_size + 1) * u:(args.t2_size + 1) * u + (args.t2_size + 1)] = np.array([pop_embed(perc, args.t2_size) for perc in percs[arr]]).T
        elif args.mode2 == "sin":
            df[7 * u:7 * u + 7] = np.array([position_encoding_basis2(perc) for perc in percs[arr]]).T
        elif args.mode2 == "perc":
            df[u:u + 1] = np.array(percs[arr])/100
    if args.mode2 == "orig":
        np.savetxt(f"{dataset}{sparse}_week_wt_embed_adj{args.extra2}.txt", df)
    elif args.mode2 == "sin":
        np.savetxt(f"{dataset}{sparse}_week_wtembed_pos_adj{args.extra2}.txt", df)
    elif args.mode2 == "perc":
        np.savetxt(f"{dataset}{sparse}_week_wt_perc_adj{args.extra2}.txt", df)
    sys.exit()

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

if not args.not_coarse:

    ototaldft3 = pd.DataFrame(columns=["time4", "item", "perc"])
    counter = Counter()
    for i, ints in grouped:
        counter = Counter({k:args.weight*v for k,v in counter.items()})
        counter.update(ints.item)
        vals = list(counter.values())
        percs = 100 * rankdata(vals, "average") / len(vals)
        if args.reverse:
            percs = 100 - percs
        noise_mask = np.random.rand(*percs.shape) < args.noise_p
        percs[noise_mask] += np.random.normal(loc=0, scale=args.noise_std, size=percs[noise_mask].shape)
        percs = np.clip(percs, 0, 100)
        item_orders = list(counter.keys())
        left = list(set(items) - set(item_orders))
        df = pd.DataFrame({"time4": [i for _ in range(len(items))], "item": item_orders + left, "perc": np.concatenate((percs, np.zeros(len(left))))})
        ototaldft3 = pd.concat([ototaldft3, df])

    # np.savetxt(f"{dataset}{sparse}_currpop.txt", ototaldft)
    # np.savetxt(f"{dataset}{sparse}_cumpop.txt", ototaldft2)
    print(ototaldft3.shape, "ototaldft3")
    np.savetxt(f"{dataset}{sparse}_{args.name}pop{args.extra2}.txt", ototaldft3)
    print("saved monthly popularity percentiles")

    # construct simple popularity feature based on each of 3 methods

    # otmp = ototaldft.pivot(index = 'time4', columns = 'item', values='perc')
    # otmp_ = otmp.apply(lambda x: list(itertools.chain.from_iterable([pop_embed(p, args.t1_size) for p in x])))
    # np.savetxt(f"{dataset}{sparse}_currembed.txt", otmp_.values)
    # otmp2 = ototaldft2.pivot(index = 'time4', columns = 'item', values='perc')
    # np.savetxt(f"{dataset}{sparse}_rawpop.txt", otmp2)
    # otmp2_ = otmp2.apply(lambda x: list(itertools.chain.from_iterable([pop_embed(p, args.t1_size) for p in x])))
    # np.savetxt(f"{dataset}{sparse}_cumembed.txt", otmp2_.values)
    otmp3 = ototaldft3.pivot(index = 'time4', columns = 'item', values='perc')
    print(otmp3.shape, "otmp3")
    if args.mode2 == "orig":
        if not args.use_perc:
            otmp3_ = otmp3.apply(lambda x: list(itertools.chain.from_iterable([pop_embed(p, args.t1_size) for p in x])))
        else:
            otmp3_ = otmp3
        print(otmp3_.shape, "otmp3_")
        np.savetxt(f"{dataset}{sparse}_{args.name}embed{args.extra2}.txt", otmp3_.values)
    elif args.mode2 == "sin":
        otmp3_pb = otmp3.apply(lambda x: list(itertools.chain.from_iterable([position_encoding_basis(p) for p in x])))
        np.savetxt(f"{dataset}{sparse}_{args.name}embed_pos2{args.extra2}.txt", otmp3_pb.values)
    elif args.mode2 == "perc":
        np.savetxt(f"{dataset}{sparse}_{args.name}perc{args.extra2}.txt", otmp3.values/100)
    print("saved coarse popularity embeddings")


if not args.not_fine:
    # capture previous 4 weeks popularity (if we're at January 30th don't want to lose January 1-January 28 data)
    ototaldftw = pd.DataFrame(columns=["time6", "item", "perc", "vals"])
    grouped = ao.groupby('time6')
    counter = Counter()
    for i, ints in grouped:
        if i >= 4:
            counter.subtract(prev4)
        counter.update(ints.item)
        vals = list(counter.values())
        percs = 100 * rankdata(vals, "average") / len(vals)
        if args.reverse:
            percs = 100 - percs
        percs += np.random.normal(loc=0, scale=args.noise_std, size=percs.shape)
        percs = np.clip(percs, 0, 100)
        item_orders = list(counter.keys())
        left = list(set(items) - set(item_orders))
        df = pd.DataFrame({"time6": [i for _ in range(len(items))], "item": item_orders + left, "perc": np.concatenate((percs, np.zeros(len(left)))), "vals": np.concatenate((vals, np.zeros(len(left))))})
        ototaldftw = pd.concat([ototaldftw, df])
        if i >= 3:
            prev4 = prev3
        if i >= 2:
            prev3 = prev2
        if i >= 1:
            prev2 = prev1
        prev1 = ints.item
    # simple popularity feature w/ lower dimension to reduce time/space
    oraw = ototaldftw.pivot(index = 'time6', columns = 'item', values='vals')
    np.savetxt(f"{dataset}{sparse}_week_curr_raw{args.extra2}.txt", oraw.values)
    otmpw = ototaldftw.pivot(index = 'time6', columns = 'item', values='perc')
    if args.mode2 == "orig":
        otmpw_ = otmpw.apply(lambda x: list(itertools.chain.from_iterable([pop_embed(p, args.t2_size) for p in x])))
        np.savetxt(f"{dataset}{sparse}_week_embed2{args.extra2}.txt", otmpw_.values)
    elif args.mode2 == "sin":
        otmpw_pb = otmpw.apply(lambda x: list(itertools.chain.from_iterable([position_encoding_basis2(p) for p in x])))
        np.savetxt(f"{dataset}{sparse}_weekembed_pos2{args.extra2}.txt", otmpw_pb.values)
    if args.mode2 == "perc":
        np.savetxt(f"{dataset}{sparse}_week_perc{args.extra2}.txt", otmpw.values/100)
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
