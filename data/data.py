import numpy as np
import pandas as pd
from datetime import *
from collections import Counter
from scipy.stats import rankdata
import itertools
import argparse
from multiprocessing import Pool, cpu_count

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./douban/douban_music', type=str)
    parser.add_argument('--weight', default=0.5, type=float, help='weight for exponential weighted average of percentile')
    parser.add_argument('--t1_cutoff', default=366/12, type=float, help='length of coarse period (by default in days)')
    parser.add_argument('--t1_size', default=10, type=int, help='number of coarse periods')
    parser.add_argument('--t2_cutoff', default=366/62, type=float, help='length of fine period (by default in days)')
    parser.add_argument('--t2_size', default=5, type=int, help='number of fine periods')
    parser.add_argument('--not_coarse',  action='store_true', help='do not compute coarse popularity')
    parser.add_argument('--not_fine',  action='store_true', help='do not compute fine popularity')
    parser.add_argument('--noise_std', default=0, type=int, help='standard deviation of noise added to percentiles')
    parser.add_argument('--noise_p', default=1, type=float, help='proportion of item percentiles to add noise to')
    parser.add_argument('--noise_prop_t', default=0, type=float, help='standard deviation of noise added to raw time')
    parser.add_argument('--day_shift',  action='store_true', help='change cutoff units to be in hours')
    parser.add_argument('--hour_shift',  action='store_true', help='change cutoff units to be in minutes')
    args = parser.parse_args()
    dataset = args.dataset

    # each row must have item, user, interaction/rating, time (as unix timestamp) in that order
    ao = pd.read_csv(f'{dataset}.csv')
    ao.columns=["item", "user", "rate", "time"]
    ao = ao.drop_duplicates(['item', 'user'])
    # k-core filtering
    ao = filter_tot(ao,k=5,u_name='user',i_name='item',y_name='rate')
    # user, item ids
    item_map = dict(zip(sorted(ao.item.unique()), range(len(ao.item.unique()))))
    ao.item = ao.item.apply(lambda x: item_map[x])
    user_map = dict(zip(sorted(ao.user.unique()), range(len(ao.user.unique()))))
    ao.user = ao.user.apply(lambda x: user_map[x])

    arr = np.array([ao.groupby('item').apply(lambda x: len(x)).values])
    np.savetxt(f'{dataset}_rawpop.txt', arr)

    ao = ao[ao.time > 12]
    if args.noise_prop_t > 0:
        time_std = (ao.time.max() - ao.time.min()) * args.noise_prop_t
        shrink = 0.5 * (ao.time.max() - ao.time.min()) / (0.5 * (ao.time.max() - ao.time.min()) + 3 * time_std)
        mean_time = ao.time.mean()
        ao.time = (ao.time - mean_time) * shrink + mean_time
        ao.time = (ao.time + np.random.normal(0, time_std, ao.shape[0])).astype(int)

    # handle seconds and milliseconds
    try:
        ao['time2'] = ao.time.apply(lambda x: datetime.fromtimestamp(x))
    except:
        ao['time2'] = ao.time.apply(lambda x: datetime.fromtimestamp(x/1000))

    # can change time granularity
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

    # interaction matrix processed by model with and without time embedding
    ao.sort_values(['time2'])[['user', 'item', 'time4', 'time6', 'time']].drop_duplicates().to_csv(f'{dataset}_intwtime.csv', header=False, index=False)
    ao.sort_values(['time2'])[['user', 'item', 'time4', 'time6']].drop_duplicates().to_csv(f'{dataset}_int2.csv', header=False, index=False)
    print("saved interaction matrix")

    items = sorted(ao.item.unique())

    if not args.not_coarse:
        # Compute popularity by exponential weighted average over periods
        def process_coarse_group(group_data):
            last_time, ints, items, args = group_data
            counter = {}
            for item, time4 in zip(ints.item, ints.time4):
                counter[item] = counter.get(item, 0) + args.weight ** (last_time - time4)
            vals = list(counter.values())
            percs = 100 * rankdata(vals, "average") / len(vals)
            percs = 100 * rankdata(vals, "average") / len(vals)
            noise_mask = np.random.rand(*percs.shape) < args.noise_p
            percs[noise_mask] += np.random.normal(loc=0, scale=args.noise_std, size=percs[noise_mask].shape)
            percs = np.clip(percs, 0, 100)
            item_orders = list(counter.keys())
            left = list(set(items) - set(item_orders))
            df = pd.DataFrame({
                "time4": [last_time for _ in range(len(items))],
                "item": item_orders + left,
                "perc": np.concatenate((percs, np.zeros(len(left))))
            })
            return df

        grouped = ao.groupby('time4')
        unique_times = sorted(ao['time4'].unique())
        group_size = 32
        overlapping_groups = [unique_times[:i] for i in range(1, group_size)]
        overlapping_groups.extend([unique_times[i:i+group_size] for i in range(len(unique_times) - group_size + 1)])

        ototaldft3 = pd.DataFrame(columns=["time4", "item", "perc"])
        group_data = []
        for group_times in overlapping_groups:
            group_ints = ao[ao['time4'].isin(group_times)]
            group_data.append((group_times[-1], group_ints, items, args))
        res_0 = process_coarse_group(group_data[0])
        res_1 = process_coarse_group(group_data[1])
        with Pool(processes=int(cpu_count()/2)) as pool:
            results = pool.map(process_coarse_group, group_data)

        ototaldft3 = pd.concat([ototaldft3] + results, ignore_index=True)
        ototaldft3['time4'] = pd.to_numeric(ototaldft3['time4'], errors='coerce')
        ototaldft3['item'] = pd.to_numeric(ototaldft3['item'], errors='coerce')
        ototaldft3['perc'] = pd.to_numeric(ototaldft3['perc'], errors='coerce')
        np.savetxt(f"{dataset}_wtpop.txt", ototaldft3)

        otmp3 = ototaldft3.pivot(index = 'time4', columns = 'item', values='perc')
        def process_row(row):
            return [pop_embed(p, args.t1_size) for p in row]
        with Pool(processes=int(cpu_count()) - 1) as pool:
            results = pool.map(process_row, [row for _, row in otmp3.iterrows()])
        otmp3_ = np.array(results)
        otmp3_ = np.swapaxes(otmp3_, 1, 2)
        otmp3_ = otmp3_.reshape(-1, otmp3_.shape[-1])
        # def process_row(row):
        #     return pd.Series([item for sublist in [pop_embed(p, args.t1_size) for p in row] for item in sublist])
        # with Pool(processes=int(cpu_count()/2)) as pool:
        #     results = pool.map(process_row, [row for _, row in otmp3.iterrows()])
        # otmp3_ = pd.DataFrame(results)
        np.savetxt(f"{dataset}_wtembed.txt", otmp3_)
        print("saved coarse popularity embeddings")

    if not args.not_fine:
        # Compute popularity by average over periods
        def process_fine_group(group_data):
            last_time, group_items, all_items, args = group_data
            counter = Counter(group_items)
            vals = list(counter.values())
            percs = 100 * rankdata(vals, "average") / len(vals)
            percs += np.random.normal(loc=0, scale=args.noise_std, size=percs.shape)
            percs = np.clip(percs, 0, 100)
            item_orders = list(counter.keys())
            left = list(set(all_items) - set(item_orders))
            df = pd.DataFrame({"time6": [last_time for _ in range(len(items))], "item": item_orders + left, "perc": np.concatenate((percs, np.zeros(len(left)))), "vals": np.concatenate((vals, np.zeros(len(left))))})
            return df

        grouped = ao.groupby('time6')
        unique_times = sorted(ao['time6'].unique())
        group_size = 32 
        overlapping_groups = [unique_times[:i] for i in range(1, group_size)]
        overlapping_groups.extend([unique_times[i:i+group_size] for i in range(len(unique_times) - group_size + 1)])

        ototaldftw = pd.DataFrame(columns=["time6", "item", "perc", "vals"])
        group_data = []
        for group_times in overlapping_groups:
            group_ints = ao[ao['time6'].isin(group_times)].item
            group_data.append((group_times[-1], group_ints, items, args))
        with Pool(processes=int(cpu_count()/2)) as pool:
            results = pool.map(process_fine_group, group_data)
        ototaldftw = pd.concat([ototaldftw] + results, ignore_index=True)
        ototaldftw['time6'] = pd.to_numeric(ototaldftw['time6'], errors='coerce')
        ototaldftw['item'] = pd.to_numeric(ototaldftw['item'], errors='coerce')
        ototaldftw['perc'] = pd.to_numeric(ototaldftw['perc'], errors='coerce')
        ototaldftw['vals'] = pd.to_numeric(ototaldftw['vals'], errors='coerce')

        oraw = ototaldftw.pivot(index = 'time6', columns = 'item', values='vals')
        np.savetxt(f"{dataset}_week_curr_raw.txt", oraw.values)
        otmpw = ototaldftw.pivot(index = 'time6', columns = 'item', values='perc')
        def process_row(row):
            return [pop_embed(p, args.t2_size) for p in row]
        with Pool(processes=int(cpu_count()) - 1) as pool:
            results = pool.map(process_row, [row for _, row in otmpw.iterrows()])
        otmpw_ = np.array(results)
        otmpw_ = np.swapaxes(otmpw_, 1, 2)
        otmpw_ = otmpw_.reshape(-1, otmpw_.shape[-1])
        # def process_row(row):
        #     return pd.Series([item for sublist in [pop_embed(p, args.t2_size) for p in row] for item in sublist])
        # with Pool(processes=int(cpu_count()/2)) as pool:
        #     results = pool.map(process_row, [row for _, row in otmpw.iterrows()])
        # otmpw_ = pd.DataFrame(results)
        np.savetxt(f"{dataset}_week_embed2.txt", otmpw_)
        print("saved fine popularity embeddings")