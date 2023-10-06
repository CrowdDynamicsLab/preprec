import os
import time
import torch
import argparse
import pdb
from scipy.spatial import distance_matrix

from parse import parse
from model import SASRec, NewRec, NewB4Rec, BERT4Rec, BPRMF
from utils import *
from data import *
from train_test import train_test
import pickle

args = parse()

if args.max_split_size != -1.0:
    os.environ[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = f"max_split_size_mb:{str(args.max_split_size)}"

write = "res/" + args.dataset + "/" + args.train_dir + "/"
if not os.path.isdir(write):
    os.makedirs(write)
with open(os.path.join(write, "args.txt"), "w") as f:
    f.write(
        "\n".join(
            [
                str(k) + "," + str(v)
                for k, v in sorted(vars(args).items(), key=lambda x: x[0])
            ]
        )
    )
f.close()

# for sparse dataset run
if args.sparse:
    args.monthpop = "sparse_" + args.monthpop
    args.weekpop = "sparse_" + args.weekpop
    args.week_eval_pop = "sparse_" + args.week_eval_pop

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

no_use_time = ["sasrec", "bert4rec", "bprmf"]
use_time = ["newrec", "newb4rec", "mostpop"]

# pull data 
second = False 
if args.model in no_use_time:
    dataset = data_partition2(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
elif args.model in use_time:
    if args.time_embed:
        dataset = data_partition_wtime(args.dataset, args.maxlen, sparse=args.sparse)
    else:
        dataset = data_partition(args.dataset, args.maxlen, sparse=args.sparse)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    if args.dataset2 != "":
        dataset2 = data_partition(args.dataset2, args.maxlen, sparse=args.sparse)
        [user_train2, user_valid2, user_test2, usernum2, itemnum2] = dataset2
        second = True

# randomly fix negative test samples
mod = '' if not args.sparse else '_sparse'
if args.save_neg:
    setup_negatives(dataset, args.dataset, mod)
with open(f"../data/{args.dataset}{mod}_userneg.pickle", 'rb') as handle:
    usernegs = pickle.load(handle)
if second:
    with open(f"../data/{args.dataset2}{mod}_userneg.pickle", 'rb') as handle:
        usernegs2 = pickle.load(handle)

print(f"done loading data for {args.dataset}!")

if args.model == "newrec":
    num_batch = len(user_train[0]) // args.batch_size
    if second:
        num_batch2 = len(user_train2[0]) // args.batch_size
else:
    num_batch = len(user_train) // args.batch_size

# no training needed for most popular rec
if args.model == "mostpop":
    t_test = evaluate(None, dataset, args, "test", usernegs)
    for i, k in enumerate(args.topk):
        print(f"{args.mode} (NDCG@{k}: {t_test[i][0]}, HR@{k}: {t_test[i][1]})")
    sys.exit()

# positive (and negative if applicable) sampling for training
sampler = WarpSampler(user_train, usernum, itemnum, args.model, batch_size=args.batch_size, maxlen=args.maxlen,
    n_workers=3, mask_prob=args.mask_prob, augment=args.augment)
if second:
    sampler2 = WarpSampler(user_train2, usernum2, itemnum2, args.model, batch_size=args.batch_size, maxlen=args.maxlen,
        n_workers=3, mask_prob=args.mask_prob, augment=args.augment)

# model setup
if args.model == "sasrec":
    model = SASRec(usernum, itemnum, args).to(args.device)
elif args.model == "newrec":
    model = NewRec(usernum, itemnum, args).to(args.device)
    if second:
        model2 = NewRec(usernum, itemnum, args, second=True).to(args.device)
elif args.model == "newb4rec":
    model = NewB4Rec(itemnum, itemnum // args.loss_size, args).to(args.device)
elif args.model == "bert4rec":
    model = BERT4Rec(itemnum, args).to(args.device)
elif args.model == "bprmf":
    model = BPRMF(usernum, itemnum, args).to(args.device)

for name, param in model.named_parameters():
    if (
        name == "embed_layer.fc1.bias" or name == "embed_layer.fc12.bias"
    ):  # for newrec model only
        torch.nn.init.zeros_(param.data)
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass
if second:
    for name, param in model2.named_parameters():
        if (
            name == "embed_layer.fc1.bias" or name == "embed_layer.fc12.bias"
        ):  # for newrec model only
            torch.nn.init.zeros_(param.data)
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

print(f"done data sampling / {args.model} model setup!")

model.train()  
if second:
    model2.train()
epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        # for newrec model only
        if args.transfer or args.fs_transfer:  
            loaded = torch.load(
                args.state_dict_path, map_location=torch.device(args.device)
            )
            # preprocessing specific to each dataset isn't transferred
            loaded = {k: v for k, v in loaded.items() if k not in ["popularity_enc.month_pop_table", 
                "popularity_enc.week_pop_table", "position_enc.pos_table", "user_enc.act_table", "time_position_enc.pos_table", 
                "eval_popularity_enc.week_eval_pop", "eval_popularity_enc.month_pop_table", "eval_popularity_enc.week_pop_table"]}
            model_dict = model.state_dict()
            model_dict.update(loaded)
            model.load_state_dict(model_dict)
            if args.transfer:
                args.inference_only = True
            if args.fs_transfer:
                args.num_epochs = args.fs_num_epochs
        else:
            model.load_state_dict(
                torch.load(
                    args.state_dict_path, map_location=torch.device(args.device)
                ),
                strict=False,
            )
        print("done loading model")
    except:
        raise ValueError("loading state dict failed")

if not second:
    sampler2 = None
    num_batch2 = None
    model2 = None
    dataset2 = None
    usernegs2 = None
print("starting training/testing")
train_test(args, sampler, num_batch, model, dataset, epoch_start_idx, write, usernegs, second, sampler2, num_batch2, model2, dataset2, usernegs2)