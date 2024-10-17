import os
import torch
import random
from parse import parse
from model import SASRec, NewRec, NewB4Rec, BERT4Rec, BPRMF, CL4SRec
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
    args.monthpop = args.sparse_name + args.monthpop
    args.weekpop = args.sparse_name + args.weekpop
    args.week_eval_pop = args.sparse_name + args.week_eval_pop

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

no_use_time = ["sasrec", "bert4rec", "bprmf"]
no_use_time_track_len = ["cl4srec", "duorec"]
use_time = ["newrec", "newb4rec", "mostpop"]

# pull data 
second = False 
# if args.pause:
    # pdb.set_trace()
if args.model in no_use_time:
    dataset = data_partition2(args.dataset, args.sparse_name if args.sparse else '', args.override_sparse, args.time_df_mod)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
elif args.model in no_use_time_track_len:
    dataset = data_partition3(args.dataset, args.maxlen, args.sparse_name if args.sparse else '', args.override_sparse, args.time_df_mod)
    [user_train, user_valid, user_test, usernum, itemnum, userlens] = dataset
elif args.model in use_time:
    if args.time_embed:
        dataset = data_partition_wtime(args.dataset, args.maxlen, args.sparse_name if args.sparse else '', args.override_sparse, args.time_df_mod)
    else:
        dataset = data_partition(args.dataset, args.maxlen, args.sparse_name if args.sparse else '', args.override_sparse, args.time_df_mod)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    if args.dataset2 != "":
        if args.time_embed:
            dataset2 = data_partition_wtime(args.dataset2, args.maxlen, args.sparse_name if args.sparse else '', args.override_sparse, args.time_df_mod)
        else:
            dataset2 = data_partition(args.dataset2, args.maxlen, args.sparse_name if args.sparse else '', args.override_sparse, args.time_df_mod)
        [user_train2, user_valid2, user_test2, usernum2, itemnum2] = dataset2
        second = True

# evaluation negatives
mod = args.sparse_name if args.sparse else ''
userneg_mod = mod
with open(f"./data/{args.dataset}_{userneg_mod}{args.userneg}.pickle", 'rb') as handle:
    usernegs = pickle.load(handle)
if second:
    with open(f"./data/{args.dataset2}_{userneg_mod}{args.userneg}.pickle", 'rb') as handle:
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

sampler = None
sampler2 = None
if not args.inference_only:
    if args.model in ["cl4srec", "duorec"]:
        misc = userlens
        user_comb = user_train
    else:
        misc = None
        user_comb = user_train
    sampler = WarpSampler(user_comb, usernum, itemnum, args.model, batch_size=args.batch_size, maxlen=args.maxlen,n_workers=4, mask_prob=args.mask_prob, augment=args.augment, misc=misc)
    if second:
        sampler2 = WarpSampler(user_train2, usernum2, itemnum2, args.model, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=int(os.cpu_count()/2), mask_prob=args.mask_prob, augment=args.augment)
print(f"done training sampler for {args.dataset}!")

# model setup
if args.model == "sasrec":
    model = SASRec(usernum, itemnum, args).to(args.device)
elif args.model == "newrec":
    model = NewRec(usernum, itemnum, args).to(args.device)
    if second:
        model2 = NewRec(usernum, itemnum, args, second=True)
elif args.model == "newb4rec":
    model = NewB4Rec(itemnum, itemnum // args.loss_size, args).to(args.device)
elif args.model == "bert4rec":
    model = BERT4Rec(itemnum, args).to(args.device)
elif args.model == "bprmf":
    model = BPRMF(usernum, itemnum, args).to(args.device)
elif args.model == "cl4srec":
    model = CL4SRec(itemnum, args).to(args.device)

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

print(f"done {args.model} model setup!")

model.train()
if second:
    model2.train()
epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        # for newrec model only
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
        if args.fs_emb or args.fs_emb_2:
            for param in model.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if "fs_layer" in name:
                    param.requires_grad = True
            
        if args.transfer:
            args.inference_only = True
        if args.fs_transfer:
            args.num_epochs = args.fs_num_epochs
        print("done loading model")
    except:
        raise ValueError("loading state dict failed")
if args.model == "newrec":
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model.to(args.device)

if not second:
    num_batch2 = None
    model2 = None
    dataset2 = None
    usernegs2 = None
print("starting training/testing")
train_test(args, sampler, num_batch, model, dataset, epoch_start_idx, write, usernegs, second, sampler2, num_batch2, model2, dataset2, usernegs2)
