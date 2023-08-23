import os
import time
import torch
import argparse
import pdb 
from scipy.spatial import distance_matrix

from parse import parse
from model import SASRec, NewRec, NewB4Rec, BERT4Rec, BPRMF
from utils import *

from train_test import train_test

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

args = parse()

if args.max_split_size != -1.0:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{str(args.max_split_size)}"

write = 'res/' + args.dataset + '/' + args.train_dir + '/'
if not os.path.isdir(write):
    os.makedirs(write)
with open(os.path.join(write, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed) 

    unordered = ['bprmf']
    no_use_time = ['sasrec', 'bert4rec']
    use_time = ['newrec', 'newb4rec', 'mostpop']

    # global dataset
    if args.model in no_use_time:
        dataset = data_partition2(args.dataset, args.wrong_num)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
    elif args.model in use_time:
        if args.model in ['newrec', 'newb4rec'] and args.augment:
            dataset = data_partition(args.dataset, args.maxlen, None if args.augfulllen == 0 else args.augfulllen)
            [user_train, user_valid, user_test, usernum, itemnum, user_dict] = dataset
        else:
            dataset = data_partition(args.dataset)
            [user_train, user_valid, user_test, usernum, itemnum] = dataset
    elif args.model in unordered:
        dataset = data_partition3(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset

    print("done loading data!")

    num_batch = len(user_train) // args.batch_size


    # no training needed for most popular rec
    if args.model == 'mostpop':
        t_test = evaluate(None, dataset, args) 
        for i, k in enumerate(args.topk):
            print(f"{args.mode} (NDCG@{k}: {t_test[i][0]}, HR@{k}: {t_test[i][1]})")
        sys.exit() 
    
    sampler = WarpSampler(user_train, usernum, itemnum, args.model, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, mask_prob = args.mask_prob, augment=args.augment)
    if args.model == 'sasrec':
        model = SASRec(usernum, itemnum, args).to(args.device)
    elif args.model == 'newrec':
        model = NewRec(usernum, itemnum, args).to(args.device)
    elif args.model == 'newb4rec':
        model = NewB4Rec(itemnum, itemnum//args.loss_size, args).to(args.device)
    elif args.model == 'bert4rec':
        model = BERT4Rec(itemnum, args).to(args.device)
    elif args.model == 'bprmf':
        model = BPRMF(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        if name == 'embed_layer.fc1.bias' or name == 'embed_layer.fc12.bias': # for newrec model only
            torch.nn.init.zeros_(param.data)
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 
    
    print("done data sampling / model setup!")
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            if args.transfer or args.fs_transfer: # for newrec model only
                loaded = torch.load(args.state_dict_path, map_location=torch.device(args.device))
                # preprocessing specific to each dataset isn't transferred
                for key in ['popularity_enc.month_pop_table', 'popularity_enc.week_pop_table', 'position_enc.pos_table']:
                    del loaded[key]
                model_dict = model.state_dict()
                model_dict.update(loaded)
                model.load_state_dict(model_dict)
                if args.transfer:
                    args.inference_only = True
                if args.fs_transfer:
                    args.num_epochs = args.fs_num_epochs
            else:
                model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)), strict = False)
                tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
                epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print("done loading model")
        except: 
            raise ValueError('loading state dict failed')
    
    train_test(args, sampler, num_batch, model, dataset, epoch_start_idx, write)

