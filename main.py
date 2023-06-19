import os
import time
import torch
import argparse
import pdb 

from model import SASRec, NewRec, BERT4Rec, BPRMF
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='amazon/amazon_tool | amazon/amazon_office')
parser.add_argument('--train_dir', default='test', type=str, help='directory to write model to')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float, help = 'weight of l2 loss of embedding')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only',  action='store_true')
parser.add_argument('--mode', default='valid', type=str, help='valid | test')
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--model', default='newrec', type=str, help='newrec | mostpop | sasrec | bert4rec | bprmf')
parser.add_argument('--monthpop', default='wtembed', type=str, help='format of month popularity: wtembed (time-weighted) | currembed (current month) | cumembed (cumulative)')
parser.add_argument('--weekpop', default='week_embed2', type=str, help='format of week popularity: current is 4-week popularity')
parser.add_argument('--rawpop', default='cumpop', type=str, help='format of popularity for mostpop model: current is cumulative')
parser.add_argument('--userpop', default='lastuserpop', type=str, help='ultimate user popularity used if eval_quality true')
parser.add_argument('--base_dim1', default=11, type=int, help='dimension of month popularity vector, newrec only')
parser.add_argument('--input_units1', default=132, type=int, help='base_dim1 * number of months considered, newrec only')
parser.add_argument('--base_dim2', default=6, type=int, help='dimension of week popularity vector, newrec only')
parser.add_argument('--input_units2', default=6, type=int, help='base_dim2 * number of weeks considered, newrec only')
parser.add_argument('--mask_prob', default=0, type=float, help='cloze task, bert4rec only')
parser.add_argument('--seed', default=2023, type=int)
parser.add_argument('--topk','--list', nargs='+', default=[10], type=int, help='# items for evaluation')
parser.add_argument('--augment', action='store_true', help='use data augmentation, newrec only')
parser.add_argument('--transfer', action='store_true', help='zero-shot transfer, newrec only')
parser.add_argument('--max_split_size', default=-1.0, type=float)
parser.add_argument('--no_fixed_emb', action='store_true', help='for now, available in newrec only')
parser.add_argument('--eval_method', default=1, type=int, help='1: random 100-size subset, 2: popularity 100-size subset, 3: full set')
parser.add_argument('--eval_quality', action='store_true', help='evaluate across groups of user popularity')
parser.add_argument('--quality_size', default=10, type=int, help='percentile size of group if eval_quality is True')
parser.add_argument('--wrong_num', action='store_true', help='0-index users & items (for testing old runs)')

args = parser.parse_args()

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
    use_time = ['newrec', 'mostpop']

    # global dataset
    if args.model in no_use_time:
        dataset = data_partition2(args.dataset, args.wrong_num)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
    elif args.model in use_time:
        if args.model == 'newrec' and args.augment:
            dataset = data_partition(args.dataset, args.maxlen)
            [user_train, user_valid, user_test, usernum, itemnum, user_dict] = dataset
        else:
            dataset = data_partition(args.dataset)
            [user_train, user_valid, user_test, usernum, itemnum] = dataset
    elif args.model in unordered:
        dataset = data_partition3(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset

    print("done loading data!")

    num_batch = len(user_train) // args.batch_size

    f = open(os.path.join(write, 'log.txt'), 'w')

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
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            if args.transfer: # for newrec model only
                loaded = torch.load(args.state_dict_path, map_location=torch.device(args.device))
                # preprocessing specific to each dataset isn't transferred
                for key in ['popularity_enc.month_pop_table', 'popularity_enc.week_pop_table', 'position_enc.pos_table']:
                    del loaded[key]
                model_dict = model.state_dict()
                model_dict.update(loaded)
                model.load_state_dict(model_dict)
                args.inference_only = True
            else:
                model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
                tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
                epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: 
            raise ValueError('loading state dict failed')
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args) 
        for i, k in enumerate(args.topk):
            print(f"{args.mode} (NDCG@{k}: {t_test[i][0]}, HR@{k}: {t_test[i][1]})")
    
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        if args.model == 'sasrec':
            bce_criterion = torch.nn.BCEWithLogitsLoss()
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        elif args.model == 'bert4rec':
            ce = torch.nn.CrossEntropyLoss(ignore_index=0)
            for step in range(num_batch):
                seqs, labels = sampler.next_batch()
                seqs, labels = torch.LongTensor(seqs), torch.LongTensor(labels).to(args.device).view(-1)
                logits = model(seqs)
                adam_optimizer.zero_grad()
                # indices = torch.where(labels != 0)
                loss = ce(logits, labels)
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        elif args.model == 'newrec':
            bce_criterion = torch.nn.BCEWithLogitsLoss()
            for step in range(num_batch):
                u, seq, time1, time2, pos, neg = sampler.next_batch() 
                u, seq, time1, time2, pos, neg = np.array(u), np.array(seq), np.array(time1), np.array(time2), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(seq, time1, time2, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) 

        elif args.model == 'bprmf':
            # bce_criterion = torch.nn.BCEWithLogitsLoss()
            for step in range(num_batch):
                u, pos, neg = sampler.next_batch() 
                u, pos, neg = np.array(u), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(u, pos, neg)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = - (pos_logits[indices] - neg_logits[indices]).sigmoid().log().sum()
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) 
    
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_valid = evaluate(model, dataset, args)
            print(f"epoch:{epoch}, time: {T} (NDCG@{args.topk[0]}: {t_valid[0][0]}, HR@{args.topk[0]}: {t_valid[0][1]})")
    
            f.write(str(t_valid[0][0]) + ' ' + str(t_valid[0][1]) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

            fname = '{}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.model, epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(write, fname))

    if not args.inference_only:
        fname = '{}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        fname = fname.format(args.model, args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        torch.save(model.state_dict(), os.path.join(write, fname))
    
    f.close()
    sampler.close()
    print("Done")
