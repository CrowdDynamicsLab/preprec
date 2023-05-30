import os
import time
import torch
import argparse
import pdb 

from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', default='/default', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only',  action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)

parser.add_argument('--monthpop', default='wtembed', type=str)
parser.add_argument('--weekpop', default='week_embed2', type=str)
parser.add_argument('--base_dim1', default=11, type=int)
parser.add_argument('--input_units1', default=132, type=int)
parser.add_argument('--base_dim2', default=6, type=int)
parser.add_argument('--input_units2', default=24, type=int)
parser.add_argument('--seed', default=2023, type=int)
parser.add_argument('--augment',  action='store_true')
parser.add_argument('--transfer',  action='store_true')

args = parser.parse_args()

write = 'res/' + args.dataset + args.train_dir + '/'
if not os.path.isdir(write):
    os.makedirs(write)
with open(os.path.join(write, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed) 

    # global dataset
    if args.augment:
        dataset = data_partition(args.dataset, args.maxlen)
        [user_train, user_valid, user_test, usernum, itemnum, user_dict] = dataset
    else:
        dataset = data_partition(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    # cc = 0.0
    # for u in user_train:
    #     cc += len(user_train[u])
    # print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(write, 'log.txt'), 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, augment=args.augment)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        if name == 'embed_layer.fc1.bias' or name == 'embed_layer.fc12.bias':
            torch.nn.init.zeros_(param.data)
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            if args.transfer:
                loaded = torch.load(args.state_dict_path, map_location=torch.device(args.device))
                for key in ['popularity_enc.month_pop_table', 'popularity_enc.week_pop_table', 'position_enc.pos_table']:
                    del loaded[key]
                model_dict = model.state_dict()
                model_dict.update(loaded)
                model.load_state_dict(model_dict)
#                for key in model.state_dict().keys():
#                    if key == 'last_layernorm.weight':
#                        pdb.set_trace()
#                    if key not in 
#                        model.state_dict()[key] = loaded[key]
                args.inference_only = True
            else:
                model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
                tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
                epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, time1, time2, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, time1, time2, pos, neg = np.array(u), np.array(seq), np.array(time1), np.array(time2), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, time1, time2, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
    
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(write, fname))

    if not args.inference_only:
        fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        torch.save(model.state_dict(), os.path.join(write, fname))
    
    f.close()
    sampler.close()
    print("Done")
