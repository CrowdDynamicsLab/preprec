import os
import time
import torch
from scipy.spatial import distance_matrix

from utils import *


def train_test(args, sampler, num_batch, model, dataset, epoch_start_idx, write, usernegs, second, sampler2, num_batch2, model2, dataset2, usernegs2):
    f = open(os.path.join(write, "log.txt"), "w")
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.wd
    )
    if second:
        adam_optimizer2 = torch.optim.Adam(
            model2.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.wd
        )

    T = 0.0
    t0 = time.time()

    # add regularization
    if args.triplet_loss or args.cos_loss:
        user_feat = np.loadtxt(f"./data{args.dataset}_{args.reg_file}.txt")

    best_ndcg = 0
    best_state = model.state_dict()
    stop_early = 0
    if args.first_eval:
        model.eval()
        mode = "valid" if not args.sparse or args.override_sparse else "test"
        t_valid = evaluate(model, dataset, args, mode, usernegs)
        model.train()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break  # just to decrease identition
        if args.model == "sasrec":
            bce_criterion = torch.nn.BCEWithLogitsLoss()
            for step in range(num_batch):
                # get batch data
                u, seq, pos, neg = sampler.next_batch()
                u, seq, pos, neg = (
                    np.array(u),
                    np.array(seq),
                    np.array(pos),
                    np.array(neg),
                )
                # model output
                pos_logits, neg_logits = model(seq, pos, neg)
                pos_labels, neg_labels = torch.ones(
                    pos_logits.shape, device=args.device
                ), torch.zeros(neg_logits.shape, device=args.device)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                # loss function 
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        elif args.model == "bert4rec":
            ce = torch.nn.CrossEntropyLoss(ignore_index=0)
            for step in range(num_batch):
                # get batch data
                seqs, labels = sampler.next_batch()
                seqs, labels = torch.LongTensor(seqs), torch.LongTensor(labels).to(
                    args.device
                ).view(-1)
                # model output
                logits = model(seqs)
                adam_optimizer.zero_grad()
                # loss function 
                loss = ce(logits, labels)
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        elif args.model == "newrec":
            bce_criterion = torch.nn.BCEWithLogitsLoss()
            for step in range(int(num_batch*args.fs_prop)):
                # batch data based on if relative time encodings are used
                if not args.time_embed:
                    u, seq, time1, time2, pos, neg = sampler.next_batch()
                    u, seq, time1, time2, pos, neg = (np.array(u), np.array(seq), np.array(time1), np.array(time2), np.array(pos), np.array(neg))
                    time_embed = None
                else:
                    u, seq, time1, time2, time_embed, pos, neg = sampler.next_batch()
                    u, seq, time1, time2, time_embed, pos, neg = (np.array(u), np.array(seq), np.array(time1), np.array(time2), np.array(time_embed), np.array(pos), np.array(neg))
                # find closest and furthest user pairs within sample for regularization
                if args.triplet_loss or args.cos_loss:
                    batch_dist = distance_matrix(user_feat.T[u - 1], user_feat.T[u - 1])
                    pos_user = np.argpartition(batch_dist, args.reg_num)[
                        :, : args.reg_num
                    ]
                    neg_user = np.argpartition(-batch_dist, args.reg_num)[
                        :, : args.reg_num
                    ]
                else:
                    pos_user = np.array([])
                    neg_user = np.array([])
                # model output 
                pos_logits, neg_logits, embed, pos_embed, neg_embed = model(
                    u, seq, time1, time2, time_embed, pos, neg, pos_user, neg_user
                )
                pos_labels, neg_labels = torch.ones(
                    pos_logits.shape, device=args.device
                ), torch.zeros(neg_logits.shape, device=args.device)
                adam_optimizer.zero_grad()
                # loss function, split into regularization and BCE
                loss = 0
                if args.only_reg:
                    bceloss = 0
                else:
                    indices = np.where(pos != 0)
                    loss += bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                    bceloss = loss.item()
                # loss += args.reg_coef * model.regloss(
                #     embed, pos_embed, neg_embed, args.triplet_loss, args.cos_loss
                # )
                loss.backward()
                adam_optimizer.step()
                print(
                    "loss in epoch {} iteration {}: bce {} reg {}".format(
                        epoch, step, bceloss, loss.item() - bceloss
                    )
                )

            # repeat above for newrec if second dataset is concurrently trained
            if second:
                # transfer updated parameters from first to second models
                model1_dict = {k: v for k, v in model.state_dict().items() if k not in ["popularity_enc.month_pop_table", 
                    "popularity_enc.week_pop_table", "position_enc.pos_table", "user_enc.act_table"]}
                model2_dict = model2.state_dict()
                model2_dict.update(model1_dict)
                model2.load_state_dict(model2_dict)
                for step in range(num_batch):
                    if not args.time_embed:
                        u, seq, time1, time2, pos, neg = sampler2.next_batch()
                        u, seq, time1, time2, pos, neg = (np.array(u), np.array(seq), np.array(time1), np.array(time2), np.array(pos), np.array(neg))
                        time_embed = None
                    else:
                        u, seq, time1, time2, time_embed, pos, neg = sampler2.next_batch()
                        u, seq, time1, time2, time_embed, pos, neg = (np.array(u), np.array(seq), np.array(time1), np.array(time2), np.array(time_embed), np.array(pos), np.array(neg))
                    pos_logits, neg_logits, embed, pos_embed, neg_embed = model2(
                        u, seq, time1, time2, time_embed, pos, neg, np.array([]), np.array([])
                    )
                    pos_labels, neg_labels = torch.ones(
                        pos_logits.shape, device=args.device
                    ), torch.zeros(neg_logits.shape, device=args.device)
                    adam_optimizer.zero_grad()
                    loss = 0
                    indices = np.where(pos != 0)
                    loss += bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                    bceloss = loss.item()
                    loss.backward()
                    adam_optimizer.step()
                    print(
                        "loss in epoch {} iteration {} dataset 2: bce {} reg {}".format(
                            epoch, step, bceloss, loss.item() - bceloss
                        )
                    )
                # transfer updated parameters from second to first models
                model2_dict = {k: v for k, v in model2.state_dict().items() if k not in ["popularity_enc.month_pop_table", "popularity_enc.week_pop_table", "position_enc.pos_table", "user_enc.act_table"]}
                model1_dict = model.state_dict()
                model1_dict.update(model2_dict)
                model.load_state_dict(model1_dict)
                    

        elif args.model == "newb4rec":
            ce = torch.nn.CrossEntropyLoss(ignore_index=0)
            for step in range(num_batch):
                # get batch data
                seqs, labels, t1, t2 = sampler.next_batch()
                seqs, labels, t1, t2 = (
                    np.array(seqs),
                    torch.LongTensor(labels).to(args.device).view(-1),
                    np.array(t1),
                    np.array(t2),
                )
                # model output
                logits = model(seqs, t1, t2)
                adam_optimizer.zero_grad()
                # loss function 
                loss = ce(
                    logits[labels != 0],
                    torch.full(labels[labels != 0].shape, logits.shape[1] - 1).to(
                        args.device
                    ),
                )
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        elif args.model == "bprmf":
            for step in range(num_batch):
                # get batch data
                u, pos, neg = sampler.next_batch()
                u, pos, neg = np.array(u), np.array(pos), np.array(neg)
                # model output
                pos_logits, neg_logits = model(u, pos, neg)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                # loss function
                loss = (-(pos_logits[indices] - neg_logits[indices]).sigmoid().log().sum())
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        elif args.model == "cl4srec":
            bce_criterion = torch.nn.BCEWithLogitsLoss()
            for step in range(num_batch):
                # get batch data
                seqs, lens, pos, neg = sampler.next_batch()
                seqs, lens, pos, neg = (
                    np.array(seqs),
                    torch.LongTensor(np.array(lens)).to(args.device),
                    np.array(pos),
                    np.array(neg),
                )

                pos_logits, neg_logits, aug_loss = model(seqs, lens, pos, neg)
                pos_labels, neg_labels = torch.ones(
                    pos_logits.shape, device=args.device
                ), torch.zeros(neg_logits.shape, device=args.device)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                loss += args.aug_coef * aug_loss
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        # validation and check early stopping
        if epoch % args.epoch_test == 0:
            t1 = time.time() - t0
            T += t1
            t0 = time.time()
            # get validation results
            model.eval()
            # model.popularity_enc = model.popularity_enc.to("cpu")
            # model.eval_popularity_enc = model.eval_popularity_enc.to("cuda")
            mode = "valid" if not args.sparse or args.override_sparse else "test"
            t_valid = evaluate(model, dataset, args, mode, usernegs)
            # model.eval_popularity_enc = model.eval_popularity_enc.to("cpu")
            # model.popularity_enc = model.popularity_enc.to("cuda")
            model.train()
            ndcg, hr = t_valid[0][0], t_valid[0][1]
            f.write(f"epoch:{epoch}, time: {T} (NDCG@{args.topk[0]}: {ndcg}, HR@{args.topk[0]}: {hr})" + "\n")
            f.flush()
            if second:
                model2.eval()
                t_valid2 = evaluate(model2, dataset2, args, mode, usernegs2, True)
                model2.train()
                ndcg2, hr2 = t_valid2[0][0], t_valid2[0][1]
                f.write(f"Validation at epoch:{epoch}, time: {T}, dataset 2: (NDCG@{args.topk[0]}: {ndcg2}, HR@{args.topk[0]}: {hr2})" + "\n")
                f.flush()
                ndcg = (ndcg + ndcg2)/2

            fname = f"epoch={epoch}.pth"
            if epoch % (args.epoch_test) == 0:
                torch.save(model.state_dict(), os.path.join(write, fname))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_state = model.state_dict()
                stop_early = 0
            else:
                stop_early += 1

        # stop if 3 consecutive validations without improving ndcg
        if stop_early == args.stop_early:
            break
    
    if best_ndcg != 0:
        fname = "best.pth"
        torch.save(best_state, os.path.join(write, fname))

    # testing
    if args.inference_only or not args.train_only:
        model.eval()
        # # model.popularity_enc = model.popularity_enc.to("cpu")
        # model.handle_inference()
        # model.eval_popularity_enc = model.eval_popularity_enc.to("cuda")
        # if we've trained, used best training parameters
        if not args.inference_only and not args.state_override:
            model_dict = model.state_dict()
            model_dict.update(best_state)
            model.load_state_dict(model_dict)
        f.write("\nTest results:\n")
        f.flush()
        t_test = evaluate(model, dataset, args, "test", usernegs)
        for i, k in enumerate(args.topk):
            f.write(f"NDCG@{k}: {t_test[i][0]}, HR@{k}: {t_test[i][1]} \n")
            f.flush()
        if second:
            model2.eval()
            # # model2.popularity_enc = model2.popularity_enc.to("cpu")
            # model2.handle_inference()
            # model2.eval_popularity_enc = model2.eval_popularity_enc.to("cpu")
            t_test2 = evaluate(model2, dataset2, args, "test", usernegs2, True)
            for i, k in enumerate(args.topk):
                f.write(f"NDCG@{k}: {t_test2[i][0]}, HR@{k}: {t_test2[i][1]} \n")
                f.flush()

    f.close()
    if sampler:
        sampler.close()
    print("Done")