import numpy as np
import torch
import pdb

# taken from https://github.com/pmixer/SASRec.pytorch/blob/master/model.py
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# taken from https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch/blob/master/models/bert_modules/utils/feed_forward.py
class PointWiseFeedForward2(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointWiseFeedForward2, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def GELU(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        return self.w_2(self.dropout(self.GELU(self.w_1(x))))


class InitFeedForward(torch.nn.Module):
    def __init__(self, input_units, hidden1 = 100, hidden2 = 50, dropout_rate = 0):
        super(InitFeedForward, self).__init__()

        self.fc1 = torch.nn.Linear(input_units, hidden1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden1, hidden2)

    def forward(self, inputs):
        outputs = self.fc2(self.dropout(self.relu(self.fc1(inputs))))
        return outputs


# adapted from https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py
class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(CausalMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) 
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs


# taken from https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch/blob/master/models/bert_modules/attention/multi_head.py 
class MultiHeadAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, hidden_size, head_num, dropout_rate):
        super(MultiHeadAttention, self).__init__()

        # We assume d_v always equals d_k
        self.d_k = hidden_size // head_num
        self.h = head_num

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.output_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, query, mask=None):
        key = query
        value = query
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / np.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        x, attn = torch.matmul(p_attn, value), p_attn

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


# taken from https://github.com/jadore801120/attention-is-all-you-need-pytorch
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()



class PopularityEncoding(torch.nn.Module):
    def __init__(self, args):
        super(PopularityEncoding, self).__init__()
        n_position = args.maxlen
        d_hid = args.hidden_units
        self.input1 = args.input_units1
        self.input2 = args.input_units2
        self.base_dim1 = args.base_dim1
        self.base_dim2 = args.base_dim2
        # table of fixed feature vectors for items by time, shape: (num_times*base_dim, num_items)
        month_pop = np.loadtxt(f'../data/{args.dataset}_{args.monthpop}.txt')
        week_pop = np.loadtxt(f'../data/{args.dataset}_{args.weekpop}.txt')
        # add zeros for the index-0 empty item placeholder and initial time period
        self.register_buffer('month_pop_table', torch.cat((torch.zeros((month_pop.shape[0] + self.input1 - self.base_dim1, 1)), torch.cat((torch.zeros((self.input1 - self.base_dim1, month_pop.shape[1])), torch.FloatTensor(month_pop)), dim=0)), dim=1))
        self.register_buffer('week_pop_table', torch.cat((torch.zeros((week_pop.shape[0] + self.input2 - self.base_dim2, 1)), torch.cat((torch.zeros((self.input2 - self.base_dim2, week_pop.shape[1])), torch.FloatTensor(week_pop)), dim=0)), dim=1))

    def forward(self, log_seqs, time1_seqs, time2_seqs):
        month_table_rows = torch.flatten(torch.flatten(torch.LongTensor(time1_seqs)).reshape((-1, 1))*self.base_dim1 + torch.arange(self.input1))
        month_table_cols = torch.repeat_interleave(torch.flatten(torch.LongTensor(log_seqs)), self.input1)
        week_table_rows = torch.flatten(torch.flatten(torch.LongTensor(time2_seqs)).reshape((-1, 1))*self.base_dim2 + torch.arange(self.input2))
        week_table_cols = torch.repeat_interleave(torch.flatten(torch.LongTensor(log_seqs)), self.input2)
        if torch.max(month_table_rows) >= self.month_pop_table.shape[0] or torch.max(month_table_cols) >= self.month_pop_table.shape[1] or torch.max(week_table_rows) >= self.week_pop_table.shape[0] or torch.max(week_table_cols) >= self.week_pop_table.shape[1]:
            raise IndexError('row or column accessed out-of-index in popularity table')
        month_pop = torch.reshape(self.month_pop_table[month_table_rows, month_table_cols], (log_seqs.shape[0], log_seqs.shape[1], self.input1))
        week_pop = torch.reshape(self.week_pop_table[week_table_rows, week_table_cols], (log_seqs.shape[0], log_seqs.shape[1], self.input2))
        return torch.cat((month_pop, week_pop), 2).clone().detach()



class NewRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(NewRec, self).__init__()
        assert args.input_units1 % args.base_dim1 == 0
        assert args.input_units2 % args.base_dim2 == 0

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.model = args.model
        self.no_fixed_emb = args.no_fixed_emb

        self.popularity_enc = PopularityEncoding(args) 
        self.embed_layer = InitFeedForward(args.input_units1 + args.input_units2, args.hidden_units*2, args.hidden_units)
        if self.no_fixed_emb:
            self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) 
        else:
            self.position_enc = PositionalEncoding(args.hidden_units, args.maxlen)

        if args.triplet_loss:
            self.triplet_loss = torch.nn.TripletMarginLoss(margin=0.0, p=2)
        if args.cos_loss:
            self.cos_loss = torch.nn.CosineEmbeddingLoss() # torch.nn.CosineSimilarity()

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = CausalMultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate,
                                                            self.dev)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs, time1_seqs, time2_seqs):
        # obtain popularity-based feature vectors for sequence history, apply embedding layer, add positional encoding
        seqs = self.popularity_enc(log_seqs, time1_seqs, time2_seqs)
        seqs = self.embed_layer(seqs)
        if self.no_fixed_emb:
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
            # seqs = self.emb_dropout(seqs)
        else:
            seqs = self.position_enc(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs, 
                                            time_mask=timeline_mask,
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, users, log_seqs, time1_seqs, time2_seqs, pos_seqs, neg_seqs, pos_user, neg_user): # for training        
        log_feats = self.log2feats(log_seqs, time1_seqs, time2_seqs)
        pos_embed = log_feats[:, -1, :][pos_user]
        neg_embed = log_feats[:, -1, :][neg_user]

        # obtain popularity-based embeddings for positive and negative item sequences
        pos_embs = self.embed_layer(self.popularity_enc(pos_seqs, time1_seqs, time2_seqs))
        neg_embs = self.embed_layer(self.popularity_enc(neg_seqs, time1_seqs, time2_seqs))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, log_feats[:, -1, :], pos_embed, neg_embed

    def predict(self, log_seqs, time1_seqs, time2_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs, time1_seqs, time2_seqs) 
        final_feat = log_feats[:, -1, :]

        # use most recent interaction time to obtain popularity embedding
        time1_pred = np.tile(time1_seqs[0][-1], item_indices.shape)
        time2_pred = np.tile(time2_seqs[0][-1], item_indices.shape)
        item_embs = self.embed_layer(self.popularity_enc(np.expand_dims(item_indices, axis=1), time1_pred, time2_pred))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def regloss(self, users, pos_users, neg_users, triplet_loss, cos_loss):
        users, pos_users, neg_users = users.to(self.dev), pos_users.to(self.dev), neg_users.to(self.dev)
        loss = 0
        if triplet_loss:
            loss += self.triplet_loss(torch.unsqueeze(users, 1), pos_users, neg_users)
        if cos_loss:
            loss += self.cos_loss(torch.repeat_interleave(users, repeats=10, dim=0), torch.reshape(pos_users, (pos_users.shape[0]*pos_users.shape[1], pos_users.shape[2])), torch.Tensor([1]).to(self.dev)) 
            loss += self.cos_loss(torch.repeat_interleave(users, repeats=10, dim=0), torch.reshape(neg_users, (neg_users.shape[0]*neg_users.shape[1], neg_users.shape[2])), torch.Tensor([-1]).to(self.dev))
        return loss 



class NewB4Rec(torch.nn.Module):
    def __init__(self, itemnum, compare_size, args):
        super(NewB4Rec, self).__init__()	
        assert args.input_units1 % args.base_dim1 == 0	
        assert args.input_units2 % args.base_dim2 == 0	

        self.maxlen = args.maxlen
        self.item_num = itemnum
        self.dev = args.device	
        self.no_fixed_emb = args.no_fixed_emb
        self.compare_size = compare_size

        self.popularity_enc = PopularityEncoding(args) 	
        self.embed_layer = InitFeedForward(args.input_units1 + args.input_units2, args.hidden_units*2, args.hidden_units)	
        if self.no_fixed_emb:	
            self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) 	
        else:	
            self.position_enc = PositionalEncoding(args.hidden_units, args.maxlen)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        if args.triplet_loss:	
            self.triplet_loss = torch.nn.TripletMarginLoss(margin=0.0, p=2)	
        if args.cos_loss:	
            self.cos_loss = torch.nn.CosineEmbeddingLoss() # torch.nn.CosineSimilarity()

        # multi-layers transformer blocks, deep network
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = MultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward2(args.hidden_units, args.hidden_units*4, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.out = torch.nn.Linear(args.hidden_units, args.hidden_units) 

    def GELU(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def log2feats(self, log_seqs, time1_seqs, time2_seqs):
        tensor_seqs = torch.LongTensor(log_seqs)
        mask = (tensor_seqs > 0).unsqueeze(1).repeat(1, tensor_seqs.size(1), 1).unsqueeze(1).to(self.dev)
        seqs = self.popularity_enc(log_seqs, time1_seqs, time2_seqs)
        seqs = self.embed_layer(seqs)
        if self.no_fixed_emb:
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        else:
            seqs = self.position_enc(seqs)
        for i in range(len(self.attention_layers)):
            # seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, mask)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        return self.out(seqs)

    def forward(self, seqs, time1_seqs, time2_seqs, candidates = None):
        final_feat = self.log2feats(seqs, time1_seqs, time2_seqs)  # B x T x V
        final_feat = self.GELU(final_feat)
        if candidates is not None:
            items = candidates 
            t1 = np.repeat(time1_seqs.flatten()[-1], candidates.shape) 
            t2 = np.repeat(time2_seqs.flatten()[-1], candidates.shape)  
            items_, t1_, t2_ = np.expand_dims(items, -1), np.expand_dims(t1, -1), np.expand_dims(t2, -1)  
            item_embs = self.embed_layer(self.popularity_enc(items_, t1_, t2_))
            return item_embs.squeeze(1).matmul(final_feat.squeeze(0).T)[:, -1]
            
        # randomly choose group to rank and obtain loss from, all items is too large, appending actual labels to end of random ones
        items = np.append(np.random.choice(np.arange(1, self.item_num+1), size=(seqs.shape[0], seqs.shape[1], self.compare_size)), np.expand_dims(seqs, axis=-1), axis=2)
        t1 = np.tile(np.expand_dims(time1_seqs, -1), (1, 1, self.compare_size+1))
        t2 = np.tile(np.expand_dims(time2_seqs, -1), (1, 1, self.compare_size+1)) 
        items_, t1_, t2_ = items.reshape((items.shape[0], items.shape[1]*items.shape[2])), t1.reshape((t1.shape[0], t1.shape[1]*t1.shape[2])), t2.reshape((t2.shape[0], t2.shape[1]*t2.shape[2]))
        item_embs = self.embed_layer(self.popularity_enc(items_, t1_, t2_))
        item_embs = item_embs.reshape((item_embs.shape[0], seqs.shape[1], -1, item_embs.shape[-1]))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) 
        logits = self.logsoftmax(logits) 
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V

        return logits

    def predict(self, seqs, time1_seqs, time2_seqs, candidates):
        scores = self.forward(seqs, time1_seqs, time2_seqs, candidates)  # T x V
        return scores



# taken from https://github.com/guoyang9/BPR-pytorch/tree/master
class BPRMF(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(BPRMF, self).__init__()

        self.user_emb = torch.nn.Embedding(user_num+1, args.hidden_units)
        self.item_emb = torch.nn.Embedding(item_num+1, args.hidden_units)
        self.dev = args.device

    def forward(self, user, pos_item, neg_item):
        user = self.user_emb(torch.LongTensor(user).to(self.dev))
        item_i = self.item_emb(torch.LongTensor(pos_item).to(self.dev))
        item_j = self.item_emb(torch.LongTensor(neg_item).to(self.dev))

        prediction_i = item_i.matmul(user.unsqueeze(-1)).squeeze(-1)
        prediction_j = item_j.matmul(user.unsqueeze(-1)).squeeze(-1)
        return prediction_i, prediction_j

    def predict(self, user, item_indices):
        user = self.user_emb(torch.LongTensor(user).to(self.dev))
        items = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = (user * items).sum(dim = -1)
        return logits



# taken from https://github.com/pmixer/SASRec.pytorch/blob/master/model.py
class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) 
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = CausalMultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate,
                                                            self.dev)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs, 
                                            time_mask=timeline_mask,
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits 

    def predict(self, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits



# adapted from https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch/tree/master
class BERT4Rec(torch.nn.Module):
    def __init__(self, itemnum, args):
        super(BERT4Rec, self).__init__()
        self.maxlen = args.maxlen
        self.item_num = itemnum
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) 
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        # multi-layers transformer blocks, deep network
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = MultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward2(args.hidden_units, args.hidden_units*4, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.out = torch.nn.Linear(args.hidden_units, args.hidden_units) # self.item_num+1)
        self.out_bias = torch.nn.Parameter(torch.zeros(self.item_num+1)).cuda()

    def GELU(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def log2feats(self, log_seqs):
        mask = (log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1).to(self.dev)

        # embedding the indexed sequence to sequence of vectors
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        for i in range(len(self.attention_layers)):
            # seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, mask)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        return self.out(seqs)

    def forward(self, seqs):
        final_feat = self.log2feats(seqs)  # B x T x V
        final_feat = self.GELU(final_feat)
        item_embs = self.item_emb(torch.arange(0,self.item_num+1).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) 
        logits = logits + self.out_bias #, causing out of memory issues for large batch size
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        logits = self.logsoftmax(logits) 

        return logits

    def predict(self, seqs, candidates):
        scores = self.forward(seqs)  # T x V
        scores = scores[-1, :]  # V
        candidates = candidates.to(self.dev)
        scores = scores.gather(0, candidates)  # C

        return scores
