import math
import numpy as np
import torch
import pdb
import copy


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
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
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
        return (
            0.5
            * x
            * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
        )

    def forward(self, x):
        return self.w_2(self.dropout(self.GELU(self.w_1(x))))


# taken from https://github.com/RuihongQiu/DuoRec/blob/master/recbole/model/layers.py
class PointWiseFeedForward3(torch.nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(PointWiseFeedForward3, self).__init__()
        self.dense_1 = torch.nn.Linear(hidden_size, inner_size)
        self.dense_2 = torch.nn.Linear(inner_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class InitFeedForward(torch.nn.Module):
    def __init__(self, input_units, hidden1=100, hidden2=50, dropout_rate=0):
        super(InitFeedForward, self).__init__()

        self.fc1 = torch.nn.Linear(input_units, hidden1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden1, hidden2)

    def forward(self, inputs):
        outputs = self.fc2(self.dropout(self.relu(self.fc1(inputs))))
        return outputs


class Gate(torch.nn.Module):
    def __init__(self, hidden=50, dropout_rate=0):
        super(Gate, self).__init__()

        self.fc1 = torch.nn.Linear(hidden, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.sigm = torch.nn.Sigmoid()

    def forward(self, input1, input2):
        gated = self.sigm(self.fc1(input1) + self.fc2(input2))
        return input1 * gated + input2 * (1 - gated)


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
        paddings = torch.ones(attn_weights.shape) * (
            -(2**32) + 1
        )  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights)  # remove empty sequence
        attn_weights = torch.where(attn_mask, paddings, attn_weights)  # enforcing causality

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(
            torch.split(outputs, Q.shape[0], dim=0), dim=2
        )  # div batch_size

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

        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(3)]
        )
        self.output_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, query, mask=None):
        key = query
        value = query
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(query.size(-1))

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
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, : x.size(1)].clone().detach()
        

class ModPositionalEncoding(torch.nn.Module):
    def __init__(self, d_hid, n_position):
        super(ModPositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[0, x.flatten()].clone().detach().reshape(x.shape[0], x.shape[1], -1)
        


class UserActivityEncoding(torch.nn.Module):
    def __init__(self, d_hid, n_position, table_type):
        super(UserActivityEncoding, self).__init__()
        if table_type == "sin":
            self.register_buffer(
                "act_table", self._get_sinusoid_encoding_table(n_position, d_hid)
            )
        elif table_type == "lin":
            self.register_buffer(
                "act_table", self._get_linear_encoding_table(n_position, d_hid)
            )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def _get_linear_encoding_table(self, n_position, d_hid):
        # TODO: do this
        return None 

    def forward(self, x):
        return self.act_table[:, x].clone().detach()



class PopularityEncoding(torch.nn.Module):
    def __init__(self, args, second):
        super(PopularityEncoding, self).__init__()
        n_position = args.maxlen
        d_hid = args.hidden_units
        self.input1 = args.input_units1
        self.input2 = args.input_units2
        self.base_dim1 = args.base_dim1
        self.base_dim2 = args.base_dim2
        # table of fixed feature vectors for items by time, shape: (num_times*base_dim, num_items)
        if not second:
            month_pop = np.loadtxt(f"./data/{args.dataset}_{args.monthpop}.txt")
            week_pop = np.loadtxt(f"./data/{args.dataset}_{args.weekpop}.txt")
        else:
            month_pop = np.loadtxt(f"./data/{args.dataset2}_{args.monthpop}.txt")
            week_pop = np.loadtxt(f"./data/{args.dataset2}_{args.weekpop}.txt")
        # add zeros for the index-0 empty item placeholder and initial time period
        self.register_buffer(
            "month_pop_table",
            torch.cat(
                (
                    torch.zeros((month_pop.shape[0] + self.input1 - self.base_dim1, 1)),
                    torch.cat(
                        (
                            torch.zeros(
                                (self.input1 - self.base_dim1, month_pop.shape[1])
                            ),
                            torch.FloatTensor(month_pop),
                        ),
                        dim=0,
                    ),
                ),
                dim=1,
            ),
        )
        self.register_buffer(
            "week_pop_table",
            torch.cat(
                (
                    torch.zeros((week_pop.shape[0] + self.input2 - self.base_dim2, 1)),
                    torch.cat(
                        (
                            torch.zeros(
                                (self.input2 - self.base_dim2, week_pop.shape[1])
                            ),
                            torch.FloatTensor(week_pop),
                        ),
                        dim=0,
                    ),
                ),
                dim=1,
            ),
        )

    def forward(self, log_seqs, time1_seqs, time2_seqs):
        month_table_rows = torch.flatten(
            torch.flatten(torch.LongTensor(time1_seqs)).reshape((-1, 1))
            * self.base_dim1
            + torch.arange(self.input1)
        )
        month_table_cols = torch.repeat_interleave(
            torch.flatten(torch.LongTensor(log_seqs)), self.input1
        )
        week_table_rows = torch.flatten(
            torch.flatten(torch.LongTensor(time2_seqs)).reshape((-1, 1))
            * self.base_dim2
            + torch.arange(self.input2)
        )
        week_table_cols = torch.repeat_interleave(
            torch.flatten(torch.LongTensor(log_seqs)), self.input2
        )
        if (
                (month_table_rows.numel() > 0 and torch.max(month_table_rows) >= self.month_pop_table.shape[0]) or
                (month_table_cols.numel() > 0 and torch.max(month_table_cols) >= self.month_pop_table.shape[1]) or
                (week_table_rows.numel() > 0 and torch.max(week_table_rows) >= self.week_pop_table.shape[0]) or
                (week_table_cols.numel() > 0 and torch.max(week_table_cols) >= self.week_pop_table.shape[1])
        ):
            # week_table_rows[week_table_rows >= self.week_pop_table.shape[0]]
            pdb.set_trace()
            # raise IndexError("row or column accessed out-of-index in popularity table")
        month_pop = torch.reshape(
            self.month_pop_table[month_table_rows, month_table_cols],
            (log_seqs.shape[0], log_seqs.shape[1], self.input1),
        )
        week_pop = torch.reshape(
            self.week_pop_table[week_table_rows, week_table_cols],
            (log_seqs.shape[0], log_seqs.shape[1], self.input2),
        )
        return torch.cat((month_pop, week_pop), 2).clone().detach()


class EvalPopularityEncoding(torch.nn.Module):
    def __init__(self, args):
        super(EvalPopularityEncoding, self).__init__()
        n_position = args.maxlen
        d_hid = args.hidden_units
        self.input1 = args.input_units1
        self.input2 = args.input_units2
        self.base_dim1 = args.base_dim1
        self.base_dim2 = args.base_dim2
        self.pause = args.pause
        # table of fixed feature vectors for items by time, shape: (num_times*base_dim, num_items)
        month_pop = np.loadtxt(f"./data/{args.dataset}_{args.monthpop}.txt")
        week_pop = np.loadtxt(f"./data/{args.dataset}_{args.weekpop}.txt")
        week_eval_pop = np.loadtxt(f"./data/{args.dataset}_{args.week_eval_pop}.txt")
        self.register_buffer("week_eval_pop", torch.FloatTensor(week_eval_pop))
        # add zeros for the index-0 empty item placeholder and initial time period
        self.register_buffer(
            "month_pop_table",
            torch.cat(
                (
                    torch.zeros((month_pop.shape[0] + self.input1 - self.base_dim1, 1)),
                    torch.cat(
                        (
                            torch.zeros(
                                (self.input1 - self.base_dim1, month_pop.shape[1])
                            ),
                            torch.FloatTensor(month_pop),
                        ),
                        dim=0,
                    ),
                ),
                dim=1,
            ),
        )
        self.register_buffer(
            "week_pop_table",
            torch.cat(
                (
                    torch.zeros((week_pop.shape[0] + self.input2 - self.base_dim2, 1)),
                    torch.cat(
                        (
                            torch.zeros(
                                (self.input2 - self.base_dim2, week_pop.shape[1])
                            ),
                            torch.FloatTensor(week_pop),
                        ),
                        dim=0,
                    ),
                ),
                dim=1,
            ),
        )

    def forward(self, log_seqs, time1_seqs, time2_seqs, user):
        month_table_rows = torch.flatten(
            torch.flatten(torch.LongTensor(time1_seqs)).reshape((-1, 1))
            * self.base_dim1
            + torch.arange(self.input1)
        )
        month_table_cols = torch.repeat_interleave(
            torch.flatten(torch.LongTensor(log_seqs)), self.input1
        )
        if self.input2 > self.base_dim2:
            week_table_rows = torch.flatten(
                torch.flatten(torch.LongTensor(time2_seqs)).reshape((-1, 1))
                * self.base_dim2
                + torch.arange(self.input2 - self.base_dim2)
            )
            week_table_cols = torch.repeat_interleave(
                torch.flatten(torch.LongTensor(log_seqs)), self.input2 - self.base_dim2
            )
            if (torch.max(week_table_rows) >= self.week_pop_table.shape[0] or torch.max(week_table_cols) >= self.week_pop_table.shape[1]):
                raise IndexError("row or column accessed out-of-index in popularity table")

        if (
            torch.max(month_table_rows) >= self.month_pop_table.shape[0]
            or torch.max(month_table_cols) >= self.month_pop_table.shape[1]
        ):
            raise IndexError("row or column accessed out-of-index in popularity table")
        month_pop = torch.reshape(
            self.month_pop_table[month_table_rows, month_table_cols],
            (log_seqs.shape[0], log_seqs.shape[1], self.input1),
        )

        week_eval_rows = torch.flatten((torch.LongTensor(user-1)*self.base_dim2).unsqueeze(1) + torch.arange(self.base_dim2))
        recent_pop = torch.swapaxes(self.week_eval_pop[week_eval_rows].reshape((len(user), 6, -1)), 1, 2)
        if self.input2 > self.base_dim2:
            week_pop = torch.reshape(
                self.week_pop_table[week_table_rows, week_table_cols],
                (log_seqs.shape[0], log_seqs.shape[1], self.input2 - self.base_dim2),
            )
            return torch.cat((month_pop, week_pop, recent_pop), 2).clone().detach()
        else:
            return torch.cat((month_pop, recent_pop), 2).clone().detach()