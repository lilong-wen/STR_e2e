import math
import torch
import torch.nn as nn

'''
query: (N, query_num, query_embed_len, hidden_dim)
key: (N, key_num, query_embed_len, hidden_dim)
value: (N, value_embed_len, hidden_dim)

query_embed_len * heads_len == value_embed_len


eg: q:(1, 100, 200, 80)
    v:(1, 1100, 80)
    v --> (1, 80, 1100) --> 100 * linear
      --> v' = 100 * (1, 80, 200)
      --> (1, 100, 200, 80)

out: (1, 100, 200, 80)
'''

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_qk_2(X):
    """Transposition for parallel computation of multiple attention heads."""
    # out: (N, num_query, query_embed, hidden_dim)

    # N*num_heads, query_num, query_embed, num_hiddens/num_heads
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_v(X):
    """Transposition for parallel computation of multiple attention heads."""
    # (N, heads_len, value_len, hidden_dim)
    # N*num_heads, query_num, query_embed, num_hiddens/num_heads
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # X = X.permute(0, 2, 1, 3)
    # return X.reshape(X.shape[0], X.shape[1], -1)
    return X

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiLenAttention(nn.Module):

    def __init__(self, num_heads, query_len, value_len):
        super(MultiLenAttention ,self).__init__()

        self.num_heads = num_heads
        self.linear = nn.Linear(value_len, query_len)

    def forward(self, values):
        # N, value_len, hidden_dim
        # to (N, hidden_dim, value_len)
        values = values.permute(0, 2, 1)
        snip_list = []
        for i in range(self.num_heads):

            snip = self.linear(values) # (N, hidden_dim, query_len)
            snip_list.append(snip)

        result = torch.stack(snip_list, dim=1) # (N, num_heads, hidden_dim, query_len)
        result = result.permute(0, 1, 3, 2)

        return result

class MultiHeadAttention_for_len(nn.Module):
    """Multi-head attention."""
    def __init__(self,
                 num_hiddens,
                 num_heads,
                 query_len,
                 value_len,
                 dropout,
                 bias=False, **kwargs):
        super(MultiHeadAttention_for_len, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.value_att = MultiLenAttention(num_heads, query_len, value_len)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)


        queries = transpose_qk_2(self.W_q(queries))
        keys = self.value_att(keys)
        keys = transpose_v(self.W_k(keys))
        # keys = transpose_qk_2(self.W_k(keys))
        values = self.value_att(values)
        values = transpose_v(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)

        # print(queries.shape)
        # print(keys.shape)
        # print(values.shape)

        output = self.attention(queries, keys, values, valid_lens)
        # print(f"output.shape{output.shape}")

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

if __name__ == "__main__":

    # key_size, query_size, value_size, num_hiddens,
    # num_heads, dropout
    num_hiddens = 80
    num_heads = 100
    query_len = 200
    value_len = 1100
    # key_size, query_size, value_size, num_hiddens,
    # num_heads, query_len, value_len, dropout
    attention =  MultiHeadAttention_for_len(num_hiddens, num_heads, query_len, value_len, 0.5)
    batch_size = 1
    num_queries = num_heads

    X = torch.ones((batch_size, num_queries, query_len, num_hiddens))
    Y = torch.ones((batch_size, value_len, num_hiddens))

    print(attention(X, Y, Y).shape)
    # attn = MultiHeadAttention(key_size, query_size, value_size,
    #                           num_hiddens, num_heads, dropout)
