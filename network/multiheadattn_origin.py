import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q = q.permute(1,0,2)
        v = v.permute(1,0,2)
        k = k.permute(1,0,2)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, self.n_head, length, d_tensor)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length1, length2, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.view(batch_size, length1, length2, d_model)
        return tensor

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()
        self.w_score = nn.Linear(50, 1100) # 50, 1100

    def threshold(self, score, num):

        # score (N, heads, len_q, len_v)
        result_list = []
        batch_size, head, length, d_tensor = score.size()
        new_score = score.reshape(-1, d_tensor)
        for item in new_score:
            index = item.sort()[1][-num:]
            # item[:] = 0
            # item[index] = 1
            result_list.append(index.unsqueeze(0))

        # result = new_score.reshape(batch_size, head, length, d_tensor)
        result = torch.cat(result_list)
        result = result.reshape(batch_size, head, length, num)

        return result

    def extract(self, score, v):

        qbatch_size, qhead, qlength, qd_tensor = score.size()
        vbatch_size, vhead, vlength, vd_tensor = v.size()
        score_t = score.reshape(-1, qlength, qd_tensor)
        v_t = v.reshape(-1, vlength, vd_tensor)

        result_list = []
        for v_item, s_item  in zip(v_t, score_t):

            extract_value = [v_item[index_item].unsqueeze(0) for index_item in s_item]

            result_list.append(torch.cat(extract_value).unsqueeze(0))

        result = torch.cat(result_list)
        result = result.reshape(vbatch_size, vhead, result.shape[1], result.shape[2], result.shape[3])
        return result

    def forward(self, q, k, v, mask=None, e=1e-12, num=100):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.view(batch_size, head, d_tensor, length)  # transpose
        # q_t = q.view(batch_size, head, d_tensor, length)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        # score = (q_t @ k) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        result = self.threshold(score, num)

        # 4. multiply with Value
        v  = self.extract(result, v)
        #v = score @ v

        return v, score


if __name__ == "__main__":

    mul_attn = MultiHeadAttention(512, 8)

    q = torch.ones((3, 50, 512))
    k = torch.ones((3, 1100, 512))
    v = torch.ones((3, 1100, 512))

    out = mul_attn(q,k,v)
    print(out[0].shape)
