# # %%
# @inproceedings{opennmt,
#   author    = {Guillaume Klein and
#                Yoon Kim and
#                Yuntian Deng and
#                Jean Senellart and
#                Alexander M. Rush},
#   title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
#   booktitle = {Proc. ACL},
#   year      = {2017},
#   url       = {https://doi.org/10.18653/v1/P17-4012},
#   doi       = {10.18653/v1/P17-4012}
# }

# %%
import torch
import torch.nn as nn
import math, copy, time
import torch.nn.functional as F


# %%
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        sublayer_out = sublayer(self.norm(x))
        return x + self.dropout(sublayer(self.norm(x)))


# %%
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        total_aux_loss = 0.0

        # 첫 입력이 (Tensor, aux_loss) 튜플인 경우 방지 (예외 상황)
        if isinstance(x, (tuple, list)):
            x = x[0]

        for layer in self.layers:
            x, aux_loss = layer(x, mask)  # 항상 tuple 반환 가정
            total_aux_loss += aux_loss

        x = self.norm(x)  # x는 반드시 tensor
        return x, total_aux_loss


class EncoderLayer_ffn_moe(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer_ffn_moe, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        if isinstance(self.feed_forward, Top1SparseMoEFFN):
            ff_out, aux_loss = self.feed_forward(x)
            x = self.sublayer[1](x, lambda _: ff_out)
            # optional: save aux_loss somewhere if needed
            return [x, aux_loss]
        else:
            return [self.sublayer[1](x, self.feed_forward), 0.0]


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        total_aux_loss = 0.0

        # 혹시 x가 [tensor, aux_loss] 형태일 수 있으므로 방지
        if isinstance(x, (list, tuple)):
            x = x[0]

        for layer in self.layers:
            x, aux_loss = layer(x, memory, src_mask, tgt_mask)
            total_aux_loss += aux_loss

        x = self.norm(x)
        return x, total_aux_loss



class DecoderLayer_ffn_moe(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer_ffn_moe, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        # Top-1 Sparse MoE FFN 분기
        if isinstance(self.feed_forward, Top1SparseMoEFFN):
            ff_out, aux_loss = self.feed_forward(x)
            x = self.sublayer[2](x, lambda _: ff_out)
            return [x, aux_loss]
        else:
            x = self.sublayer[2](x, self.feed_forward)
            return [x, 0.0]


# %%
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # Zip only goes through the first 3 Layers - Ioan
        # Each matrix multiplications is done once and then split in heads
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


# %%
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Top1SparseMoEFFN(nn.Module):
    """
    Switch Transformer 스타일 Top-1 Sparse MoE FeedForward Layer
    """

    def __init__(self, d_model, d_ff, num_experts=4, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.size()
        x_flat = x.view(-1, D)  # (B*T, D)
        gate_logits = self.gate(x_flat)  # (B*T, E)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top1_probs, top1_idx = torch.max(gate_probs, dim=-1)

        output = torch.zeros_like(x_flat)
        expert_counter = torch.zeros(self.num_experts, device=x.device)

        for expert_id in range(self.num_experts):
            mask = (top1_idx == expert_id)
            if mask.sum() == 0:  # didn't select any expert, skip the current batch
                continue
            input_i = x_flat[mask]
            out_i = self.experts[expert_id](input_i)
            weighted_out = top1_probs[mask].unsqueeze(1) * out_i
            output[mask] = weighted_out
            expert_counter[expert_id] += mask.sum()

        output = output.view(B, T, D)
        output = self.dropout(output)

        # load balancing auxiliary loss
        avg_expert_usage = expert_counter / (B * T)
        aux_loss = (avg_expert_usage ** 2).sum() * self.num_experts

        return output, aux_loss