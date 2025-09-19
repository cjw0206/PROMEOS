from Models.harvard_transformer_ffn_moe import *

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PROMEOS(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, using_esm2=True, num_experts=32):
        super().__init__()

        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, d_model, dropout)
        ff = Top1SparseMoEFFN(d_model, dim_feedforward, num_experts=num_experts, dropout=dropout)
        self.using_esm2 = using_esm2

        self.cross_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.encoder = Encoder(EncoderLayer_ffn_moe(d_model, c(attn), c(ff), dropout), num_layers)
        # self.decoder = Decoder(DecoderLayer_ffn_moe(d_model, c(attn), c(attn), c(ff), dropout), num_layers)       # if you need decoder, then use this code.

        self.protbert_projector = nn.Linear(1024, d_model)

        self.bert_projector = nn.Sequential(
            nn.Linear(1024, 16 * 64),
            nn.ReLU()
        )
        self.esm2_projector = nn.Sequential(
            nn.Linear(1280, 20 * 64),
            nn.ReLU()
        )

        self.linear = nn.Linear(d_model, 1)

    # batch  * max_seq_len * node2vec_dim
    def forward(self, emb_proteinA, emb_proteinB, protA_mask, protB_mask, protA_seq, protB_seq):
        embA_bert = torch.stack(protA_seq).to(device)
        embB_bert = torch.stack(protB_seq).to(device)
        B = embA_bert.shape[0]

        if self.using_esm2:
            embA_esm = self.esm2_projector(embA_bert).view(B, 20, 64)
            embB_esm = self.esm2_projector(embB_bert).view(B, 20, 64)

            concat_proteinA = torch.cat([emb_proteinA, embA_esm], dim=1)  # (B, LA, D)
            concat_proteinB = torch.cat([emb_proteinB, embB_esm], dim=1)  # (B, LB, D)
            new_maskA = F.pad(protA_mask, (0, 20, 0, 20), value=True)
            new_maskB = F.pad(protB_mask, (0, 20, 0, 20), value=True)

        else:
            embA_bert = self.bert_projector(embA_bert).view(B, 16, 64)
            embB_bert = self.bert_projector(embB_bert).view(B, 16, 64)

            concat_proteinA = torch.cat([emb_proteinA, embA_bert], dim=1)
            concat_proteinB = torch.cat([emb_proteinB, embB_bert], dim=1)
            new_maskA = F.pad(protA_mask, (0, 16, 0, 16), value=True)
            new_maskB = F.pad(protB_mask, (0, 16, 0, 16), value=True)


        encoded_A, aux_loss_A = self.encoder(concat_proteinA, new_maskA)  # (B, T, D)
        encoded_B, aux_loss_B = self.encoder(concat_proteinB, new_maskB)  # (B, T, D)


        interaction = torch.mul(encoded_A, encoded_B)

        output_c = torch.linalg.norm(interaction, dim=2)  # (B, T)
        output_c = F.softmax(output_c, dim=1).unsqueeze(1)  # (B, 1, T)
        weighted_sum = torch.bmm(output_c, interaction)  # (B, 1, T)

        sq_output = self.linear(weighted_sum).squeeze(1)
        total_aux_loss = aux_loss_A + aux_loss_B
        return sq_output.squeeze(1), total_aux_loss