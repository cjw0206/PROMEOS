# %%
# from harvard_transformer import *
from protein_ppi_encoding_module.harvard_transformer_ffn_moe import *

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class TransformerGO_Scratch(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.2, using_esm2=True, num_experts=4):
        super().__init__()

        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        # ff = Top1SparseMoEFFN(d_model, dim_feedforward, num_experts=num_experts, dropout=dropout)
        self.using_esm2 = using_esm2

        self.cross_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.encoder = Encoder(EncoderLayer_ffn_moe(d_model, c(attn), c(ff), dropout), num_layers)
        self.decoder = Decoder(DecoderLayer_ffn_moe(d_model, c(attn), c(attn), c(ff), dropout), num_layers)

        self.protbert_projector = nn.Linear(1024, d_model)

        self.bert_projector = nn.Sequential(
            nn.Linear(1024, 16 * 64),
            nn.ReLU()
            # nn.GELU(),
            # nn.Dropout(p=0.1)
        )
        self.esm2_projector = nn.Sequential(
            nn.Linear(1280, 20 * 64),
            nn.ReLU()
        )

        self.linear = nn.Linear(d_model, 1)

    # batch  * max_seq_len * node2vec_dim
    def forward(self, emb_proteinA, emb_proteinB, protA_mask, protB_mask, protA_seq, protB_seq):
        ###############################################################################################
        ##################################### ProtBERT Experiment #####################################
        ###############################################################################################
        embA_bert = torch.stack(protA_seq).to(device)
        embB_bert = torch.stack(protB_seq).to(device)

        ################################# concat ##########################################

        ################### B, 1024/64, 64 experiment #################
        B = embA_bert.shape[0]

        if self.using_esm2:
            embA_esm = self.esm2_projector(embA_bert).view(B, 20, 64)
            embB_esm = self.esm2_projector(embB_bert).view(B, 20, 64)

            concat_proteinA = torch.cat([emb_proteinA, embA_esm], dim=1)  # (B, L+20, d_model)
            concat_proteinB = torch.cat([emb_proteinB, embB_esm], dim=1)
            new_maskA = F.pad(protA_mask, (0, 20, 0, 20), value=True)  # shape: (B, L+1)
            new_maskB = F.pad(protB_mask, (0, 20, 0, 20), value=True)
        else:
            embA_bert = self.bert_projector(embA_bert).view(B, 16, 64)
            embB_bert = self.bert_projector(embB_bert).view(B, 16, 64)

            concat_proteinA = torch.cat([emb_proteinA, embA_bert], dim=1)  # (B, L+16, d_model)
            concat_proteinB = torch.cat([emb_proteinB, embB_bert], dim=1)
            new_maskA = F.pad(protA_mask, (0, 16, 0, 16), value=True)  # shape: (B, L+1)
            new_maskB = F.pad(protB_mask, (0, 16, 0, 16), value=True)
        ############### End B, 1024/64, 64 experiment ################
        ################################# concat ##########################################


        ###############################################################################################
        ##################################### ProtBERT Experiment #####################################
        ###############################################################################################

        memory, aux_loss_enc = self.encoder(concat_proteinA, new_maskA)
        output, aux_loss_dec = self.decoder(concat_proteinB, memory, new_maskA, new_maskB)


        total_aux_loss = aux_loss_enc + aux_loss_dec  # 나중에 loss에 포함시키기 위해

        # transform B * seqLen * node2vec_dim --> B * node2vec_dim (TransformerCPI paper)
        output_c = torch.linalg.norm(output, dim=2)
        output_c = F.softmax(output_c, dim=1).unsqueeze(1)
        output = torch.bmm(output_c, output)

        sq_output = self.linear(output).squeeze(1)
        return sq_output.squeeze(1), total_aux_loss

class TransformerGO_matmul(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, using_esm2=True, num_experts=32):
        super().__init__()

        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        # ff = Top1SparseMoEFFN(d_model, dim_feedforward, num_experts=num_experts, dropout=dropout)
        self.using_esm2 = using_esm2

        self.cross_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.encoder = Encoder(EncoderLayer_ffn_moe(d_model, c(attn), c(ff), dropout), num_layers)
        # self.decoder = Decoder(DecoderLayer_ffn_moe(d_model, c(attn), c(attn), c(ff), dropout), num_layers)

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

    def get_encoded_AB(self, A, B, maskA, maskB, protA_seq, protB_seq):
        embA_bert = torch.stack(protA_seq).to(device)
        embB_bert = torch.stack(protB_seq).to(device)
        Bsz = embA_bert.shape[0]

        if self.using_esm2:
            embA_esm = self.esm2_projector(embA_bert).view(Bsz, 20, 64)
            embB_esm = self.esm2_projector(embB_bert).view(Bsz, 20, 64)

            concat_A = torch.cat([A, embA_esm], dim=1)
            concat_B = torch.cat([B, embB_esm], dim=1)

            maskA = F.pad(maskA, (0, 20, 0, 20), value=True)
            maskB = F.pad(maskB, (0, 20, 0, 20), value=True)
        else:
            raise NotImplementedError

        encoded_A, _ = self.encoder(concat_A, maskA)
        encoded_B, _ = self.encoder(concat_B, maskB)
        return concat_A, concat_B, encoded_A, encoded_B

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

        # 각각 encoder에 통과
        encoded_A, aux_loss_A = self.encoder(concat_proteinA, new_maskA)  # (B, T, D)
        encoded_B, aux_loss_B = self.encoder(concat_proteinB, new_maskB)  # (B, T, D)

        # interaction matrix 계산
        interaction = torch.mul(encoded_A, encoded_B)

        # 기존 로직 유지: norm → softmax → weighted sum
        output_c = torch.linalg.norm(interaction, dim=2)  # (B, T)
        output_c = F.softmax(output_c, dim=1).unsqueeze(1)  # (B, 1, T)
        weighted_sum = torch.bmm(output_c, interaction)  # (B, 1, T)

        # 최종 예측
        sq_output = self.linear(weighted_sum).squeeze(1)
        total_aux_loss = aux_loss_A + aux_loss_B
        return sq_output.squeeze(1), total_aux_loss


class TransformerGO_late_fusion(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.2):
        super().__init__()

        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, d_model, dropout)
        ff = Top1SparseMoEFFN(d_model, dim_feedforward, num_experts=8, dropout=dropout)

        # GO stream
        self.encoder_go = Encoder(EncoderLayer_ffn_moe(d_model, c(attn), c(ff), dropout), num_layers)
        self.decoder_go = Decoder(DecoderLayer_ffn_moe(d_model, c(attn), c(attn), c(ff), dropout), num_layers)

        # ESM stream
        self.encoder_esm = Encoder(EncoderLayer_ffn_moe(d_model, c(attn), c(ff), dropout), num_layers)
        self.decoder_esm = Decoder(DecoderLayer_ffn_moe(d_model, c(attn), c(attn), c(ff), dropout), num_layers)

        self.bert_projector = nn.Sequential(
            nn.Linear(1024, 16 * 64),
            nn.ReLU()
        )
        self.esm2_projector = nn.Sequential(
            nn.Linear(1280, 20 * 64),
            nn.ReLU()
        )

        self.linear = nn.Linear(d_model, 1)

    def forward(self, embA_go, embB_go, protA_mask_go, protB_mask_go, protA_seq_esm, protB_seq_esm):
        protA_seq_esm = torch.stack(protA_seq_esm).to(device)
        protB_seq_esm = torch.stack(protB_seq_esm).to(device)

        B = protA_seq_esm.shape[0]

        # ESM projection
        embA_esm = self.esm2_projector(protA_seq_esm).view(B, 20, 64)
        embB_esm = self.esm2_projector(protB_seq_esm).view(B, 20, 64)

        # Prot A -> encoder
        encoded_A_go, auxA_go = self.encoder_go(embA_go, protA_mask_go)
        encoded_A_esm, auxA_esm = self.encoder_esm(embA_esm, None)

        # Prot B -> decoder
        decoded_B_go, auxB_go = self.decoder_go(embB_go, encoded_A_go, protA_mask_go, protB_mask_go)
        decoded_B_esm, auxB_esm = self.decoder_go(embB_esm, encoded_A_esm, None, None)

        fused_decoder = torch.cat([decoded_B_go, decoded_B_esm], dim=1)

        total_aux_loss = auxA_go + auxA_esm + auxB_go + auxB_esm

        # transform B * seqLen * node2vec_dim --> B * node2vec_dim (TransformerCPI paper)
        output_c = torch.linalg.norm(fused_decoder, dim=2)
        output_c = F.softmax(output_c, dim=1).unsqueeze(1)
        output = torch.bmm(output_c, fused_decoder)

        sq_output = self.linear(output).squeeze(1)
        return sq_output.squeeze(1), total_aux_loss


class TransformerGO_ablation_WO_seq(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, num_experts=32):
        super().__init__()

        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, d_model, dropout)
        ff = Top1SparseMoEFFN(d_model, dim_feedforward, num_experts=num_experts, dropout=dropout)

        self.cross_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.encoder = Encoder(EncoderLayer_ffn_moe(d_model, c(attn), c(ff), dropout), num_layers)
        self.decoder = Decoder(DecoderLayer_ffn_moe(d_model, c(attn), c(attn), c(ff), dropout), num_layers)


        self.linear = nn.Linear(d_model, 1)

    # batch  * max_seq_len * node2vec_dim

    def forward(self, emb_proteinA, emb_proteinB, protA_mask, protB_mask, protA_seq, protB_seq):
        
        memory, aux_loss_A = self.encoder(emb_proteinA, protA_mask)
        output, aux_loss_B = self.decoder(emb_proteinB, memory, protA_mask, protB_mask)
        #output: batch * seqLen * embDim
        
        #transform B * seqLen * node2vec_dim --> B * node2vec_dim (TransformerCPI paper)
        output_c = torch.linalg.norm(output, dim = 2)
        output_c = F.softmax(output_c, dim = 1).unsqueeze(1)
        output = torch.bmm(output_c, output)
        total_aux_loss = aux_loss_A + aux_loss_B
        
        return self.linear(output).squeeze(1).squeeze(1), total_aux_loss

class TransformerGO_ablation_WO_GO(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, num_experts=32):
        super().__init__()

        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, d_model, dropout)
        ff = Top1SparseMoEFFN(d_model, dim_feedforward, num_experts=num_experts, dropout=dropout)

        self.cross_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.encoder = Encoder(EncoderLayer_ffn_moe(d_model, c(attn), c(ff), dropout), num_layers)
        self.decoder = Decoder(DecoderLayer_ffn_moe(d_model, c(attn), c(attn), c(ff), dropout), num_layers)

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

        embA_esm = self.esm2_projector(embA_bert).view(B, 20, 64)
        embB_esm = self.esm2_projector(embB_bert).view(B, 20, 64)

        new_maskA = torch.ones(B, 20, 20, dtype=torch.bool, device=device)
        new_maskB = torch.ones(B, 20, 20, dtype=torch.bool, device=device)

        # 각각 encoder에 통과
        encoded_A, aux_loss_A = self.encoder(embA_esm, new_maskA)
        encoded_B, aux_loss_B = self.encoder(embB_esm, new_maskB)

        # interaction 계산
        interaction = torch.mul(encoded_A, encoded_B)

        output_c = torch.linalg.norm(interaction, dim=2)
        output_c = F.softmax(output_c, dim=1).unsqueeze(1)
        weighted_sum = torch.bmm(output_c, interaction)

        sq_output = self.linear(weighted_sum).squeeze(1)
        total_aux_loss = aux_loss_A + aux_loss_B
        return sq_output.squeeze(1), total_aux_loss

