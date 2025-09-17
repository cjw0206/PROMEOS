import os
import sys
import time
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm as prog_bar
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from datetime import datetime
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)

# 시작 시간 출력
print("Program started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from protein_ppi_encoding_module.transformerGO_ffn_moe import *
# from protein_ppi_encoding_module.transformerGO import *
from datasets.dataset_manip import *
from training_helper import *

# ------------------- 설정 -------------------
c_organism='S'

if c_organism == 'S':
    organism_file='S.cerevisiae'
    organism=4932
    BATCH_SIZE = 1024

elif c_organism =='H':
    organism_file='H.sapiens'
    organism = 9606
    BATCH_SIZE = 512
model_name = "model_STRING_S_exp32_wd5e-05_dr0.1_Dense_FFN_best.pt"
model_path = f"model_results/Ablation/{organism_file}/{model_name}"
# model_path = f"model_results/Comparative_Study/STRING_small/{organism_file}/{model_name}"
# model_path = f"model_results/Experts_num/{organism_file}/{model_name}"
data_path = "../datasets/ADSLab_dataset/"
# data_path = "../datasets/BioGrid_dataset/"
DROPOUT = 0.1
EXPERTS = 32


EMB_DIM = 64

go_embed_pth = f"{data_path}go-terms/emb/go-terms-{EMB_DIM}.emd"
go_id_dict_pth = f"{data_path}go-terms/go_id_dict"
# ------------------- STRING -------------------
protein_go_anno_pth = f"{data_path}stringDB-files/{'sgd.gaf.gz' if organism == 4932 else 'goa_human.gaf.gz'}"
alias_path = f"{data_path}stringDB-files/{organism}.protein.aliases.v12.0.txt.gz"
neg_path = f"{data_path}interaction-datasets/{organism}.protein.negative.v12.0.txt"
poz_path = f"{data_path}interaction-datasets/{organism}.protein.links.v12.0.txt"
# neg_path = f"{data_path}interaction-datasets/{organism}.protein.negative.v12.0_large.txt"
# poz_path = f"{data_path}interaction-datasets/{organism}.protein.links.v12.0_large.txt"
# neg_path = f"{data_path}interaction-datasets/{organism}.protein.negative.v12.0_600.txt"
# poz_path = f"{data_path}interaction-datasets/{organism}.protein.links.v12.0_600.txt"
embedding_pkl_path = f"{data_path}interaction-datasets/{organism}_esm2_embeddings.pkl"
# ------------------- STRING -------------------

# ------------------- BioGRID ------------------
# protein_go_anno_pth = f"{data_path}BioGrid-files/{'sgd.gaf.gz' if organism == 4932 else 'goa_human.gaf.gz'}"
# alias_path = f"{data_path}BioGrid-files/{organism}.protein.aliases.v12.0.txt.gz"
# neg_path = f"{data_path}interaction-datasets/{organism}_biogrid_filtered_negative.txt"
# poz_path = f"{data_path}interaction-datasets/{organism}_biogrid_filtered_positive.txt"
# # neg_path = f"{data_path}interaction-datasets/{organism}_biogrid_low_high_negative.txt"
# # poz_path = f"{data_path}interaction-datasets/{organism}_biogrid_low_high_positive.txt"
# embedding_pkl_path = f"{data_path}interaction-datasets/{organism}_esm2_embeddings.pkl"
# ------------------- BioGRID ------------------


# ------------------- 데이터 로딩 -------------------
with open(embedding_pkl_path, "rb") as f:
    seq_dict = pickle.load(f)

def helper_collate(batch):
    MAX_LEN_SEQ = get_max_len_seq(batch)
    return transformerGO_collate_fn(batch, MAX_LEN_SEQ, EMB_DIM, pytorch_pad=False)

_, _, test_set, _ = get_dataset_split_stringDB_keep_ratio(
    poz_path, neg_path,
    protein_go_anno_pth, go_id_dict_pth, go_embed_pth,
    shuffle, alias_path,
    ratio=[0.8, 0.1, 0.1],
    intr_set_size_filter=[0, 5000],
    seq_dict=seq_dict
)

test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=helper_collate, shuffle=False)

# ------------------- 모델 로딩 -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerGO_matmul(EMB_DIM, 8, 3, 4*EMB_DIM, 0.1, using_esm2=True, num_experts=EXPERTS).to(device)      # using_esm2=False: ProtBERT
# model = TransformerGO_ablation_WO_seq(EMB_DIM, 8, 3, 4*EMB_DIM, 0.1, num_experts=EXPERTS).to(device)            # W/O Seq
# model = TransformerGO_ablation_WO_GO(EMB_DIM, 8, 3, 4*EMB_DIM, 0.1, num_experts=EXPERTS).to(device)             # W/O GO
# model = TransformerGO_Scratch(EMB_DIM, 8, 3, 4*EMB_DIM, 0.1, using_esm2=True, num_experts=EXPERTS).to(device)   # W Decoder
# model = TransformerGO_Scratch(EMB_DIM, 8, 3, 4*EMB_DIM, 0.2, using_esm2=True).to(device)                        # W/O MoE
# model = TransformerGO_Original(EMB_DIM, 8, 3, 4*EMB_DIM, 0.2).to(device)                                      # Original
# model =  GO_Sum_NN(EMB_DIM, 256, 512, 256, 1, 0.2).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ------------------- 평가 함수 -------------------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, labels = [], []

    with torch.no_grad():
        for batch in prog_bar(loader):
            padded_pairs, y, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            protA_seq, protB_seq = batch[4], batch[5]
            A, B = padded_pairs[:, 0], padded_pairs[:, 1]

            pred, aux_loss = model(A, B, mask[:, 0], mask[:, 1], protA_seq, protB_seq)
            # pred = model(A, B, mask[:, 0], mask[:, 1], protA_seq, protB_seq).squeeze(1)
            loss = criterion(pred, y)
            total_loss += loss.item()

            preds += list(pred.detach().cpu().numpy())
            labels += list(y.detach().cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    preds_bin = (preds >= 0.5).astype(int)

    auc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    f1 = f1_score(labels, preds_bin)
    acc = accuracy_score(labels, preds_bin)

    return total_loss / len(loader), auc, auprc, f1, acc

# ------------------- 평가 실행 -------------------
criterion = nn.BCEWithLogitsLoss().to(device)
test_loss, test_auc, test_auprc, test_f1, test_acc = evaluate(model, test_loader, criterion)

# ------------------- 출력 -------------------
print(f'Model: {model_name}')
print("AUROC\tAUPRC\tF1\tACC")
print(f"{test_auc:.4f}\t{test_auprc:.4f}\t{test_f1:.4f}\t{test_acc:.4f}")