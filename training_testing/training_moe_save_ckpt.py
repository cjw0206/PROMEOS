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
from sklearn.metrics import roc_auc_score
from datetime import datetime

# Custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from protein_ppi_encoding_module.transformerGO_ffn_moe import *
from datasets.dataset_manip import *
from training_helper import *

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available:", device)
print("Program started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Config
EMB_DIM = 64
BATCH_SIZE = 1024
LR = 0.0005
WEIGHT_DECAY = 5e-5
N_EPOCHS = 100
EVAL_INTERVAL = 10
model_id = "model_S_time"
print(f'model id: {model_id}')
data_path = '../datasets/ADSLab_dataset/'

# Paths
dataset_name = "STRING_S"  # Choose STRING_S or STRING_H

organism = 4932 if dataset_name == "STRING_S" else 9606
data_path = '../datasets/ADSLab_dataset/'

# 공통 경로 설정
go_embed_pth = f"{data_path}go-terms/emb/go-terms-{EMB_DIM}.emd"
go_id_dict_pth = f"{data_path}go-terms/go_id_dict"
protein_go_anno_pth = f"{data_path}stringDB-files/{'sgd.gaf.gz' if organism == 4932 else 'goa_human.gaf.gz'}"
alias_path = f"{data_path}stringDB-files/{organism}.protein.aliases.v12.0.txt.gz"
neg_path = f"{data_path}interaction-datasets/{organism}.protein.negative.v12.0.txt"
poz_path = f"{data_path}interaction-datasets/{organism}.protein.links.v12.0.txt"
embedding_pkl_path = f"{data_path}interaction-datasets/{organism}_esm2_embeddings.pkl"



# Load sequence embedding
with open(embedding_pkl_path, "rb") as f:
    seq_dict = pickle.load(f)

# Dataset
train_set, valid_set, test_set, _ = get_dataset_split_stringDB_keep_ratio(
    poz_path, neg_path,
    protein_go_anno_pth, go_id_dict_pth, go_embed_pth,
    shuffle, alias_path,
    ratio=[0.8, 0.1, 0.1],
    intr_set_size_filter=[0, 5000],
    seq_dict=seq_dict
)

def helper_collate(batch):
    MAX_LEN_SEQ = get_max_len_seq(batch)
    return transformerGO_collate_fn(batch, MAX_LEN_SEQ, EMB_DIM, pytorch_pad=False)

params = {'batch_size': BATCH_SIZE, 'collate_fn': helper_collate}
train_loader = data.DataLoader(train_set, **params, shuffle=True)
valid_loader = data.DataLoader(valid_set, **params, shuffle=True)
test_loader = data.DataLoader(test_set, **params, shuffle=False)

# Model
model = TransformerGO_matmul(EMB_DIM, 8, 3, 4*EMB_DIM, 0.2, using_esm2=True, num_experts=32).to(device)
# model = TransformerGO_ablation_WO_seq(EMB_DIM, 8, 3, 4*EMB_DIM, 0.2, num_experts=32).to(device)
# model = TransformerGO_ablation_WO_GO(EMB_DIM, 8, 3, 4*EMB_DIM, 0.2, num_experts=32).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss().to(device)

# 전체 파라미터와 학습 파라미터 개수 출력
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

import time

print("잠시 멈춥니다...")
time.sleep(10)  # 10초 대기
print("다시 시작합니다!")


# Train & Eval
def train(model, loader):
    model.train()
    total_loss, total_acc, preds, labels = 0, 0, [], []
    for batch in prog_bar(loader):
        optimizer.zero_grad()
        padded_pairs, y, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        protA_seq, protB_seq = batch[4], batch[5]
        A, B = padded_pairs[:, 0], padded_pairs[:, 1]
        pred, aux_loss = model(A, B, mask[:, 0], mask[:, 1], protA_seq, protB_seq)
        loss = criterion(pred, y) + 0.01 * aux_loss
        acc = binary_accuracy(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
        preds += list(pred.detach().cpu().numpy())
        labels += list(y.detach().cpu().numpy())
    return total_loss / len(loader), total_acc / len(loader), roc_auc_score(labels, preds)

def evaluate(model, loader):
    model.eval()
    total_loss, total_acc, preds, labels = 0, 0, [], []
    with torch.no_grad():
        for batch in prog_bar(loader):
            padded_pairs, y, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            protA_seq, protB_seq = batch[4], batch[5]
            A, B = padded_pairs[:, 0], padded_pairs[:, 1]
            pred, aux_loss = model(A, B, mask[:, 0], mask[:, 1], protA_seq, protB_seq)
            loss = criterion(pred, y)
            acc = binary_accuracy(pred, y)
            total_loss += loss.item()
            total_acc += acc.item()
            preds += list(pred.detach().cpu().numpy())
            labels += list(y.detach().cpu().numpy())
    return total_loss / len(loader), total_acc / len(loader), roc_auc_score(labels, preds)

def epoch_time(s, e):
    return int((e - s) // 60), int((e - s) % 60)

# Training loop
save_dir = os.path.join("saved_models", model_id)
os.makedirs(save_dir, exist_ok=True)
best_auc = float('-inf')
best_epoch = -1
epochs_no_improve = 0

for epoch in range(1, N_EPOCHS + 1):
    start = time.time()
    train_loss, train_acc, train_auc = train(model, train_loader)
    valid_loss, valid_acc, valid_auc = evaluate(model, valid_loader)
    end = time.time()
    mins, secs = epoch_time(start, end)
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val AUC: {valid_auc:.4f} ({mins}m {secs}s)")

    if valid_auc > best_auc:
        best_auc = valid_auc
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_id}_best.pt"))
    else:
        epochs_no_improve += 1

    if epoch % EVAL_INTERVAL == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_id}_epoch{epoch}.pt"))
        test_loss, test_acc, test_auc = evaluate(model, test_loader)[:3]
        print(f'{model_id} | Epoch {epoch} test_auc: {test_auc:.4f}')

    if epochs_no_improve >= 10:
        print(f"Early stopping at epoch {epoch} (best at epoch {best_epoch})")
        break

# Test
model.load_state_dict(torch.load(os.path.join(save_dir, f"{model_id}_best.pt")))
test_loss, test_acc, test_auc = evaluate(model, test_loader)[:3]

# Log
result_file = "model_results.txt"
write_header = not os.path.exists(result_file)
with open(result_file, "a") as f:
    if write_header:
        f.write("model_id\torganism\tbatch_size\tLR\tweight_decay\tbest_valid_AUC\ttest_AUC\n")
    f.write(f"{model_id}\t{organism}\t{BATCH_SIZE}\t{LR}\t{WEIGHT_DECAY}\t{best_auc:.4f}\t{test_auc:.4f}\n")

print(f"[Test] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUROC: {test_auc:.4f}")
