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
from datasets.dataset_manip import *
from training_helper import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available:", device)

# 기본 설정
EMB_DIM = 64
LR = 0.0005
EVAL_INTERVAL = 10

# 실험할 하이퍼파라미터 조합
# experts_list = [4, 8, 16, 32]
experts_list = [32]
dropout_list = [0.1]
weight_decay_list = [5e-5]


dataset_name = "STRING_S"

if dataset_name in ['STRING_S', 'STRING_S_Big', 'STRING_S_600', 'STRING_S_700']:
    BATCH_SIZE = 1024
    N_EPOCHS = 100
    organism = 4932
elif dataset_name in ['STRING_H' ,"STRING_H_Big", 'STRING_H_600', 'STRING_H_700', 'STRING_H_600-800', "STRING_H_400-600"]:
    BATCH_SIZE = 512
    N_EPOCHS = 50
    organism = 9606
elif dataset_name in ['STRING_D']:
    BATCH_SIZE = 512
    N_EPOCHS = 50
    organism = 7227
elif dataset_name in ['STRING_M']:
    BATCH_SIZE = 512
    N_EPOCHS = 50
    organism = 10090
    
data_path = '../datasets/ADSLab_dataset/'

# 공통 경로 설정
go_embed_pth = f"{data_path}go-terms/emb/go-terms-{EMB_DIM}.emd"
go_id_dict_pth = f"{data_path}go-terms/go_id_dict"
protein_go_anno_pth = f"{data_path}stringDB-files/" + (
    "sgd.gaf.gz" if organism == 4932 else
    "fb.gaf.gz" if organism == 7227 else
    "mgi.gaf.gz" if organism == 10090 else
    "goa_human.gaf.gz"
)
alias_path = f"{data_path}stringDB-files/{organism}.protein.aliases.v12.0.txt.gz"
neg_path = f"{data_path}interaction-datasets/{organism}.protein.negative.v12.0.txt"
poz_path = f"{data_path}interaction-datasets/{organism}.protein.links.v12.0.txt"
# neg_path = f"{data_path}interaction-datasets/{organism}.protein.negative.400-600.v12.0.txt"
# poz_path = f"{data_path}interaction-datasets/{organism}.protein.links.400-600.v12.0.txt"
embedding_pkl_path = f"{data_path}interaction-datasets/{organism}_esm2_embeddings.pkl"

# 시퀀스 임베딩 로드
with open(embedding_pkl_path, "rb") as f:
    seq_dict = pickle.load(f)

# 데이터셋 로딩
# train_set, valid_set, test_set, _ = get_dataset_split_stringDB(
#     poz_path, neg_path,
#     protein_go_anno_pth, go_id_dict_pth, go_embed_pth,
#     shuffle, alias_path,
#     ratio=[0.8, 0.1, 0.1],
#     intr_set_size_filter=[0, 5000],
#     seq_dict=seq_dict
# )
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

# 학습 및 평가 함수
def train(model, loader, optimizer, criterion):
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

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_acc, preds, labels = 0, 0, [], []
    with torch.no_grad():
        for batch in loader:
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
    elapsed = e - s
    mins = int(elapsed // 60)
    secs = round(elapsed % 60, 2)  # 초를 소수점 둘째 자리까지
    return mins, secs


result_file = "PROMEOS_Dense_FFN.txt"

total_exp_num = len(experts_list) * len(dropout_list) * len(weight_decay_list)
current_exp_num=1
for experts in experts_list:
    for weight_decay in weight_decay_list:
        for dropout in dropout_list:
            model_id = f"model_{dataset_name}_exp{experts}_wd{weight_decay}_dr{dropout}_time_checking"
            print(f'-------------- Processing: {model_id} --------------({current_exp_num}/{total_exp_num})')
            current_exp_num+=1

            save_dir = os.path.join("saved_models", model_id)
            os.makedirs(save_dir, exist_ok=True)

            model = TransformerGO_matmul(EMB_DIM, 8, 3, 4 * EMB_DIM, dropout,using_esm2=True, num_experts=experts).to(device)
            # model = TransformerGO_Scratch(EMB_DIM, 8, 3, 4*EMB_DIM, 0.1, using_esm2=True, num_experts=experts).to(device)   # W Decoder
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
            criterion = nn.BCEWithLogitsLoss().to(device)

            best_auc = float('-inf')
            best_epoch = -1
            epochs_no_improve = 0

            for epoch in range(1, N_EPOCHS + 1):
                # start = time.time()
                # train_loss, train_acc, train_auc = train(model, train_loader, optimizer, criterion)
                # valid_loss, valid_acc, valid_auc = evaluate(model, valid_loader, criterion)
                # end = time.time()
                # mins, secs = epoch_time(start, end)
                # print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val AUC: {valid_auc:.4f} ({mins}m {secs}s)")

                # ---- Train ----
                start_train = time.time()
                train_loss, train_acc, train_auc = train(model, train_loader, optimizer, criterion)
                end_train = time.time()
                train_mins, train_secs = epoch_time(start_train, end_train)

                # ---- Validation ----
                start_valid = time.time()
                valid_loss, valid_acc, valid_auc = evaluate(model, valid_loader, criterion)
                end_valid = time.time()
                valid_mins, valid_secs = epoch_time(start_valid, end_valid)

                print(f"[Epoch {epoch}] "
                    f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} "
                    f"| Time: {train_mins}m {train_secs}s || "
                    f"Val Loss: {valid_loss:.4f} | Val AUC: {valid_auc:.4f} "
                    f"| Time: {valid_mins}m {valid_secs}s")

                if valid_auc > best_auc:
                    best_auc = valid_auc
                    best_epoch = epoch
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_id}_best.pt"))
                else:
                    epochs_no_improve += 1

                if epoch % EVAL_INTERVAL == 0:
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_id}_epoch{epoch}.pt"))
                    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion)[:3]
                    print(f'{model_id} | Epoch {epoch} test_auc: {test_auc:.4f}')

                if epochs_no_improve >= 10:
                    print(f"Early stopping at epoch {epoch} (best at epoch {best_epoch})")
                    break

            # 최종 평가
            model.load_state_dict(torch.load(os.path.join(save_dir, f"{model_id}_best.pt")))
            test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion)[:3]
            print(f"[{model_id}] Test AUROC: {test_auc:.4f}")

            # 결과 저장 (append)
            write_header = not os.path.exists(result_file)
            with open(result_file, "a") as f:
                if write_header:
                    f.write("model_id\tdataset\tbatch_size\tLR\tweight_decay\tdropout\texperts\tbest_epoch\tbest_valid_AUC\ttest_AUC\n")
                f.write(f"{model_id}\t{dataset_name}\t{BATCH_SIZE}\t{LR}\t{weight_decay}\t{dropout}\t{experts}\t{best_epoch}\t{best_auc:.4f}\t{test_auc:.4f}\n")


            # ★ 실험 하나 끝날 때마다 파일 정렬 후 다시 저장 (test_auc 기준)
            with open(result_file, "r") as f:
                lines = f.readlines()
            header = lines[0]
            records = lines[1:]
            records_sorted = sorted(records, key=lambda x: float(x.strip().split('\t')[-1]), reverse=True)
            with open(result_file, "w") as f:
                f.write(header)
                f.writelines(records_sorted)

