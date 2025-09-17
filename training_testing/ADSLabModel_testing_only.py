import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tqdm import tqdm as prog_bar

# 시작 시간
print("Test-only mode started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 환경 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from protein_ppi_encoding_module.transformerGO_ffn_moe import *
from datasets.dataset_manip import *
from training_helper import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available:", device)

# 하이퍼파라미터 및 경로 설정
model_id = "model_STRING_S_exp32_wd0.0005_dr0.3"   # <-- 여기에 불러올 모델 ID를 입력
save_dir = os.path.join("saved_models", model_id)
model_path = os.path.join(save_dir, f"{model_id}_best.pt")

dataset_name = "STRING_S"  # "STRING_S" 또는 "STRING_H"
organism = 4932 if dataset_name == "STRING_S" else 9606
EMB_DIM = 64
BATCH_SIZE = 1024

data_path = '../datasets/ADSLab_dataset/'
go_embed_pth = f"{data_path}go-terms/emb/go-terms-{EMB_DIM}.emd"
go_id_dict_pth = f"{data_path}go-terms/go_id_dict"
protein_go_anno_pth = f"{data_path}stringDB-files/{'sgd.gaf.gz' if organism == 4932 else 'goa_human.gaf.gz'}"
alias_path = f"{data_path}stringDB-files/{organism}.protein.aliases.v12.0.txt.gz"
neg_path = f"{data_path}interaction-datasets/{organism}.protein.negative.v12.0.txt"
poz_path = f"{data_path}interaction-datasets/{organism}.protein.links.v12.0.txt"
embedding_pkl_path = f"{data_path}interaction-datasets/{organism}_esm2_embeddings.pkl"

# 데이터셋 로딩
with open(embedding_pkl_path, "rb") as f:
    seq_dict = pickle.load(f)

train_set, valid_set, test_set, _ = get_dataset_split_stringDB(
    poz_path, neg_path,
    protein_go_anno_pth, go_id_dict_pth, go_embed_pth,
    shuffle, alias_path,
    ratio=[0.64, 0.16, 0.2],
    intr_set_size_filter=[0, 5000],
    seq_dict=seq_dict
)

def helper_collate(batch):
    MAX_LEN_SEQ = get_max_len_seq(batch)
    return transformerGO_collate_fn(batch, MAX_LEN_SEQ, EMB_DIM, pytorch_pad=False)

params = {'batch_size': BATCH_SIZE, 'collate_fn': helper_collate}
test_loader = data.DataLoader(test_set, **params, shuffle=False)

# 평가 함수
def evaluate(model, loader, criterion):
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

# 모델 초기화 및 로드
# 필요한 하이퍼파라미터는 model_id에서 추출하거나 수동 지정
dropout = 0.3
experts = 32

model = TransformerGO_matmul(EMB_DIM, 8, 3, 4 * EMB_DIM, dropout, using_esm2=True, num_experts=experts).to(device)
model.load_state_dict(torch.load(model_path))
criterion = nn.BCEWithLogitsLoss().to(device)

# 평가
test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion)[:3]
print(f"[TEST ONLY] {model_id} → Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUROC: {test_auc:.4f}")
