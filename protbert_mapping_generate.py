# import gzip
# import torch
# import pickle
# from transformers import BertModel, BertTokenizer
# from Bio import SeqIO
# from tqdm import tqdm
#
# # 경로 설정
# fasta_path = "/home/25p_01/JW/ADSModel_STRING/datasets/ADSLab_dataset/interaction-datasets/4932.protein.sequences.v12.0.fa.gz"
# save_path = "/home/25p_01/JW/ADSModel_STRING/datasets/ADSLab_dataset/interaction-datasets/4932_protbert_embeddings.pkl"
#
# # ProtBert 로드
# tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
# model = BertModel.from_pretrained("Rostlab/prot_bert")
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# # 입력 전처리 함수 (공백 삽입)
# def preprocess_sequence(sequence):
#     return ' '.join(list(sequence))
#
# # 임베딩 저장 딕셔너리
# embedding_dict = {}
#
# # 압축된 FASTA 파일 열기
# with gzip.open(fasta_path, "rt") as handle:
#     records = list(SeqIO.parse(handle, "fasta"))
#
# print(f"총 {len(records)}개의 단백질 시퀀스를 처리합니다...")
#
# for record in tqdm(records):
#     protein_id = record.id.split('|')[1] if '|' in record.id else record.id
#     seq = preprocess_sequence(str(record.seq))
#
#     # 토크나이징 및 텐서 변환
#     tokens = tokenizer(seq, return_tensors="pt", padding=True)
#     input_ids = tokens["input_ids"].to(device)
#     attention_mask = tokens["attention_mask"].to(device)
#
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         # CLS 토큰 임베딩 사용
#         embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
#
#     embedding_dict[protein_id] = embedding
#
# # 딕셔너리 저장 (Pickle)
# with open(save_path, "wb") as f:
#     pickle.dump(embedding_dict, f)
#
# print(f"ProtBert 임베딩이 {save_path}에 저장되었습니다.")



#################################################### token-level ######################################################
# import gzip
# import torch
# import pickle
# from transformers import BertModel, BertTokenizer
# from Bio import SeqIO
# from tqdm import tqdm
#
# # 경로 설정
# fasta_path = "/home/25p_01/JW/TransformerGO_py_version/datasets/onto2vec-datasets-string/data/9606.protein.sequences.v11.0.fa.gz"
# save_path = "/home/25p_01/JW/TransformerGO_py_version/datasets/onto2vec-datasets-string/data/9606_protbert_token_embeddings.pkl"
#
# # ProtBert 로드
# tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
# model = BertModel.from_pretrained("Rostlab/prot_bert")
# model.eval()
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# # 입력 전처리 함수 (공백 삽입)
# def preprocess_sequence(sequence):
#     return ' '.join(list(sequence))
#
# # 임베딩 저장 딕셔너리
# embedding_dict = {}
#
# # 압축된 FASTA 파일 열기
# with gzip.open(fasta_path, "rt") as handle:
#     records = list(SeqIO.parse(handle, "fasta"))
#
# print(f"총 {len(records)}개의 단백질 시퀀스를 처리합니다...")
#
# for record in tqdm(records):
#     protein_id = record.id.split('|')[1] if '|' in record.id else record.id
#     seq = preprocess_sequence(str(record.seq))
#
#     # 토크나이징 및 텐서 변환
#     tokens = tokenizer(seq, return_tensors="pt", padding=True)
#     input_ids = tokens["input_ids"].to(device)
#     attention_mask = tokens["attention_mask"].to(device)
#
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         # token-level 임베딩 사용 (CLS 제외)
#         embedding = outputs.last_hidden_state[:, 1:-1, :].squeeze(0).cpu()  # shape: (T-2, 1024)
#
#     embedding_dict[protein_id] = embedding  # 각 단백질 ID에 대해 (T-2, 1024)
#
# # 딕셔너리 저장 (Pickle)
# with open(save_path, "wb") as f:
#     pickle.dump(embedding_dict, f)
#
# print(f"ProtBert token-level 임베딩이 {save_path}에 저장되었습니다.")



########################################### ESM Embedding ##################################################
import gzip
import torch
import pickle
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO
from tqdm import tqdm

# 경로 설정
# fasta_path = "/home/25p_01/JW/TransformerGO_py_version/datasets/onto2vec-datasets-string/data/9606.protein.sequences.v11.0.fa.gz"
# save_path = "/home/25p_01/JW/TransformerGO_py_version/datasets/onto2vec-datasets-string/data/9606_esm2_embeddings.pkl"

fasta_path = "/home/25p_01/JW/ADSModel_STRING/datasets/ADSLab_dataset/interaction-datasets/7227.protein.sequences.v12.0.fa.gz"
save_path = "/home/25p_01/JW/ADSModel_STRING/datasets/ADSLab_dataset/interaction-datasets/7227_esm2_embeddings.pkl"

# ESM-2 모델 및 토크나이저 로드
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 임베딩 저장 딕셔너리
embedding_dict = {}

# 압축된 FASTA 파일 열기
with gzip.open(fasta_path, "rt") as handle:
    records = list(SeqIO.parse(handle, "fasta"))

print(f"총 {len(records)}개의 단백질 시퀀스를 처리합니다...")

for record in tqdm(records):
    protein_id = record.id.split('|')[1] if '|' in record.id else record.id
    seq = str(record.seq)

    # ESM2는 amino acid 사이에 공백을 넣지 않음
    # tokens = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)

    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # CLS 토큰 임베딩 사용
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()

    embedding_dict[protein_id] = embedding

# 딕셔너리 저장 (Pickle)
with open(save_path, "wb") as f:
    pickle.dump(embedding_dict, f)

print(f"ESM-2 임베딩이 {save_path}에 저장되었습니다.")
