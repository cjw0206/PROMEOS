import gzip
import torch
import pickle
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO
from tqdm import tqdm

# enter your pasta file path and save path
fasta_path = "/home/25p_01/JW/ADSModel_STRING/datasets/STRING_dataset/interaction-datasets/7227.protein.sequences.v12.0.fa.gz"
save_path = "/home/25p_01/JW/ADSModel_STRING/datasets/STRING_dataset/interaction-datasets/7227_esm2_embeddings.pkl"

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embedding_dict = {}

with gzip.open(fasta_path, "rt") as handle:
    records = list(SeqIO.parse(handle, "fasta"))

print(f"Total Protein: {len(records)}")

for record in tqdm(records):
    protein_id = record.id.split('|')[1] if '|' in record.id else record.id
    seq = str(record.seq)

    tokens = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)

    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()

    embedding_dict[protein_id] = embedding

with open(save_path, "wb") as f:
    pickle.dump(embedding_dict, f)

print(f"save path: {save_path}")
