import pandas as pd
import gzip
from collections import defaultdict
import random
from Bio import SeqIO
import random

random.seed(10)

# Step 1: BioGRID 데이터 읽기 및 필터링
def load_and_filter_biogrid(biogrid_path):
    df = pd.read_csv(biogrid_path, sep='\t')
    df = df[
        (df["Organism ID Interactor A"] == 559292) &
        (df["Organism ID Interactor B"] == 559292) &
        (df["Experimental System Type"] == "physical") &
        (
            (df["Throughput"] == "High Throughput") |
            (df["Throughput"] == "Low Throughput")
        )
    ]
    return df[["Official Symbol Interactor A", "Official Symbol Interactor B"]]


# Step 2: Symbol → STRING ID 매핑
def build_symbol_to_stringid(alias_path):
    mapping = defaultdict(set)
    with gzip.open(alias_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            string_id, symbol = parts[0], parts[1]
            mapping[symbol].add(string_id)
    return mapping

# Step 3: sequence 있는 단백질만 추리기
def get_valid_proteins_from_fasta(fasta_path):
    valid_proteins = set()
    with gzip.open(fasta_path, 'rt') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            valid_proteins.add(record.id.split()[0])
    return valid_proteins

# Step 4: symbol → string id로 바꾸고 sequence 존재 필터링
def convert_to_string_ids(df, mapping_dict, valid_proteins):
    interactions = set()
    for a, b in zip(df["Official Symbol Interactor A"], df["Official Symbol Interactor B"]):
        ids_a = mapping_dict.get(a, [])
        ids_b = mapping_dict.get(b, [])
        for ida in ids_a:
            for idb in ids_b:
                # A != B 조건과 (A,B), (B,A) 중복 제거
                if ida != idb and ida in valid_proteins and idb in valid_proteins:
                    ordered_pair = tuple(sorted([ida, idb]))
                    interactions.add(ordered_pair)
    print(f'positive interaction: {len(interactions)}')
    return list(interactions)


# Step 5: Negative sampling (1:1 비율, positive 내 protein pool 기준)
def generate_negatives(positive_pairs):
    # proteins = set([p for pair in positive_pairs for p in pair])
    proteins = [p for pair in positive_pairs for p in pair]  # ✅ 중복 포함됨
    negatives = set()
    while len(negatives) < len(positive_pairs):
        a, b = random.sample(proteins, 2)
        if (a,b) in negatives or (b,a) in negatives:
            continue
        if (a, b) not in positive_pairs and (b, a) not in positive_pairs:
            negatives.add((a, b))

            if len(negatives) % 10000 == 0:
                print("Negative interactions added: ", len(negatives), f"/{len(positive_pairs)}")
    neg_proteins = set([p for pair in negatives for p in pair])

    print(f"▶ Unique proteins in POSITIVE pairs: {len(set(proteins))}")
    print(f"▶ Unique proteins in NEGATIVE pairs: {len(neg_proteins)}")
    return list(negatives)


def generate_negatives_fast(positive_pairs):
    from itertools import combinations

    print("Generating candidate negative pairs...")
    # 중복 제거된 protein 목록
    proteins = list(set([p for pair in positive_pairs for p in pair]))

    # 모든 가능한 조합 (unordered pair)
    all_pairs = set(combinations(proteins, 2))  # (A, B) with A < B

    # positive pair도 (A, B)로 정렬해서 set으로
    positive_set = set(tuple(sorted(pair)) for pair in positive_pairs)

    # negative 후보 = 전체 조합 - positive
    candidate_negatives = list(all_pairs - positive_set)

    print(f"Total candidate negatives: {len(candidate_negatives)}")

    # 무작위로 섞고 앞에서부터 뽑기
    random.shuffle(candidate_negatives)
    selected_negatives = candidate_negatives[:len(positive_pairs)]

    neg_proteins = set([p for pair in selected_negatives for p in pair])

    print(f"▶ Unique proteins in POSITIVE pairs: {len(proteins)}")
    print(f"▶ Unique proteins in NEGATIVE pairs: {len(neg_proteins)}")

    return selected_negatives


# Step 6: 파일 저장
def save_pairs(path, pairs):
    with open(path, 'w') as f:
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")

# Main 실행
# biogrid_path = "BioGrid-files/BIOGRID-ORGANISM-Homo_sapiens-4.4.246.tab3.txt"
biogrid_path = "BioGrid-files/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-4.4.246.tab3.txt"
alias_path = "BioGrid-files/4932.protein.aliases.v12.0.txt.gz"
fasta_path = "interaction-datasets/4932.protein.sequences.v12.0.fa.gz"

biogrid_df = load_and_filter_biogrid(biogrid_path)
symbol_to_stringid = build_symbol_to_stringid(alias_path)
valid_prots = get_valid_proteins_from_fasta(fasta_path)

positive_pairs = convert_to_string_ids(biogrid_df, symbol_to_stringid, valid_prots)
negative_pairs = generate_negatives_fast(positive_pairs)

print("▶ Positive pairs:", len(positive_pairs))
print("▶ Negative pairs:", len(negative_pairs))

save_pairs("4932_biogrid_low_high_positive.txt", positive_pairs)
save_pairs("4932_biogrid_low_high_negative.txt", negative_pairs)
