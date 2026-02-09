import pandas as pd
import gzip
from collections import defaultdict
import random
from Bio import SeqIO
import random
from itertools import combinations

random.seed(10)

# Step 1: BioGRID data read and filtering
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


# Step 2: Symbol → STRING ID mapping
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

# Step 3: choose proteins that have sequence
def get_valid_proteins_from_fasta(fasta_path):
    valid_proteins = set()
    with gzip.open(fasta_path, 'rt') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            valid_proteins.add(record.id.split()[0])
    return valid_proteins

# Step 4: symbol → string id, sequence filtering
def convert_to_string_ids(df, mapping_dict, valid_proteins):
    interactions = set()
    for a, b in zip(df["Official Symbol Interactor A"], df["Official Symbol Interactor B"]):
        ids_a = mapping_dict.get(a, [])
        ids_b = mapping_dict.get(b, [])
        for ida in ids_a:
            for idb in ids_b:
                # A != B and (A,B), (B,A) duplication delete
                if ida != idb and ida in valid_proteins and idb in valid_proteins:
                    ordered_pair = tuple(sorted([ida, idb]))
                    interactions.add(ordered_pair)
    print(f'positive interaction: {len(interactions)}')
    return list(interactions)


# Step 5: Negative sampling (1:1 ratio)
def generate_negatives(positive_pairs):
    # proteins = set([p for pair in positive_pairs for p in pair])
    proteins = [p for pair in positive_pairs for p in pair]
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
    print("Generating candidate negative pairs...")
    # no duplicated protein list
    proteins = list(set([p for pair in positive_pairs for p in pair]))

    # all possible pairs
    all_pairs = set(combinations(proteins, 2))  # (A, B) with A < B

    positive_set = set(tuple(sorted(pair)) for pair in positive_pairs)
    candidate_negatives = list(all_pairs - positive_set)

    print(f"Total candidate negatives: {len(candidate_negatives)}")

    # shuffle randomly and select from the front
    random.shuffle(candidate_negatives)
    selected_negatives = candidate_negatives[:len(positive_pairs)]

    neg_proteins = set([p for pair in selected_negatives for p in pair])

    print(f"▶ Unique proteins in POSITIVE pairs: {len(proteins)}")
    print(f"▶ Unique proteins in NEGATIVE pairs: {len(neg_proteins)}")

    return selected_negatives


# Step 6: save file
def save_pairs(path, pairs):
    with open(path, 'w') as f:
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")

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
