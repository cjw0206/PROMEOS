import urllib.request
import gzip
import random
from pathlib import Path
from itertools import combinations

random.seed(10)

def download_file(url, save_path):
    urllib.request.urlretrieve(url, save_path)

def save_interactions(file_name, interactions):
    with open(file_name, 'w') as f:
        for protA, protB in interactions:
            f.write(f"{protA}\t{protB}\n")

def generate_negatives_fast(positive_pairs):
    print("Generating candidate negative pairs...")
    proteins = list(set([p for pair in positive_pairs for p in pair]))
    all_pairs = set(combinations(proteins, 2))  # unordered pairs
    positive_set = set(tuple(sorted(pair)) for pair in positive_pairs)
    candidate_negatives = list(all_pairs - positive_set)
    print(f"Total candidate negatives: {len(candidate_negatives)}")

    random.shuffle(candidate_negatives)
    selected_negatives = candidate_negatives[:len(positive_pairs)]

    neg_proteins = set([p for pair in selected_negatives for p in pair])
    print(f"▶ Unique proteins in POSITIVE pairs: {len(proteins)}")
    print(f"▶ Unique proteins in NEGATIVE pairs: {len(neg_proteins)}")

    return selected_negatives

def generate_dataset(protein_links_path, poz_intr_file_name, neg_intr_file_name,
                     min_score=600, max_score=800):
    self_ppis = 0
    total_lines = 0
    pozitive_intr = []
    poz_proteins = []
    intr_sets = {}

    print("Filtering POSITIVE interactions from STRING...")
    with gzip.open(protein_links_path, 'rt') as f:
        next(f)  # skip header
        for line in f:
            total_lines += 1
            protA, protB, score = line.strip().split()
            score = float(score)

            if protA == protB:
                self_ppis += 1
                continue

            # initialize neighbor sets
            intr_sets[protA] = intr_sets.get(protA, set())
            intr_sets[protB] = intr_sets.get(protB, set())

            # positive pair
            if min_score <= score < max_score and protA not in intr_sets[protB]:
                pozitive_intr.append((protA, protB))
                poz_proteins.extend([protA, protB])

            # update adjacency
            if protA not in intr_sets[protB]:
                intr_sets[protA].add(protB)
                intr_sets[protB].add(protA)

    print("Self interactions skipped:", self_ppis)
    print("Total STRING PPI entries read:", total_lines)
    print(f"Positive interactions with score ∈ [{min_score}, {max_score}):", len(pozitive_intr))
    print("Unique proteins in positive pairs:", len(set(poz_proteins)), "\n")

    print("Saving POSITIVE pairs...")
    save_interactions(poz_intr_file_name, pozitive_intr)

    print("Generating NEGATIVE pairs...")
    negative_intr = generate_negatives_fast(pozitive_intr)
    print("Saving NEGATIVE pairs...")
    save_interactions(neg_intr_file_name, negative_intr)

if __name__ == "__main__":
    organism = 9606
    version = "12.0"
    folder = 'interaction-datasets'
    Path(folder).mkdir(parents=True, exist_ok=True)

    link_file = f'{folder}/{organism}.protein.links.v{version}.txt.gz'
    poz_output = f'{folder}/{organism}.protein.links.400-600.v{version}.txt'
    neg_output = f'{folder}/{organism}.protein.negative.400-600.v{version}.txt'

    # # Optional: 다운로드 코드
    # url = f'https://stringdb-static.org/download/protein.links.v{version}/{organism}.protein.links.v{version}.txt.gz'
    # download_file(url, link_file)

    print("Generating STRING dataset (400 ≤ score < 600)...")
    generate_dataset(link_file, poz_output, neg_output, min_score=400, max_score=600)
