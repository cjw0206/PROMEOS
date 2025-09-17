
import urllib.request
import gzip
import random
from pathlib import Path

random.seed(10)

def download_file(url, save_path):
    urllib.request.urlretrieve(url, save_path)

def save_interactions(file_name, interactions):
    with open(file_name, 'w') as f:
        for protA, protB in interactions:
            f.write(f"{protA}\t{protB}\n")

def generate_dataset(protein_links_path, poz_intr_file_name, neg_intr_file_name, score_treshold = 800):
    self_ppis = 0
    all_prot_nr = 0
    poz_proteins = []
    pozitive_intr = []
    negative_intr = []
    intr_sets = {}
    line_nr = 0

    print("Filtering POSITIVE interactions...")
    with gzip.open(protein_links_path, 'rt') as f:
        next(f)  # skip header
        for line in f:
            line_nr += 1
            protA, protB, score = line.strip().split()

            if protA == protB:
                self_ppis += 1
                continue  # self-loop 제거

            intr_sets[protA] = intr_sets.get(protA, set())
            intr_sets[protB] = intr_sets.get(protB, set())

            if float(score) >= score_treshold and protA not in intr_sets[protB]:
                pozitive_intr.append((protA, protB))
                poz_proteins.append(protA)
                poz_proteins.append(protB)

            if protA not in intr_sets[protB]:
                intr_sets[protA].add(protB)
                intr_sets[protB].add(protA)

    print("Self interactions in STRING-DB file:", self_ppis)
    print('Total number of positive interactions in STRING-DB file:', line_nr)
    print(f'Total number of positive interactions with confidence >= {score_treshold}:', len(pozitive_intr))
    print('Total number of proteins in the selected positive interactions:', len(set(poz_proteins)), "\n")

    print('Saving POSITIVE interactions to files...')
    save_interactions(poz_intr_file_name, pozitive_intr)

    print("Generating NEGATIVE interactions that do not appear in STRING-DB (regardless of confidence score)...")
    negative_intr = generate_negatives_fast(pozitive_intr)
    # while len(negative_intr) < len(pozitive_intr):
    #     protA, protB = random.sample(poz_proteins, 2)
    #
    #     if (protA, protB) in negative_intr or (protB, protA) in negative_intr:
    #         continue
    #
    #     if protA not in intr_sets[protB]:
    #         negative_intr.append((protA, protB))
    #         neg_proteins.append(protA)
    #         neg_proteins.append(protB)
    #
    #         if len(negative_intr) % 20000 == 0:
    #             print("Negative interactions added: ", len(negative_intr), f"/{len(pozitive_intr)}")
    #
    # print('Total number of negative interactions:', len(negative_intr))
    # print('Total number of proteins in the selected negative interactions:', len(set(neg_proteins)), "\n")

    print('Saving NEGATIVE interactions to files...')
    save_interactions(neg_intr_file_name, negative_intr)
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

if __name__ == "__main__":
    organism = 9606
    version = "12.0"
    folder = 'interaction-datasets'
    Path(folder).mkdir(parents=True, exist_ok=True)

    link_file = f'{folder}/{organism}.protein.links.v{version}.txt.gz'
    poz_output = f'{folder}/{organism}.protein.links.v{version}.txt'
    neg_output = f'{folder}/{organism}.protein.negative.v{version}.txt'

    # url = f'https://stringdb-static.org/download/protein.links.v{version}/{organism}.protein.links.v{version}.txt.gz'
    # Path("stringDB-files").mkdir(parents=True, exist_ok=True)
    # download_file(url, link_file)

    print("STRING interaction file downloaded. Generating dataset...")
    generate_dataset(link_file, poz_output, neg_output)


# -------------------------------
# GO Ontology and Annotation Files
# -------------------------------
# print("\nDownloading Gene Ontology and annotation files...")
#
# folder = 'go-terms'
# Path(folder).mkdir(parents=True, exist_ok=True)
#
# go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
# go_obo_path = folder + '/go-basic.obo'
# download_file(go_obo_url, go_obo_path)
# print("GO ontology downloaded")
#
# # Download Human GO annotation
# goa_human_url = 'http://geneontology.org/gene-associations/goa_human.gaf.gz'
# goa_human_path = folder + '/goa_human.gaf.gz'
# download_file(goa_human_url, goa_human_path)
# print("Human GO annotation file downloaded")
#
# # Download Yeast GO annotation
# goa_yeast_url = 'http://current.geneontology.org/annotations/sgd.gaf.gz'
# goa_yeast_path = folder + '/sgd.gaf.gz'
# download_file(goa_yeast_url, goa_yeast_path)
# print("Yeast GO annotation file downloaded")