# %%
import pandas as pd
from collections import defaultdict
import math
import pickle
import numpy as np
import random
from random import shuffle
import torch
import torch.utils.data as data
import os

import requests
import re
import gzip
import collections
from sklearn.model_selection import train_test_split
random.seed(10)
# %%
def get_GO_explanation(go_terms, obo_csv_path = "..\\term-encoding-module\\go-basic.obo.csv"):
    go_terms_csv = pd.read_csv(obo_csv_path)
    go_terms = [ "['" + go + "']" for go in go_terms ]
    go_terms_csv = go_terms_csv.query('id == @go_terms')
    
    #do the query one by one to keep the order (using list query does not maintain it)
    final_query = go_terms_csv.query('id == @go_terms[0]')
    for i in range(1, len(go_terms)):
        go = go_terms[i]
        query = go_terms_csv.query('id == @go')
        final_query = final_query.append(query, ignore_index = True)
    return final_query

def trim_GO_expl(expls):
    return [expl.strip("'[").strip("]'") for expl in expls ]

def process_to_capital(onthologies):
    cap = {"['biological_process']":"BP", "['cellular_component']":"CC", "['molecular_function']": "MF"}
    return [cap[ontho] for ontho in onthologies] 

# %%
def get_STRING_id_dict(aliases_path):
    """ Functions that reads a file containing protein aliases
          and returns this mapping as a dict
    Args:
        aliases_path (str): path to aliases file of proteins eg. ProtA is also known as kv010 in other databases    
    Returns:
    dict: dictionary of interaction aliases : 'ProtA' -> 'kv010'
    """ 
    aliases = pd.read_table(aliases_path, delimiter = "\t", compression = 'gzip', skiprows=[0])
    #alias and proteinID
    return dict(zip(aliases.iloc[:, 1],aliases.iloc[:, 0]))

# %%
def write_experiment_type_from_biogrid_stringDB_dataset(ppi_pth = "onto2vec-datasets-string/data/train/9606.no-mirror.protein.links.v11.0.txt", \
               exper_type_path = "experiment-type-biogrid/BIOGRID-ORGANISM-Homo_sapiens-4.4.199.tab3.txt.gz",  aliases_path = "onto2vec-datasets-string/data/9606.protein.aliases.v11.0.txt.gz"):
      
    """ Functions that reads protein interactions and generates a mapping to the type of interaction
         e.g. HighThroughput or LowThroughput
    Args:
        ppi_pth (str): path to the interaction file
        exper_type_path (str): path to the file containing information about the type of the interaction
        aliases_path (str): path to aliases of proteins eg. ProtA is also known as kv010 in other databases
        
    Saves:
    dict: dictionary of interaction type: IntrId -> 'High Throughput'
    """ 
    
    #upper is used to ensure no case sensitive matching are missed
    aliases = get_STRING_id_dict(aliases_path)
    aliases = {str(k).upper(): v  for k,v in aliases.items()}
    exper_type = pd.read_table(exper_type_path, compression = 'gzip')
        
    #retrieve protein names from biogrid and then try to find a StringDB synonim
    protAs = [aliases.get(str(protA).upper(), "NOT_FOUND") for protA in exper_type['Official Symbol Interactor A']]
    protBs = [aliases.get(str(protB).upper(), "NOT_FOUND") for protB in exper_type['Official Symbol Interactor B']]
    
    # StringID-protA + StringID-protB -> Throughput, meaning interaction id to type of experiment
    exper_type_dict = dict(zip(  [a+b for (a,b) in zip(protAs, protBs)],  exper_type['Throughput'])) 
    
    print('Not found aliases in stringDB: ', len(['1' for x in protAs + protBs if 'NOT_FOUND' in x]))
   
    intrs = get_ppi_list(ppi_pth)
    intr_exper_type = {}
    row = 0
    for (protA,protB) in intrs:
        
        intr_id = str(protA) + str(protB)
        intr_id_mirror = str(protB) + str(protA)
        
        if intr_id in exper_type_dict:
            intr_exper_type[intr_id] = exper_type_dict[intr_id]
        elif intr_id_mirror in exper_type_dict:
            intr_exper_type[intr_id] = exper_type_dict[intr_id_mirror]
        else:
            intr_exper_type[intr_id] = "NOT_FOUND"
            
    print(collections.Counter(intr_exper_type.values())) 
    with open(ppi_pth.split("/")[-1] + ".experiment_type_dict", 'wb') as handle:
        pickle.dump(intr_exper_type, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
def protein_annot_string_to_dict(annot_filename, aliases_path, unique = True):
    
    """ Functions that reads protein annotations and creates a dict 
    Args:
        annot_filename (str): the path to the annotation file
        aliases_path (str): path to aliases of proteins eg. ProtA is also known as kv010 in other databases
        unique (bool): the inclusion of duplicate GO terms in the annotation dictionary
    Returns:
    dict: dictionary of annoatations: ProtId -> [GO1, GO2 ...]
    """ 
    
    string_id_dict =  get_STRING_id_dict(aliases_path)
    protein_go_anno = defaultdict(list)
    
    with gzip.open(annot_filename,'rt', encoding='utf8', errors="ignore") as f:
        for line in f:
            if line.startswith('!'): # Skip header
                continue
            line = line.strip().split('\t')
            protein_id = line[1]
            go_term = line[4]
            if line[6] == 'IEA' or line[6] == 'ND': # Ignore predicted or no data annotations
                continue
            if protein_id not in string_id_dict: # Not in StringDB
                continue
                
            #get the id that corresponds to stringDB
            string_id = string_id_dict[protein_id]
            
            if not unique:
                protein_go_anno[string_id].append(go_term)
            elif(go_term not in protein_go_anno[string_id] ):
                 protein_go_anno[string_id].append(go_term)
    return protein_go_anno

def protein_annot_to_dict(annot_filename, unique = True):
    
    """ Functions that reads protein annotations and creates a dict 
    Args:
        annot_filename (str): the path to the annotation file
        unique (bool): the inclusion of duplicate GO terms in the annotation dictionary
    Returns:
    dict: dictionary of annoatations: ProtId -> [GO1, GO2 ...]
    """ 
    gene_anno = pd.read_csv(annot_filename, sep = "\t", header=None)  
    protein_go_anno = defaultdict(list)

    protein_ids = gene_anno[1].values
    go_terms = gene_anno[4].values
    for i in range (0,len(protein_ids)):
        if not  unique:
            protein_go_anno[protein_ids[i]].append(go_terms[i]) 
        elif( go_terms[i] not in protein_go_anno[protein_ids[i]] ):
            protein_go_anno[protein_ids[i]].append(go_terms[i])   
           
    return protein_go_anno

# %%
def go_embeddings_to_dict(go_embed_pth):
    
    """
    read the embeddings generated by Node2vec
    :return dict of GOid -> embeddings
    """
    #load the embeddings generated by node2vec for each index GO, ignore first information line
    embeddings_dict = {}
    embeddings = open(go_embed_pth).read().splitlines()[1:]
    embeddings = [ x.split(" ") for x in embeddings ]

    for i in range(0, len(embeddings)):
        #set the GO id as the key
        key = int(embeddings[i][0]) 
        #add all the dimension of the embedings as a list of floats
        embeddings_dict[key] = [ float(x) for x in embeddings[i][1:]]
        
    return embeddings_dict

# %%
def get_ppi_list(ppi_pth):
    """reads the PPI file and 
    adds the interactins to a list of tuples
    """
    ppi = []
    with open(ppi_pth, "r") as f:  
        ppi += [ (x.split(",")[0], x.split(",")[1].strip('\n')) if ',' in x else\
                 (x.split("\t")[0], x.split("\t")[1].strip('\n'))  for x in f ] 
    return ppi

# %%
# def get_max_len_seq(dataset):
#     """Finds the protein with the most annotations and returns the size"""
#     batch_features, batch_labels, batch_ids = zip(*dataset)
#
#     max_len = 0
#     for i in range(len(batch_features)):
#         max_len = max(max_len, len(batch_features[i][0]), len(batch_features[i][1]))
#     return max_len

def get_max_len_seq(dataset):
    """Finds the maximum number of GO annotations and adds 1 for ProtBERT token"""
    batch_features, batch_labels, batch_ids = zip(*dataset)

    max_len = 0
    for i in range(len(batch_features)):
        lenA = len(batch_features[i][0][0])  # protA의 GO 임베딩 개수
        lenB = len(batch_features[i][1][0])  # protB의 GO 임베딩 개수
        max_len = max(max_len, lenA, lenB)

    return max_len + 1  # ProtBERT 토큰 추가


# %%
"""This part of the code allows the model to generate embeddings on the go when there is a new batch generated.
This is way more memory efficient than emmbeding the entire dataset and then keep it in memory.
"""
class Dataset_stringDB(torch.utils.data.Dataset):
    #Characterizes a dataset for PyTorch
    def __init__(self, all_ppi, labels, protein_go_anno, go_id_dict_pth, go_embed_pth,  shuffle, aliases_path, stringDB, seq_dict=None):
        self.all_ppi = all_ppi
        self.labels = labels
        self.seq_dict = seq_dict
        
        #load the mapping from 'GO name' to index ex: GO0001 to 1
        with open(go_id_dict_pth, 'rb') as fp:
            go_id_dict = pickle.load(fp)
        go_emb_dict = go_embeddings_to_dict(go_embed_pth)
        
        self.protein_go_anno = protein_go_anno
        self.go_id_dict = go_id_dict
        self.go_emb_dict = go_emb_dict
        self.stringDB = stringDB
        self.shuffle = shuffle
    def __len__(self):
        return len(self.all_ppi)

    def __getitem__(self, index):

        label = self.labels[index]    
        protA,protB = self.all_ppi[index]
        features, idi = get_embedded_proteins(
            self.protein_go_anno, self.go_id_dict, self.go_emb_dict, self.shuffle,
            protA, protB,
            seq_dict=self.seq_dict
        )

        return np.array((features, label, idi), dtype=object)

def filter_interactions(ppi, protein_go_anno, go_id_dict_pth, aliases_path, stringDB, go_name_space_dict_pth, go_filter, intr_set_size_filter, max_intr_size):
    
    """ Checks the annotations of the interacting proteins for certain filters 
    Args:
        ppi (list): the protein interactions ex. ['ProtA ProtB', ...]
        protein_go_anno_pth (str): path to the annotation file
        go_id_dict_pth (str): path to the dictionary of 'GOname -> 1', 1 representing the index found in node2vec edgelist
        go_embed_pth (str): path to the embedings generated by node2vec for GO terms
        aliases_path (str): path to the protein aliases 
        stringDB (bool): there are two different functions that generate dictionaries for protein annotations depending on the dataset
        go_filter (str): what type of GO temrs to keep: ALL, CC, BP or MF        
        intr_set_size_filter (list): the range of the the GO set size: e.g. [0,10] meaning the range from 0 to 10
        max_intr_size (int): the maximum number of interactions to add to the dataset 
        
    Returns:
    list: returns only the valid interactions
    """ 

    #load the mapping from 'GO name' to index ex: GO0001 -> 1
    with open(go_id_dict_pth, 'rb') as fp:
        go_id_dict = pickle.load(fp)
        
    #load the mapping from 'GO name' to namespace: GO0001 -> 'biological_process'
    with open(go_name_space_dict_pth, 'rb') as fp:
        go_name_space_dict = pickle.load(fp)
    
    filtered_ppi = []
    rejected_no_annot = 0
    rejected_filter = 0
    for (protA, protB) in ppi:
        
        for prot in [protA, protB]:
            #fileter those GO terms that are not found in the GO file used by us
            protein_go_anno[prot] = [go for go in protein_go_anno[prot] if go in go_id_dict]

            #Filter GO terms according to Ontology terms ('cellular', 'biological', 'moLecular')
            if go_filter != "ALL":
                protein_go_anno[prot] = [go for go in protein_go_anno[prot] if go_filter in go_name_space_dict[go]]
        
        #check if both proteins have atleast 1 GO term associated
        if len(protein_go_anno[protA]) >0 and len(protein_go_anno[protB]) >0:
            
            # check if intr size is in range defined by filter
            intr_set_size = (len(protein_go_anno[protA]) + len(protein_go_anno[protB]))
            if intr_set_size >= intr_set_size_filter[0] and intr_set_size <= intr_set_size_filter[1]:
                
                #break if we added enoguh interactions to the dataset
                if max_intr_size == len(filtered_ppi):
                    return filtered_ppi, protein_go_anno
                filtered_ppi.append((protA, protB))
            else:
                rejected_filter += 1
        else:
            rejected_no_annot += 1
         
    print("Rejected interactions where at least one protein has no annotation: ", rejected_no_annot)
    print(f"Rejected interactions where go_filter={go_filter} and intr_set_size_filter={intr_set_size_filter}: ", rejected_filter)
    print("Number of interactions:", len(filtered_ppi))
    return filtered_ppi, protein_go_anno


def get_embedded_proteins(protein_go_anno, go_id_dict, go_emb_dict, shuffle, protA, protB,
                          seq_dict=None):
    # 기존 GO 임베딩 처리
    protein_go_anno[protA] = [go for go in protein_go_anno[protA] if go in go_id_dict]
    protein_go_anno[protB] = [go for go in protein_go_anno[protB] if go in go_id_dict]

    emb_protA = [go_emb_dict[go_id_dict[go]] for go in protein_go_anno[protA]]
    emb_protB = [go_emb_dict[go_id_dict[go]] for go in protein_go_anno[protB]]

    if shuffle:
        zippedA = list(zip(emb_protA, protein_go_anno[protA]))
        zippedB = list(zip(emb_protB, protein_go_anno[protB]))
        shuffle(zippedA)
        shuffle(zippedB)
        emb_protA, protein_go_anno[protA] = zip(*zippedA)
        emb_protB, protein_go_anno[protB] = zip(*zippedB)

    emb_protA = [np.array(e) for e in emb_protA]
    emb_protB = [np.array(e) for e in emb_protB]

    protA_seq = seq_dict.get(protA)
    protB_seq = seq_dict.get(protB)

    return ((emb_protA, protA_seq), (emb_protB, protB_seq)), ((protA, protein_go_anno[protA]), (protB, protein_go_anno[protB]))


def get_dataset_split_stringDB(poz_path, neg_path, protein_go_anno_pth, go_id_dict_pth, go_embed_pth, shuffle, aliases_path = "", ratio = [0.8, 0.2, 0],\
                               stringDB = True, go_name_space_dict_pth = "../datasets/transformerGO-dataset/go-terms/go_namespace_dict", go_filter = "ALL",\
                               intr_set_size_filter = [0,500], max_intr_size = None, seq_dict=None):
    
    """ Splitting up the interaction data into train/valid/test and generating embeddings 
    Args:
        poz_path (str): path to the positive interactions
        poz_path (str): path to the negative interactions
        protein_go_anno_pth (str): path to the annotation file
        go_id_dict_pth (str): path to the the GO id dict, e.g. GO1 -> 1
        shuffle (funct): function for shuffling the GO terms
        aliases_path (str): path to the aliases file 
        ratio (list): list of 3 floats specifing the split between train/valid/test
        stringDB (bool): weather we are using stringDB datasets or the Jains datasets
        go_name_space_dict_pth (str): path to the dictionary between name of GO terms and explanation e.g. GO1 -> 'MF molecular function'
        go_filter (str): filter for the GO terms (what terms to keep), e.g.  ALL, CC, BP, MF
        intr_set_size_filter (list): the range of the the GO set size: e.g. [0,10] meaning the range from 0 to 10
        max_intr_size (int): the maximum number of interactions to add to the dataset 
        
    Returns:
    Dataset_stringDB: 4 datasets objects for the train, valid, test and full datasets
    """ 
    ppi_poz = get_ppi_list(poz_path)
    ppi_neg = get_ppi_list(neg_path)  
    
    if stringDB:
        protein_go_anno = protein_annot_string_to_dict(protein_go_anno_pth, aliases_path)
    else:
        protein_go_anno = protein_annot_to_dict(protein_go_anno_pth, True)
    
    ppi_poz, updated_protein_go_anno = filter_interactions(ppi_poz, protein_go_anno, go_id_dict_pth, aliases_path, stringDB, go_name_space_dict_pth, go_filter, intr_set_size_filter, max_intr_size)
    ppi_neg, updated_protein_go_anno = filter_interactions(ppi_neg, updated_protein_go_anno, go_id_dict_pth, aliases_path, stringDB, go_name_space_dict_pth, go_filter, intr_set_size_filter, max_intr_size)
    
    all_ppi = ppi_poz + ppi_neg
    labels =  [1] * len(ppi_poz) + [0] * len(ppi_neg)
    ##shuffle the data such that the poz and neg don't appear toghether
    full_dataset = list(zip(all_ppi, labels))
    random.shuffle(full_dataset)
    all_ppi, labels = zip(*full_dataset)
    full_dataset = Dataset_stringDB(all_ppi, labels, updated_protein_go_anno, go_id_dict_pth, go_embed_pth, shuffle, aliases_path, stringDB, seq_dict=seq_dict)

    sz = len(full_dataset)
    train, valid, test = data.random_split(full_dataset, [int(ratio[0]*sz), int(ratio[1]*sz) , sz - (int(ratio[0]*sz) + int(ratio[1]*sz)) ] )

    return train, valid, test, full_dataset



def get_dataset_split_stringDB_keep_ratio(poz_path, neg_path, protein_go_anno_pth, go_id_dict_pth, go_embed_pth, shuffle, aliases_path="", ratio=[0.8, 0.1, 0.1],
                                          stringDB=True, go_name_space_dict_pth="../datasets/transformerGO-dataset/go-terms/go_namespace_dict",
                                          go_filter="ALL", intr_set_size_filter=[0, 500], max_intr_size=None, seq_dict=None):
    """
    Load STRINGDB positive/negative PPI and split into stratified train/valid/test sets.
    Keeps class balance in each split.

    Returns:
        train_dataset, valid_dataset, test_dataset, full_dataset
    """
    # Load PPI lists
    ppi_poz = get_ppi_list(poz_path)
    ppi_neg = get_ppi_list(neg_path)

    # Build GO annotations
    if stringDB:
        protein_go_anno = protein_annot_string_to_dict(protein_go_anno_pth, aliases_path)
    else:
        protein_go_anno = protein_annot_to_dict(protein_go_anno_pth, True)

    # Apply filtering
    ppi_poz, updated_protein_go_anno = filter_interactions(ppi_poz, protein_go_anno, go_id_dict_pth, aliases_path, stringDB, go_name_space_dict_pth, go_filter, intr_set_size_filter, max_intr_size)
    ppi_neg, updated_protein_go_anno = filter_interactions(ppi_neg, updated_protein_go_anno, go_id_dict_pth, aliases_path, stringDB, go_name_space_dict_pth, go_filter, intr_set_size_filter, max_intr_size)

    # Label and combine
    poz = [(ppi, 1) for ppi in ppi_poz]
    neg = [(ppi, 0) for ppi in ppi_neg]

    def stratified_split(data, r):
        train, temp = train_test_split(data, test_size=r[1] + r[2], random_state=10)
        valid, test = train_test_split(temp, test_size=r[2] / (r[1] + r[2]), random_state=10)
        return train, valid, test

    poz_train, poz_valid, poz_test = stratified_split(poz, ratio)
    neg_train, neg_valid, neg_test = stratified_split(neg, ratio)

    # Merge and shuffle
    train_data = poz_train + neg_train
    valid_data = poz_valid + neg_valid
    test_data  = poz_test  + neg_test

    random.seed(10)
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)
    # save_ppi_splits_once(train_data, valid_data, test_data)

    def wrap_dataset(data):
        ppi, labels = zip(*data)
        return Dataset_stringDB(ppi, labels, updated_protein_go_anno, go_id_dict_pth, go_embed_pth, shuffle, aliases_path, stringDB, seq_dict=seq_dict)

    return wrap_dataset(train_data), wrap_dataset(valid_data), wrap_dataset(test_data), wrap_dataset(train_data + valid_data + test_data)

def save_ppi_splits_once(train_data, valid_data, test_data, save_dir="keep_ratio_split_saved"):
    """
    Save train, valid, test PPI splits to txt files only once.
    Each line: ProtA\tProtB\tLabel
    """

    os.makedirs(save_dir, exist_ok=True)

    def save_split_to_file(data, filename):
        with open(os.path.join(save_dir, filename), "w") as f:
            for (protA, protB), label in data:
                f.write(f"{protA}\t{protB}\t{label}\n")

    save_split_to_file(train_data, "STRING_H_400-600_train.txt")
    save_split_to_file(valid_data, "STRING_H_400-600_valid.txt")
    save_split_to_file(test_data, "STRING_H_400-600_test.txt")
    print(f"Saved PPI splits to '{save_dir}/'")

