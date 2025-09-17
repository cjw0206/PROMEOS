import pandas as pd
from collections import defaultdict
import math
import pickle
from pathlib import Path

def obo_csv_trim(csv_path = 'go-basic.obo.csv'):
    
    """
    Function that selects only the necessary 
    values related to GO terms
    :return CSV panda that is filtered 
    """
    
    go_terms = pd.read_csv(csv_path)
    
    #get only those GO terms that are not obsolete (not good anymore)
    valid_go_terms = go_terms.loc[go_terms['is_obsolete'].isna() ]

    #selecting only those relationships mentioned in the paper
    terms_for_node2vec = valid_go_terms[["id", "is_a", "relationship", "namespace"]]
    terms_for_node2vec['id'] = terms_for_node2vec['id'].apply(lambda x: x.strip("['']")) 
    terms_for_node2vec['is_a'] = terms_for_node2vec['is_a'].apply(lambda x:  x.strip("[']").replace(' ', '').split("','") if type(x) is str else x) 
    terms_for_node2vec['relationship'] = terms_for_node2vec['relationship'].apply(lambda x:  x.strip("[]").split(", ") if type(x) is str else x) 
    
    terms_for_node2vec['namespace'] = terms_for_node2vec['namespace'].apply(lambda x:  x.strip("['']") if type(x) is str else x) 
    terms_for_node2vec.reset_index(inplace=True, drop = True)
    terms_for_node2vec['index_mapping'] = terms_for_node2vec.index
    
    return terms_for_node2vec

def create_edge_list(terms_for_node2vec):
    
    """
    Function that takes all the node2vec terms
    adds all the relationships of type 'is_a' and 'part_of'
    :return lists of all the edges
    """
    
    is_a_dict = dict(zip(terms_for_node2vec["index_mapping"].values,
                     terms_for_node2vec["is_a"].values))
    part_of_dict = dict(zip(terms_for_node2vec["index_mapping"].values,
                     terms_for_node2vec["relationship"].values))
    go_to_index_dict = dict(zip(terms_for_node2vec["id"].values,
                     terms_for_node2vec["index_mapping"].values))

    go_graph_edges = defaultdict(list)

    #adding all the 'is_a' edges
    for i, is_a_list in is_a_dict.items():
        if type(is_a_list) is list: #non root GO term that does not have a 'is_a'
            for is_a in is_a_list:
                if type(is_a) is str:
                    go_graph_edges[i].append(go_to_index_dict[is_a])            
    
    #adding all the 'part_of' edges
    for i, part_of_list in part_of_dict.items():
        if type(part_of_list) is list: #no relationship present
            for part_of in part_of_list:
                if type(part_of) is str and "part_of" in part_of:
                    part_of =  part_of.strip("'part_of ").replace("''", "")
                    go_graph_edges[i].append(go_to_index_dict[part_of])    
    return go_graph_edges

def save_go_mapping(terms_for_node2vec, save_path = 'go_id_dict'):
    
    """
    saves the GO terms name with the specific ID
    dict of  |GO name --> ID |
    """
    
    go_to_index_dict = dict(zip(terms_for_node2vec["id"].values,
                     terms_for_node2vec["index_mapping"].values))
    with open(save_path, 'wb') as fp:
        pickle.dump(go_to_index_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return

def write_edge_list(go_graph_edges, save_path = "graph/go-terms.edgelist"):
    
    """Writes all the GO 'is_a' and 'part_of' to a file: ex: 1->2
    Args:
        go_graph_edges (dict): dict of GO relations ex. GO -> [GO1, GO2, ...] 
    """
    
    with open(save_path, "w") as f:  
        for node, edge_list in go_graph_edges.items():
            for edge in edge_list:
                #adding 1 as the weight
                f.write(str(node) + "  " + str(edge)) #+ " " + str(1)) 
                f.write("\n")
    return

def save_go_process(terms_for_node2vec, save_path = 'go_namespace_dict'):
    
    """
    Function that saves the namespace MF,CC, BP
    with the id as a dict   |id --> onthology|
    """
    go_namespace_dict = dict(zip(terms_for_node2vec["id"].values,
                     terms_for_node2vec["namespace"].values))  
    with open(save_path, 'wb') as fp:
        pickle.dump(go_namespace_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return

folder_path = '../datasets/ADSLab_dataset/go-terms/'
terms_for_node2vec = obo_csv_trim(csv_path = folder_path + 'go-basic.obo.csv')
save_go_mapping(terms_for_node2vec, save_path = folder_path + 'go_id_dict')

go_graph_edges = create_edge_list(terms_for_node2vec)
Path(folder_path + "graph").mkdir(parents=True, exist_ok=True)
write_edge_list(go_graph_edges, save_path = folder_path + "graph/go-terms.edgelist")

save_go_process(terms_for_node2vec, save_path = folder_path + 'go_namespace_dict')