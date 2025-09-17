import urllib.request
import json 
import collections
import pandas as pd
import numpy as np

def download_ontology(url = 'http://release.geneontology.org/2018-09-05/ontology/go-basic.obo',
                      save_path = r'go-basic.obo'):
    urllib.request.urlretrieve(url, save_path)
    return

def obo_file_to_dict(filename):
    
    """ Function that reads an obo file and creates a GO dict with information

    Args:
        filename (string): path to the .obo file 

    Returns:
    dict: dictionary containing mapping GO -> [info about relations,names,definition]
    list: list that contains the unique tags in the dictionary
    """
    
    ONLY_ONE_ALLOWED_PER_STANZA = set(["id", "name", "def", "comment"])
    unique_tags = set([])

    current_type = None
    current_dict = None
    obo_dict = collections.OrderedDict()
    with open(filename) as lines: 
  
        for line in lines:
        
            #ignore the information from the head of the file
            if line.startswith("["):
                current_type = line.strip("[]\n")
                continue
            if current_type != "Term":
                continue
        
            #remove new-line character and comments
            line = line.strip().split("!")[0]
            if len(line) == 0:
                continue
            
            #take line and divide into tag and value
            line = line.split(": ")
            tag = line[0]
            value = line[1]
        
            unique_tags.add(tag)
        
            #create new record for the new GO term
            if tag == "id":
                current_record = collections.defaultdict(list)
                obo_dict[value] = current_record
            
            if tag in current_record and tag in ONLY_ONE_ALLOWED_PER_STANZA:
                raise ValueError("more than one '%s' found in '%s' " % (tag, ", ".join([current_record[tag], value])) )
        
            current_record[tag].append(value)
            
    return obo_dict, unique_tags

def obo_dict_to_pandas(obo_dict, unique_tags):
    
    """ Function that creates a .csv file from dictionary information
    Args:
        obo_dict (dictionary): information about each GO term
        unique_tags (list): column names of GO information (id, name, is_a etc)

    Returns:
    panda: returns the .csv file created
    """
    obo_panda = pd.DataFrame(columns = list(unique_tags))
    list_of_rows = []
    
    for key, dicto in obo_dict.items():
        new_row = pd.DataFrame([dicto])
        list_of_rows.append(new_row)
    
    obo_panda = pd.concat(list_of_rows, axis=0)    
    
    return obo_panda

filename = '../datasets/ADSLab_dataset/go-terms/go-basic.obo'
obo_dict, unique_tags = obo_file_to_dict(filename)
obo_panda = obo_dict_to_pandas(obo_dict, unique_tags)
obo_panda.to_csv(filename + ".csv", index=False)