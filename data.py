import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset


class PGMDataset(Dataset):
    def __init__(self, sample_list):
        super(PGMDataset, self).__init__()
        #list->ndarray
        sample_ndarray = np.array(sample_list)
        #ndarray->tensor
        self.sample_tensor = torch.from_numpy(sample_ndarray)
        
    def __getitem__(self, index):
        return self.sample_tensor[index]
    
    def __len__(self):
        return len(self.sample_tensor)


def read_all_triples(root_path: str, sample_type: str):
    suffix = ["train", "valid", "test"]
    entity_dict = {}
    entity_idx = 0
    relation_dict = {}
    relation_idx = 0
    sample_list = []
    for s in suffix:
        file_name = root_path + f"/fb15k237.0.{s}.graph"
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                head_entity, relation, tail_entity = line.strip().split("\t")
                if head_entity not in entity_dict:
                    entity_dict[head_entity] = entity_idx
                    entity_idx += 1
                if tail_entity not in entity_dict:
                    entity_dict[tail_entity] = entity_idx
                    entity_idx += 1
                if relation not in relation_dict:
                    relation_dict[relation] = relation_idx
                    relation_idx += 1
                # sample_list:每个三元组的idx形式,根据实际需要构造出相应类型('train, valid, test')sample_list 并返回
                if sample_type == s:
                    sample_list.append([entity_dict[head_entity], relation_dict[relation], entity_dict[tail_entity]])
        f.close()
    return entity_dict, relation_dict, sample_list


if __name__=="__main__":
    root_to_triples = "../MyPra/DATA/processed/fb15k237/hold_out_0"
    entity_dict, relation_dict, sample_list = read_all_triples(root_to_triples)
