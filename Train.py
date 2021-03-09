# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from model import TriplePGM, loss_function
from data import read_all_triples, PGMDataset
from torch import optim
import time
from tqdm import tqdm 
import pdb

def train(args, device, entity_dict, relation_dict, sample_list):
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    type_num = args.type_number
    embedding_dim = args.embedding_dim
    temp = 1.
    # 根据具体数据要求选择train、test以及valid数据集
    train_dataset = PGMDataset(sample_list)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=2)
    print('Data Is Processed')

    device = device
    # 这里的num是否是三种数据集加在一起的？
    entity_num = len(entity_dict.keys())
    relation_num = len(relation_dict.keys())
    print(f'entity number:{entity_num}',
          f'relation number:{relation_num}')
    model = TriplePGM(entity_num,
                      relation_num,
                      type_num,
                      embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print('This is in ' + str(epoch) + ' Epochs')
        t = time.time()
        for step, data in enumerate(tqdm(train_loader)):
            #pdb.set_trace()
            #data: batch*(h_idx, r_idx, t_idx)
            recon_t_given_Tt_score, t_idx, q_1_Th_given_h_r, prior_Th_given_h_r, q_2_Tt_given_t, prior_Tt_given_Th_r = model(data, temp)

            loss = loss_function(q_1_Th_given_h_r, prior_Th_given_h_r, 
                                 q_2_Tt_given_t, prior_Tt_given_Th_r,
                                 recon_t_given_Tt_score, t_idx)
            optimizer.zero_grad()
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()
            if step % 100 == 0:
                print(f'DataShape:{data.shape}',
                    f'Batch:{step + 1}',
                    f'train_loss={cur_loss}')

    
    pass


if __name__ == "__main__":
    pass
