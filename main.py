import argparse
import torch
from torch import nn
from data import read_all_triples
from Train import train
import os

# Training settings
parser = argparse.ArgumentParser(description='Type Embedding Implementation')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num_processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--embedding_dim', type=int, default=128, 
                    help='the dimention of the embedding')
parser.add_argument('--type_number', type=int, default=64,
                    help='the number of the entity types')


if __name__=="__main__":
    #print(os.path.dirname())
    root_to_triples = "../../MyPra/DATA/processed/fb15k237/test_case/fb15k237/hold_out_0/"
    # 构造出训练用的sample_list，传入'trian'作为参数
    entity_dict, relation_dict, sample_list = read_all_triples(root_to_triples, 'train')
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train(args, device, entity_dict, relation_dict, sample_list)
