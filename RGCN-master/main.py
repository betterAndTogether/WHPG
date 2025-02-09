import argparse
import numpy as np
import torch
import random
import torch.nn as nn
from tqdm import tqdm, trange
from dataGeneration import dataGeneration
from utils import load_data, generate_graph_and_labels
from models import RGCN


def main(args):
    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data('./data/recipe')
    # 这里设置关系数量
    num_relations = 8
    num_entities = len(entity2id)
    # 这里设置训练数据
    train_triplets = dataGeneration()
    model = RGCN(num_entities, num_relations, num_bases=args.n_bases, dropout=args.dropout)
    train_data = generate_graph_and_labels(train_triplets, batch_size=args.graph_batch_size,
                                           split_size=args.graph_split_size, num_entity=len(entity2id),
                                           num_rels=num_relations,
                                           negative_rate=args.negative_sample)
    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    print(model)
    print(entity_embedding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')

    # 这里设置子图的边的数量为900
    parser.add_argument("--graph-batch-size", type=int, default=900)  # 数据总量是27w 3w 1/9  现在是1800 200
    parser.add_argument("--graph-split-size", type=float, default=0.5)
    parser.add_argument("--negative-sample", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=100000)
    parser.add_argument("--evaluate-every", type=int, default=1000)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-bases", type=int, default=4)

    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--grad-norm", type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    main(args)