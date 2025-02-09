import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def load_data(file_path):
    '''
        argument:
            file_path: ./data/FB15k-237
        
        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''

    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'entities.dict'), 'r', encoding='utf-8') as f:
        entity2id = dict()

        for line in f:
            # eid, entity = line.strip().split('\t')
            eid, entity = line.strip().split()
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relations.dict'), 'r', encoding='utf-8') as f:
        relation2id = dict()

        for line in f:
            # rid, relation = line.strip().split('\t')
            rid, relation = line.strip().split()
            relation2id[relation] = int(rid)

    train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets


def generate_graph_and_labels(triplets, batch_size, split_size, num_entity, num_rels, negative_rate):
    # edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges = triplets
    graph_size = len(edges)
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(graph_size * split_size)  # 450 900*0.5
    graph_split_ids = np.random.choice(np.arange(graph_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype=torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype=torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype=torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data


def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # head, relation, tail = line.strip().split('\t') #5.31 修改
            head, relation, tail = line.strip().split()
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    # 把边进行独热编码 输入是tensor 和 编码长度
    one_hot = F.one_hot(edge_type, num_classes=2 * num_relation).to(torch.float)
    # ?不懂
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm
