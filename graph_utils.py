import numpy as np
import pandas as pd
import pickle
import csv
import os
import torch
import sys
import jsonlines
import argparse
import random
random.seed(42)
import itertools
import logging
from collections import defaultdict


### Feature processing
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from utils import json_util
import torch.nn.functional as F
from torch_scatter import scatter_add

from filelock import FileLock

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

### Data Loader and mini-batch preperation
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data


seed =42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def read_examples_from_file(data, connection_type, mode, link2id, linkType2id):

        max_node_num_per_process = 0
        guid_index = 1
        examples = []
        pos_of_examples = []
        syn_of_examples = []
        example_size = []
        edges = []
        atten_edges = []
        conn_edges = []
        examples_added = 0
        sent_types = []  # sentence_type
        conn_edge_types = []  #
        for eachfile in data:
            filetext = eachfile['data']
            if len(filetext) <= 1:  # 只包含0/1个句子的场景，不参与句子运算
                continue
            if max_node_num_per_process < len(filetext):
                max_node_num_per_process =  len(filetext)
            # print(filetext)
            edge_index = []
            each_process_edge_type = []
            each_process_sen_type = []
            example = []
            pos_of_example = []
            syn_of_example = []
            for idx, eachsentattr in enumerate(filetext):
                # words = eachsentattr['sentence'].split()
                words = [value for key, value in eachsentattr['id2token'].items()]
                example.append(words)
                pos_of_example.append(eachsentattr["pos"])
                syn_of_example.append(eachsentattr['syntatic_triplets'])
                each_process_sen_type.append(eachsentattr['sen_type'])
                guid_index += 1
                relations = eachsentattr['relation_id']  # 即关联的关系
                relation_types = eachsentattr['rel_type']  # 关系的类型

                for each_rel in relations:
                    edge_index.append((idx, each_rel))

                for each_type in relation_types:
                    each_process_edge_type.append(each_type)

            examples_added += len(filetext)
        
            example_size.append(len(filetext))
            conn_edges.append(edge_index)
            conn_edge_types.append(each_process_edge_type)
            sent_types.append([link2id[type_name] for type_name in each_process_sen_type])  # 等待
            # each_process_sen_type_bi = []
            # for type_name in each_process_sen_type:
            #     tmp = [0, 0, 0]
            #     if type_name == "both":
            #         tmp = [0, 1, 1]
            #     else:
            #         tmp[link2id[type_name]] = 1
            #     each_process_sen_type_bi.append(tmp)
            # sent_bi_types.append(each_process_sen_type_bi)

            edges.append(list(itertools.combinations(range(0, len(filetext)), 2)))  # 穷举了所有
            examples.append(example)
            pos_of_examples.append(pos_of_example)
            syn_of_examples.append(syn_of_example)

        # 构建link prediction 和  link type classification的label
        link_labels = []
        link_type_labels = []
        for ee, ec, et in zip(edges, conn_edges, conn_edge_types):
            each_label = []
            each_link_type_label = []
            type_idx = 0
            for e in ee:
                if e in ec:
                    each_label.append(1)
                    each_link_type_label.append(linkType2id[et[type_idx]])
                    type_idx += 1
                else:
                    each_label.append(0)
                    each_link_type_label.append(linkType2id['none'])
            link_labels.append(each_label)
            link_type_labels.append(each_link_type_label)
        print("Examples to Graph Structure Stats :", len(edges), len(conn_edges),len(example_size),len(examples), len(link_labels), len(link_type_labels))

        if connection_type == "complete":
            conn_edges = edges
        elif connection_type == "linear":
            linear_edges = []
            for each in edges:
                eachgraph_linear_edges = []
                for each_edge in each:
                    # First node in edge is 1 diff from other node
                    if each_edge[1]-each_edge[0] == 1:
                        eachgraph_linear_edges.append(each_edge)
                linear_edges.append(eachgraph_linear_edges)
            conn_edges = linear_edges

        return examples, pos_of_examples, syn_of_examples, example_size, conn_edges, edges, link_labels, link_type_labels, sent_types, max_node_num_per_process


def generate_graph_data(triplets, num_rels):
    edges = np.array(triplets)
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))

    src = torch.tensor(src, dtype=torch.long).contiguous()
    dst = torch.tensor(dst, dtype=torch.long).contiguous()
    rel = torch.tensor(rel, dtype=torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)

    return data


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
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

    
def convert_examples_to_features(examples : List, pos_of_examples: List, syn_of_examples: List, max_seq_length: int,
                                 pos2id: dict, syn_rel2id: dict, tokenizer: PreTrainedTokenizer, model_type,
                                 cls_token_at_end=False, cls_token="[CLS]",
                                 cls_token_segment_id=0, sep_token="[SEP]", sep_token_extra=False, pad_on_left=False,
                                 pad_token=0, pad_token_segment_id=0, pad_token_label_id=-100, sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):  # cls_token_segment_id=1 to check why it was 1
    all_features = []
    for each_process, each_pos_of_process, each_syn_of_process in zip(examples, pos_of_examples, syn_of_examples):  # 遍历每一个程序
        features = []

        for example, pos_of_example, syn_of_example in zip(each_process, each_pos_of_process, each_syn_of_process):
            tokens = []
            pos_of_tokens = []
            synid_of_tokens = []
            input_ids = []

            if "roberta_1" not in model_type:  # update  ？？ 为什么要拒绝roberta_1?
                word_idx = 1  # 语法解析是从1开始，对单词进行编码。下标 （0， 1）默认为root语法关系
                for word, pos in zip(example, pos_of_example):
                    word_tokens = tokenizer.tokenize(word)  # =tokenizer(word)
                    if len(word_tokens) > 0:
                        tokens.extend(word_tokens)
                        pos_of_tokens.extend([pos] * len(word_tokens))  # 如果单词被拆分为多个时候，词性也对应对应复制
                        synid_of_tokens.extend([word_idx] * len(word_tokens))  # 同理，如果单词
                    word_idx += 1

                # Account for [CLS] and [SEP] with "- 2"
                special_tokens_count = tokenizer.num_special_tokens_to_add()
                if len(tokens) > max_seq_length - special_tokens_count:
                    tokens = tokens[: (max_seq_length - special_tokens_count)]
                    pos_of_tokens = pos_of_tokens[: (max_seq_length - special_tokens_count)]
                    synid_of_tokens = synid_of_tokens[: (max_seq_length - special_tokens_count)]

                tokens += [sep_token]
                pos_of_tokens += [sep_token]
                synid_of_tokens += [0]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]
                    pos_of_tokens += [sep_token]
                segment_ids = [sequence_a_segment_id] * len(tokens)  # [ 0, 0, 0, 1 , 1, 1 ]
                if cls_token_at_end:
                    tokens += [cls_token]
                    pos_of_tokens += [cls_token]
                    segment_ids += [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    pos_of_tokens = [cls_token] + pos_of_tokens
                    synid_of_tokens = [0] + synid_of_tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                """
                    ID 化 
                """
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                pos_ids = [pos2id[pos_item] for pos_item in pos_of_tokens]
                syn_input_ids = synid_of_tokens
                syn_triplets = [[triplet[0], syn_rel2id[triplet[1]], triplet[2]] for triplet in syn_of_example]

            else:
                print("Do Not Implement")
                exit()
                # ### For RoBERTa ###
                # example_sent = " ".join(example)
                # input_ids = tokenizer.encode(example_sent, add_special_tokens=True,  max_length=min(max_seq_length, tokenizer.max_len))
                # tokens = tokenizer.decode(input_ids).split() # This is just to show not used so <s> , . will get attached to some other tokens on splits
                # segment_ids = [sequence_a_segment_id] * len(input_ids)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)

            input_ids += [pad_token] * padding_length
            pos_ids += [pad_token] * padding_length
            syn_input_ids += [0] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

            # 过滤超过max_length的三元组进行删除
            filtered_syn_triplets = []
            for triplet in syn_triplets:
                if triplet[0] in syn_input_ids and triplet[2] in syn_input_ids:
                    filtered_syn_triplets.append(triplet)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(pos_ids) == max_seq_length
            assert len(syn_input_ids) == max_seq_length

            graph_data = generate_graph_data(filtered_syn_triplets, num_rels=len(syn_rel2id))

            features.append([tokens, input_ids, input_mask, segment_ids, pos_ids, syn_input_ids, graph_data])
        all_features.append(features)
    return all_features


def prepare_data_graph(features, conn_edges, edges, labels, link_type_labels, sent_types):
    
    data_list = []
    for eachgraph_features, eachgraph_connedges, eachgraph_alledges, eachgraph_labels, eachgraph_linkType, eachgraph_senType in zip(features, conn_edges, edges, labels, link_type_labels, sent_types):

        txt_tokens = [e[0] for e in eachgraph_features]
        token_ids = [e[1] for e in eachgraph_features]
        input_masks = [e[2] for e in eachgraph_features]
        segment_ids = [e[3] for e in eachgraph_features]
        pos_ids = [e[4] for e in eachgraph_features]
        syn_input_ids = [e[5] for e in eachgraph_features]
        syn_graph_data = [e[6] for e in eachgraph_features]
        neg_index = [idx for idx, e in enumerate(eachgraph_labels) if e == 0]
        sen_lengths = [np.sum(mask_arr) for mask_arr in input_masks]
        edge_distance = [e[1]-e[0] for e in eachgraph_alledges]

        # 考虑是否全部转变成tensor
        each_feature = torch.tensor(token_ids, dtype=torch.long)
        each_pos_feature = torch.tensor(pos_ids, dtype=torch.long)
        each_dist_feature = torch.tensor(edge_distance, dtype=torch.long)
        each_syn_feature = torch.tensor(syn_input_ids, dtype=torch.long)
        each_edge_index = torch.tensor(eachgraph_connedges, dtype=torch.long).T

        data = Data(x=each_feature,
                    txt_tokens= txt_tokens,
                    edge_index=each_edge_index, 
                    all_edges=eachgraph_alledges, 
                    y=eachgraph_labels,
                    linkType=eachgraph_linkType,
                    senType=eachgraph_senType,
                    input_masks=input_masks,
                    segment_ids=segment_ids,
                    pos_ids=each_pos_feature,
                    syn_input_ids=each_syn_feature,
                    syn_graph_data=syn_graph_data,
                    neg_index = neg_index,
                    sen_lengths= sen_lengths,
                    dis_distances=each_dist_feature
                    )
        data_list.append(data)
        
    return data_list


def balance_connected_unconnected_edges(edges, labels):
    
    bal_edges = []
    bal_labels = []
    for eedges, elabels in zip(edges, labels):
        distr = defaultdict(list)
        each_bal_edges = []
        each_bal_labels = []
        for e,l in zip(eedges,elabels):
            distr[l].append(e)
        
        random.shuffle(distr[0])    #select same number of no_connections as connections
        to_consider = len(distr[1]) #Since less
        temp_list = []
        for each in distr[1]:
            temp_list.append((each,1))
        for each in distr[0][:to_consider]:
            temp_list.append((each,0))
        random.shuffle(temp_list)
        for each in temp_list:
            each_bal_edges.append(each[0])
            each_bal_labels.append(each[1])
        
        assert len(each_bal_edges)==len(each_bal_labels)
        
        bal_edges.append(each_bal_edges)
        bal_labels.append(each_bal_labels)

    assert len(edges) == len(bal_edges)
    assert len(labels) == len(bal_labels)
    
    return bal_edges, bal_labels
    
    
def windowed_pairs_selection(edges, conn_edges, labels, link_type_labels, window):
    
    win_edges, win_labels, win_type_labels = [], [], []
    for eedges, elabels, etypelabels in zip(edges, labels, link_type_labels):
        sel_edges, sel_labels, sel_typelabels = [], [], []
        for e, l, ltype in zip(eedges, elabels, etypelabels):
#             print(e,l, e[1]-e[0])
            # Remove considering those edges between nodes which are more than window_size distance apart and are all 0s
            if e[1]-e[0] > window and l==0:
                continue
            sel_edges.append(e)
            sel_labels.append(l)
            sel_typelabels.append(ltype)

        win_edges.append(sel_edges)
        win_labels.append(sel_labels)
        win_type_labels.append(sel_typelabels)

    # selecting graph connections which are within the window
    win_conn_edges = []
    for each_graph_conn in conn_edges:
        sel_conn_edges = []
        for e in each_graph_conn:
            if e[1]-e[0] > window:
                continue
            sel_conn_edges.append(e)
        win_conn_edges.append(sel_conn_edges)

    return win_edges, win_conn_edges, win_labels, win_type_labels

def find_label_stats(labels):
    
    l = defaultdict(int)
    for each in labels:
        for e in each:
            l[e]+=1
            
    return l[0], l[1], round(l[1]/(l[0]+l[1]),2)


def oversample_positive_data(edges, labels):
    
    new_edges, new_labels = [], []
    for eedges, elabels in zip(edges, labels):
        sel_edges, sel_labels = [], []
        for e, l in zip(eedges, elabels):
            sel_edges.append(e)
            sel_labels.append(l)
            if l==1:  # If there is an edge n1n2 oversample that with n2n1 since procedural text no back direction
                sel_edges.append((e[1],e[0]))
                sel_labels.append(l)
                
        new_edges.append(sel_edges)
        new_labels.append(sel_labels)
        
    return new_edges, new_labels
    

def prepare_data(filepath, domain, max_seq_len, tokenizer, window, is_oversample, graph_connection, model_type):


    Xtrain = []
    Xval = []
    Xtest = []
    with jsonlines.open(os.path.join(filepath, 'train.jsonl'),'r') as f:
        for each in f:
            Xtrain.append(each)
    with jsonlines.open(os.path.join(filepath, 'test.jsonl'),'r') as f:
        for each in f:
            Xval.append(each)
    with jsonlines.open(os.path.join(filepath, 'test.jsonl'),'r') as f:
        for each in f:
            Xtest.append(each)
    link2id = json_util.load(os.path.join(filepath, 'link2id.json'))
    linkType2id = json_util.load(os.path.join(filepath, 'linkType2id.json'))
    pos2id = json_util.load(os.path.join(filepath, 'pos2id.json'))
    syn_rel2id = json_util.load(os.path.join(filepath, 'syn_rel2id.json'))

    ### Gather Train Features
    ### sentences    : each sentence of each graph
    ### example_size : size of each graph
    ### conn_edges   : graph connections [adj matrix to learn features]
    ### edges        : all edges [complete graph - excluding the self loop]
    ### labels       : ylabels [whether node exists between every pairs of nodes]
    ### link_type_lables : The type of each link
    ### sen_types: The types of the sentence in each flow texts
    sentences, pos_of_sentences, syn_of_sentences, example_size, conn_edges, edges, labels, link_type_labels, sent_types, max_node_num_train= read_examples_from_file(Xtrain, graph_connection, "train", link2id, linkType2id)

    features = convert_examples_to_features(examples=sentences, pos_of_examples=pos_of_sentences, syn_of_examples=syn_of_sentences,
                                            pos2id=pos2id, syn_rel2id=syn_rel2id, max_seq_length=max_seq_len,
                                            tokenizer=tokenizer, model_type=model_type)

    logger.info("Feature length Train: %d", len(features))
#     logger.info("Features: %s", " ".join([str(x) for x in features[0][:3]]))
    
    if is_oversample:
        edges, labels = oversample_positive_data(edges, labels)
    if window:
        edges, conn_edges, labels, link_type_labels = windowed_pairs_selection(edges, conn_edges, labels, link_type_labels, window)  # 按照windows进行切割

    print("Pre-Balanced Train", find_label_stats(labels))
#     edges, labels = balance_connected_unconnected_edges(edges, labels)

    d_train = prepare_data_graph(features, conn_edges, edges, labels, link_type_labels, sent_types)
    print("Post-Balanced Train", find_label_stats(labels))
    _, _, edge_percent = find_label_stats(labels)

    """
        Valid Dataset
    """
    ### Gather Val Features
    sentences, pos_of_sentences, syn_of_sentences, example_size, conn_edges, edges, labels, link_type_labels, sent_types, max_node_num_val= read_examples_from_file(Xval, graph_connection, "val", link2id, linkType2id)
    features = convert_examples_to_features(examples=sentences, pos_of_examples=pos_of_sentences, syn_of_examples=syn_of_sentences,
                                            pos2id=pos2id, syn_rel2id=syn_rel2id, max_seq_length=max_seq_len,
                                            tokenizer=tokenizer, model_type=model_type)
    logger.info("Feature length Val: %d",len(features))
#     logger.info("Features: %s", " ".join([str(x) for x in features[0][:3]]))
    
    if window:
        edges, conn_edges, labels, link_type_labels = windowed_pairs_selection(edges, conn_edges, labels, link_type_labels, window)

    d_val = prepare_data_graph(features, conn_edges, edges, labels, link_type_labels, sent_types)
    print("Val Samples", find_label_stats(labels))

    ### Gather Test Features
    sentences, pos_of_sentences, syn_of_sentences, example_size, conn_edges, edges, labels, link_type_labels, sent_types, max_node_num_test = read_examples_from_file(Xtest, graph_connection, "test", link2id, linkType2id)
#     print(len(sentences), len(example_size), len(conn_edges), len(edges), len(labels))
    
    features = convert_examples_to_features(examples=sentences, pos_of_examples=pos_of_sentences, syn_of_examples=syn_of_sentences,
                                            pos2id=pos2id, syn_rel2id=syn_rel2id, max_seq_length=max_seq_len,
                                            tokenizer=tokenizer, model_type=model_type)
    logger.info("Feature length Val: %d",len(features))

    if window:
        edges, conn_edges, labels, link_type_labels = windowed_pairs_selection(edges, conn_edges, labels, link_type_labels, window)

    d_test = prepare_data_graph(features, conn_edges, edges, labels, link_type_labels, sent_types)
    print("Test Samples",find_label_stats(labels))

    max_node_num = max(max_node_num_train, max_node_num_val, max_node_num_test)
    
    return d_train, d_val, d_test, edge_percent, max_node_num


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datafile", default="", type=str)
    parser.add_argument("--domain", default="cooking", type=str)
    args = parser.parse_args()
    
    data = prepare_data(args.datafile, args.domain)
#     print(data[0])
    loader = DataLoader(data, batch_size=2, shuffle=False)
    
    for batch in loader:
        print("Batch:",batch) 
        logger.info("Graphs: %d, Nodes: %d, Edges: %d", batch.num_graphs, batch.num_nodes, batch.num_edges)