"""
    探索节点关系的特征表示
"""

from collections  import defaultdict
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

from transformers import BertModel, RobertaModel
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d
from transformers import AutoConfig, AutoModel

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU
    
import random
import numpy as np

from RGCN_Model import RGCN
from torch_geometric.data import Data
from torch_scatter import scatter_add


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max')  #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


class StructureAwareAttention(nn.Module):
    def __init__(self, node_embed_size, edge_embed_size, head_num, hidden_size, dropout_rate):
        super(StructureAwareAttention, self).__init__()

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.q_transform = nn.Linear(node_embed_size, hidden_size)
        self.k_transform = nn.Linear(node_embed_size, hidden_size)
        self.v_transform = nn.Linear(node_embed_size, hidden_size)

        self.struct_k_transform = nn.Linear(edge_embed_size, hidden_size // head_num)
        self.struct_v_transform = nn.Linear(edge_embed_size, hidden_size // head_num)
        self.o_transform = nn.Linear(hidden_size, hidden_size)

        self.path_norm = nn.LayerNorm(edge_embed_size)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, nodes, edge_repr, edge_mask_label, bias=0.0):
        edge_mask_label = edge_mask_label.unsqueeze(0).expand(self.head_num, -1, -1)
        q, k, v = self.q_transform(nodes), self.k_transform(nodes), self.v_transform(nodes)  # [node_num, bert_embed_size]
        q = self.split_heads(q, self.head_num)  # [head_num, node_num, hidden_size // head_num]
        k = self.split_heads(k, self.head_num)
        v = self.split_heads(v, self.head_num)  # [head_num, node_num, hidden_size // head_num]
        edges = self.path_norm(edge_repr)  # [node_num, node_num, edge_embed_size]
        struct_k, struct_v = self.struct_k_transform(edges), self.struct_v_transform(edges)  # [node_num, node_num, hidden_size // head_num]
        q = q * (self.hidden_size // self.head_num) ** -0.5
        w = torch.matmul(q, k.transpose(-1, -2))  # [head_num, node_num, node_num]
        struct_w = torch.matmul(q.transpose(0, 1), struct_k.transpose(-1, -2)).transpose(0, 1)  # [head_num, node_num, node_num]
        w = w + struct_w + bias  # 这里缺一个  [head_num, node_num, node_num]  mask
        mask = (edge_mask_label == 0.0)
        w[mask] = -1e9
        # w = w * (edge_mask_label * -1e9) # [head_num, node_num, node_num]
        w = torch.nn.functional.softmax(w, dim=-1)
        output = torch.matmul(w, v) + torch.matmul(w.transpose(0, 1), struct_v).transpose(0, 1)  # [head_num, node_num, hidden_size // head_num]
        output = self.activation(self.o_transform(self.combine_heads(output)))   # [node_num, hidden_size]
        return self.norm(nodes + self.dropout(output))  # [node, hidden_size]

    @staticmethod
    def combine_heads(x):
        heads = x.shape[0]
        node_num = x.shape[1]
        channels = x.shape[2]

        y = torch.transpose(x, 1, 0)
        return torch.reshape(y, [node_num, heads*channels])

    @staticmethod
    def split_heads(x, heads):
        node_num = x.shape[0]
        channels = x.shape[1]
        y = torch.reshape(x, [node_num, heads, channels//heads])
        return y.transpose(1, 0)

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = ~mask * inf
        return torch.unsqueeze(ret, 1)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class MultiGrainedAndSynticEnhancedModel(torch.nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.bert_hidden_size = config.hidden_size
        #self.conv_hidden_2 = 128
        self.conv_hidden_out = 128
        self.num_classes = args.num_classes
        self.num_node_type_classes = args.num_node_classes
        self.num_link_type_classes = args.num_link_classes
        self.pos_embed_size= args.pos_embedding_size
        self.dist_embed_size = args.dist_embeding_size
        self.att_hidden_size = args.att_hidden_size
        self.att_layer_num = args.layer_num
        self.device = args.device
        self.negative_loss_weight = args.negative_loss_weight

        set_seed(args)
        self.bert = BertModel(config)

        """
            SenType Classificaton Task
        """
        self.node_classification_linear_l1 = nn.Sequential(nn.Linear(self.bert_hidden_size+self.pos_embed_size, self.pos_embed_size), nn.ReLU())  # 变成非线性的表示，作为linkPred / LinkTypePred使用
        self.node_classification_linear_l2 = nn.Linear(self.pos_embed_size, self.num_node_type_classes)

        """
            LinkPrediction Task 
        """
        self.node_transform = nn.Linear(self.bert_hidden_size, self.att_hidden_size)
        # self.link_classification_linear = nn.Sequential(nn.Linear(self.bert_hidden_size+self.pos_embed_size, self.conv_hidden_out))
        # self.conv1 = GCNConv(self.bert_hidden_size, self.bert_hidden_size)
        # self.bn1 = BatchNorm1d(self.bert_hidden_size)
        self.link_pred_linear = nn.Sequential(nn.Linear(2 * (self.dist_embed_size+2*self.pos_embed_size), self.dist_embed_size+2*self.pos_embed_size),
                                              nn.Tanh(),
                                              nn.Linear(self.dist_embed_size+2*self.pos_embed_size, 2)
                                              )
        self.sent_gru = nn.GRU(self.bert_hidden_size, self.bert_hidden_size // 2, batch_first=True,
                               bidirectional=True)
        self.dropout = nn.Dropout(args.rgcn_dropout)
        self.dialog_gru = nn.GRU(self.bert_hidden_size, self.bert_hidden_size // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(self.bert_hidden_size)
        self.norm_const_edge = nn.LayerNorm(self.pos_embed_size)

        self.structure_aware_att = StructureAwareAttention(node_embed_size=self.att_hidden_size,
                                                           edge_embed_size=self.dist_embed_size+2*self.pos_embed_size,
                                                           head_num=args.head_num,
                                                           hidden_size=args.att_hidden_size,
                                                           dropout_rate=args.att_dropout)
        self.edge_update = EdgeUpdateModel(node_hidden_size=self.att_hidden_size,
                                           edge_hidden_size=self.dist_embed_size+2*self.pos_embed_size)
        self.att_dropout = nn.Dropout(args.att_dropout)


        """
            Link Type Classification
        """
        self.link_type_pred_linear = nn.Sequential(nn.Linear(2 * (self.dist_embed_size+2*self.pos_embed_size), self.dist_embed_size+2*self.pos_embed_size),
                                                   nn.Tanh(),
                                                   nn.Linear(self.dist_embed_size+2*self.pos_embed_size, self.num_link_type_classes))

        self.edge_percent = args.edge_percent
        weights = [self.edge_percent, 1-self.edge_percent]
        self.weights = torch.tensor(weights, dtype=torch.float).to(self.device)
        self.loss_fn_bi = torch.nn.CrossEntropyLoss(self.weights)
        # self.loss_fn_bi = BCEFocalLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.multi_label_loss_fn = nn.MultiLabelSoftMarginLoss()

        # self.edge_embedding = EdgeEmbedding(args)

        # pos encoder embedding
        self.pos_embedding = torch.rand(args.num_pos_type, self.pos_embed_size)
        self.dist_embedding = torch.rand(args.max_node_num, self.dist_embed_size)

        # syn_graph_rgcn_encoder
        self.max_seq_len = args.max_seq_len
        self.num_syn_rel = args.num_syn_rel
        self.rgcn_encoder= RGCN(args.max_seq_len, args.num_syn_rel, num_bases=args.rgcn_n_bases, dropout=args.rgcn_dropout, entity_embed_size=self.pos_embed_size)

    def link_type_loss_fn(self, link_type_probs, y_linkType_label, label_mask, neg_mask, negative=True):
        link_type_probs_pos = link_type_probs[label_mask]
        y_linkType_label_pos = y_linkType_label[label_mask]
        loss = torch.nn.functional.cross_entropy(link_type_probs_pos, y_linkType_label_pos, reduction="mean")
        if negative:
            negative_loss = torch.nn.functional.cross_entropy(link_type_probs[neg_mask], y_linkType_label[neg_mask], reduction="mean")
        else:
            negative_loss = torch.tensor(0.0)
        return loss, negative_loss

    def link_loss_fn(self, link_probs, y_link_label, label_mask):
        """
        :param link_probs:  [node_num, 2]
        :param label_mask:  [node_num]
        :return:
        """
        # link_probs_pos = link_probs[label_mask]
        # y_link_label_pos = y_link_label[label_mask]
        # pos_loss = nn.functional.cross_entropy(link_probs_pos, y_link_label_pos)
        class_loss = self.loss_fn_bi(link_probs, y_link_label)

        return class_loss

    def forward(self, data):

        # Data Parsing
        node_token_ids = data.x
        pos_ids = data.pos_ids
        syn_input_ids = data.syn_input_ids
        syn_graph_data = data.syn_graph_data
        edge_index = data.edge_index
        all_edges = data.all_edges
        labels = data.y
        input_masks = data.input_masks
        segment_ids = data.segment_ids
        linkType_label = data.linkType
        sentType_label = data.senType
        sen_lengths = data.sen_lengths
        dis_distances_ids = data.dis_distances

        num_graphs = data.num_graphs
        num_nodes = data.num_nodes  # 相当于batch里面所有节点个数（一个节点一句话）
        num_edges = data.num_edges
        batch = data.batch.reshape(num_nodes, 1).to(self.device)

        unfolded_input_mask = [eim for im in input_masks for eim in im]
        input_masks = torch.tensor(unfolded_input_mask, dtype=torch.long).to(self.device)  # previous(num_graphs[[11,50][7,50]]) -> [18,50]
        unfolded_segment_ids = [esi for si in segment_ids for esi in si]
        segment_ids = torch.tensor(unfolded_segment_ids, dtype=torch.long).to(self.device)
        ylabels = torch.tensor([e for each in labels for e in each], dtype=torch.long).to(self.device)  # [node_num]
        y_sentType_label = torch.tensor([e for each in sentType_label for e in each], dtype=torch.long).to(self.device)
        y_linkType_label = torch.tensor([e for each in linkType_label for e in each], dtype=torch.long).to(self.device)
        num_possible_edges_in_batch = len(ylabels)
        sen_lengths = torch.tensor([e for each in sen_lengths for e in each], dtype=torch.long).to(self.device)
        label_mask = (ylabels==1)
        negative_mask = (ylabels==0)

        assert node_token_ids.size() == segment_ids.size()
        assert segment_ids.size() == input_masks.size()

        # bert_encoder
        out = self.bert(node_token_ids, input_masks, segment_ids)
        # cls_tokens = out[1]  # Shape=(num_nodes_in_batch, bert_hidden_size)
        token_semantic_repr = out[0]  # shape=[num_nodes, max_len, bert_hidden_size]

        # pos embedding
        pos_repr = self.pos_embedding[pos_ids].to(self.device)
        dist_repr = self.dist_embedding[dis_distances_ids].to(self.device)

        # syntatic Knowledge embedding
        idx = 0
        all_syn_graph_repr = []
        for triplet_graph_batch in syn_graph_data:
            for triplet_graph in triplet_graph_batch:
                length = len(triplet_graph.entity.tolist())
                token_semantic_repr_tmp = pos_repr[idx][:length]
                syn_repr = self.rgcn_encoder(token_semantic_repr_tmp.to(self.device), triplet_graph.edge_index.to(self.device),
                                             triplet_graph.edge_type.to(self.device), triplet_graph.edge_norm.to(self.device))
                real_len = syn_repr.size()[0]
                if real_len < self.max_seq_len:  # padding
                    m = torch.nn.ZeroPad2d((0, 0, 0, self.max_seq_len-real_len))
                    syn_repr = m(syn_repr)
                else:
                    syn_repr = syn_repr.narrow(0, 0, self.max_seq_len)
                all_syn_graph_repr.append(syn_repr)
                idx += 1
        syn_graph_repr = torch.stack(all_syn_graph_repr, dim=0).to(self.device)  # shape= (num_nodes, max_seq_len,  pos_embed_size)

        #  Syntatic Attention
        num_query = 2
        multi_query = torch.rand(num_query, self.pos_embed_size).to(self.device)  # shape=[num_query, pos_embed_size]
        multi_query = multi_query.unsqueeze(1).unsqueeze(0).expand(num_nodes, -1, self.max_seq_len, -1)  # [num_nodes, num_query, max_seq_len, pos_embed_size]
        syn_graph_repr_reshape = syn_graph_repr.unsqueeze(1).expand(-1, num_query, -1, -1)  # [num_nodes, num_query, max_seq_len, pos_embed_size]
        weight = torch.softmax(torch.sum(multi_query * syn_graph_repr_reshape, -1), -1)  # [num_nodes, num_query, max_seq_len]
        syn_graph_repr_att = torch.sum(weight.unsqueeze(-1).expand(-1, -1, -1, self.pos_embed_size) * syn_graph_repr_reshape, 2)  # shape=[node_num, num_query, pos_embed_size)]
        syn_graph_repr_att = syn_graph_repr_att.sum(1)  # [node_num, pos_embed_size]

        token_level_repr = token_semantic_repr
        # sen_level
        sent_output, sent_hx = self.sent_gru(token_level_repr)  # [node]
        sent_output = self.dropout(sent_output)  # [node_num, max_len, bert_hidden_size]
        sent_output = sent_output.reshape(num_nodes, self.max_seq_len, 2, -1)  # sent_output[0] 前向表示； sen_output[1] 后向表示 [node_num, max_len, 2, bert_hidden_size//2]
        tmp = torch.arange(num_nodes)
        node_level_repr = torch.cat((sent_output[tmp, sen_lengths - 1, 0], sent_output[tmp, 0, 1]), dim=-1)  # [node_num, bert_hidden_size]
        # dialog_level
        # 切割不同的流程，单独使用gru进行编码
        dialog_level_repr = torch.empty(num_nodes, self.bert_hidden_size)  # [num_node, bert_hidden_size]
        each_node_num = []
        start_idx = 0
        for each in sentType_label:
            each_node_level_repr = node_level_repr[start_idx:start_idx+len(each)]
            each_dialog_level_repr, dialog_hx = self.dialog_gru(each_node_level_repr.unsqueeze(0))
            each_dialog_level_repr = self.dropout(each_dialog_level_repr).squeeze()  # [len_of_each_process, bert_hidden_size]
            dialog_level_repr[start_idx:start_idx+len(each)] = each_dialog_level_repr
            start_idx += len(each)
            each_node_num.append(len(each))
        dialog_level_node_repr = self.norm(node_level_repr.to(self.device)+dialog_level_repr.to(self.device))  # [num_node, bert_hidden_size]

        """
            Node Type Classification
        """
        # sent_class_repr = self.node_classification_linear(dialog_level_repr)  # shape=[num_nodes_in_batch, num_node_type_classes]
        dialog_level_senType_repr = torch.cat((dialog_level_node_repr.to(self.device), syn_graph_repr_att.to(self.device)), -1)  # [num_node, conv_hidden_1+pos_embed_size]
        sen_type_repr = self.node_classification_linear_l1(dialog_level_senType_repr)  # [num_nodes, pos_embedding_size]
        sent_class_repr = self.node_classification_linear_l2(sen_type_repr)
        sent_class_prob = torch.nn.functional.softmax(sent_class_repr, dim=-1)
        node_classification_loss = self.loss_fn(sent_class_prob, y_sentType_label)

        const_edge_repr = torch.zeros(num_nodes, num_nodes, self.dist_embed_size+2*self.pos_embed_size)  # [num_nodes, num_nodes, dist_embed_size+pos_embed_size]
        edge_mask_label = torch.zeros(num_nodes, num_nodes) # [num_nodes, num_nodes]
        link_label_mask = torch.zeros(num_nodes, num_nodes)
        last_node = 0
        index = 0
        for i, each_graph_all_edges in enumerate(all_edges):
            max_node_id = 0
            for each_edge in each_graph_all_edges:
                start_idx = each_edge[0]+last_node
                end_idx = each_edge[1]+last_node
                # const_syn_repr = self.norm_const_edge(syn_graph_repr_att[start_idx]+syn_graph_repr_att[end_idx])
                tmp_repr = torch.cat([syn_graph_repr_att[start_idx], dist_repr[index], syn_graph_repr_att[end_idx]], -1)
                # tmp_repr = dist_repr[index]
                const_edge_repr[start_idx, end_idx, :] = tmp_repr
                const_edge_repr[end_idx, start_idx, :] = tmp_repr
                edge_mask_label[start_idx, end_idx] = torch.tensor(1)
                edge_mask_label[end_idx, start_idx] = torch.tensor(1)
                link_label_mask[start_idx, end_idx] = torch.tensor(1)  # 由小指向大的场景
                #  记录
                index += 1
                max_node_id = max(max_node_id, each_edge[0]+last_node, each_edge[1]+last_node)
            last_node = max_node_id+1
        const_edge_repr = const_edge_repr.to(self.device)
        struct_edge_repr = torch.zeros_like(const_edge_repr)  # [num_nodes, node_num, dist_embed_size+pos_embed_size]
        edge_mask_label = edge_mask_label.to(self.device)
        link_label_mask = link_label_mask.to(self.device)

        dialog_level_node_repr = self.node_transform(dialog_level_node_repr)  # shape=[node_num, att_hidden_size]
        for _ in range(self.att_layer_num):
            dialog_level_node_repr = self.structure_aware_att(dialog_level_node_repr, const_edge_repr+struct_edge_repr, edge_mask_label)  # [node_num, hidden_size]
            edge_repr = self.edge_update(dialog_level_node_repr, const_edge_repr, struct_edge_repr, edge_mask_label)  # [node_num, node_num, att_hidden_size+pos_embed_size]
            edge_repr = self.att_dropout(edge_repr)  # [node_num, node_num, att_hidden_size+pos_embed_size]
        edge_repr = edge_repr * (edge_mask_label.unsqueeze(-1).expand(-1, -1, self.dist_embed_size+2*self.pos_embed_size))  # [node_num, node_num, pos_embed_size]

        edge_repr = torch.cat((edge_repr, edge_repr.transpose(0, 1)), -1)   # [node_num, node_num, 2 * (dis_embed_size+pos_embed_size)]

        """
           LinkPred
        """
        link_probs = self.link_pred_linear(edge_repr)  # shape=[node_num, node_num, 2]
        link_probs = torch.nn.functional.softmax(link_probs, -1)  # shape=[node_num, node_num, 2]
        link_probs = link_probs[link_label_mask==1]  # [edge_nums, 2]
        # link_pos_probs = link_probs[:, 1]  # [node_num, 1]
        link_pred_loss = self.link_loss_fn(link_probs, ylabels, label_mask)

        # link_probs_pos = link_probs[:, :, 1]
        # link_probs_pos = link_probs_pos.squeeze()  # [node_num, node_num]
        # link_probs_pos[~(link_label_mask==1)]=-1e9
        # link_probs_pos = torch.nn.functional.softmax(link_probs, -1)  # [node_num, node_num]
        # link_pred_loss_pos = -torch.log(link_probs_pos[link_label_mask==1])
        # link_pred_loss_pos = link_pred_loss_pos.mean(-1)

        # link_probs = self.link_pred_linear(edge_repr)  # [node_num, node_num, 1]
        # link_probs = link_probs.squeeze()  # [node_num, node_num]
        # link_probs[~(link_label_mask==1)]=-1e9
        # link_probs = torch.nn.functional.softmax(link_probs, -1)  # [node_num, node_num]
        # link_pred_loss = -torch.log(link_probs[link_label_mask==1])
        # link_pred_loss = link_pred_loss.mean(-1)
        # link_probs = link_probs[link_label_mask==1]

        # 记录每一个节点，对应那些边。因为每个节点
        link_tags = torch.arange(num_nodes).unsqueeze(1).expand(-1, num_nodes)
        link_tags = link_tags.to(self.device)
        link_tags = link_tags[link_label_mask==1]

        """
            LinkType 
        """
        link_type_probs = self.link_type_pred_linear(edge_repr) # [node_num, node_num, num_link_type_classes]
        link_type_probs = link_type_probs[link_label_mask==1]  # [edge_nums, num_link_type_classes]
        link_type_preds = link_type_probs.argmax(-1)  # [edge_nums]
        link_type_pred_loss, negative_loss = self.link_type_loss_fn(link_type_probs, y_linkType_label, label_mask, negative_mask, negative=True)

        """
            Loss:  三个任务之间的loss是相互间会受到影响的 
        """
        # loss = node_classification_loss + link_pred_loss + link_type_pred_loss + negative_loss * self.negative_loss_weight
        loss = link_pred_loss + link_type_pred_loss + negative_loss * self.negative_loss_weight
        # loss = node_classification_loss

        return loss, sent_class_prob, y_sentType_label, link_probs, ylabels, link_type_probs, y_linkType_label, link_tags

    def generate_graph_data(self, triplets, num_rels):
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
        data.edge_norm = self.edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)

        return data

    def edge_normalization(self, edge_type, edge_index, num_entity, num_relation):
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


class EdgeUpdateModel(nn.Module):
    def __init__(self, node_hidden_size, edge_hidden_size):
        super(EdgeUpdateModel, self).__init__()
        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size

        self.r = nn.Linear(2*node_hidden_size+edge_hidden_size, self.edge_hidden_size, True)
        self.z = nn.Linear(2*node_hidden_size+edge_hidden_size, self.edge_hidden_size, True)

        self.c = nn.Linear(2*node_hidden_size, self.edge_hidden_size, True)
        self.u = nn.Linear(edge_hidden_size, edge_hidden_size, True)

    def forward(self, nodes, const_path, struct_path, edge_label_mask):
        """
        :param nodes: [node_num, node_hidden_size]
        :param const_path: [node_num, node_num, edge_hidden_size]
        :param struct_path: [node_num, node_num, edge_hidden_size]
        :param edge_label_mask: [node_num, node_num]
        :return:
        """
        node_num = nodes.shape[0]
        edge_hidden_size = struct_path.shape[-1]
        nodes = nodes.unsqueeze(1).expand(-1, node_num, -1)  # [node_num, node_num, node_hidden_size]
        nodes = torch.cat((nodes, nodes.transpose(0, 1)), dim=-1)  # [node_num, node_num, 2*node_hidden_size]

        struct_path = struct_path * (edge_label_mask.unsqueeze(-1).expand(-1, -1, edge_hidden_size))  # [node_num, node_num, edge_hidden_size]

        rz_input = torch.cat((nodes, struct_path), -1)  # [node_num, node_num, 2*node_hidden_size+edge_hidden_size]
        r = torch.sigmoid(self.r(rz_input))  # [node_num, node_num, edge_hidden_size]
        z = torch.sigmoid(self.z(rz_input))  # [node_num, node_num, edge_hidden_size]

        u = torch.tanh(self.c(nodes) + r * self.u(struct_path))  # [node_num, node_num, edge_hidden_size]

        new_h = z * struct_path + (1-z) * u  # [node_num, node_num, edge_hidden_size]

        return new_h





