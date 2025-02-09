import argparse
import logging
import numpy as np
from tqdm import tqdm
import random
import os
import warnings
warnings.filterwarnings('ignore')  # 屏蔽warning


from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader

from graph_utils import prepare_data
from model import MultiGrainedAndSynticEnhancedModel

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


from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)

from utils import json_util


def accuracy_metric(preds, labels, pos_prob, isbaseline=False):
    
    assert len(preds) == len(labels)
    lr_precision, lr_recall = 0, 0
    if not isbaseline:
        lr_precision, lr_recall, _ = precision_recall_curve(labels, pos_prob)
        
    out = {'accuracy': accuracy_score(preds, labels),
           'precision': precision_score(labels, preds),
           'recall': recall_score(labels, preds),
           'f1': f1_score(labels, preds),
           'classification_report':classification_report(labels, preds),
           'roc_auc_score':roc_auc_score(labels,pos_prob)  if not isbaseline else 0,
           "cohen":cohen_kappa_score(labels,preds),
           "pr_auc":auc(lr_recall, lr_precision) if not isbaseline else 0
          }
    return out


def precision_recall_f1_metric(preds, labels, average_type="micro"):
    assert len(preds) == len(labels)

    out = {'accuracy': accuracy_score(preds, labels),
           'precision': precision_score(labels, preds, average=average_type),
           'recall': recall_score(labels, preds, average=average_type),
           'f1': f1_score(labels, preds, average=average_type)
           }
    report = classification_report(labels, preds)

    return out, report


# 第二种
def bi_multi_metric(bi_preds, bi_labels, multi_preds, multi_labels):
    """
       bi_preds = [0, 1, 1, 0]
       bi_labels = [1, 1, 1, 0]
       multi_preds =[1, 2, 3, 0]
       multi_labels = [1, 2, 4, 0]
    """
    # return  <跟你那边两个baseline的评价指标>
    cnt_cor_bi=sum([x*y for x,y in zip(bi_preds,bi_labels)])
    cnt_pred=sum(bi_preds)
    cnt_golden=sum(bi_labels)
    if cnt_pred!=0:
        prec_bi = cnt_cor_bi * 1. / cnt_pred
    else:
        prec_bi=0

    if cnt_golden!=0:
        recall_bi=cnt_cor_bi * 1. / cnt_golden
    else:
        recall_bi=0
    if prec_bi == 0 or recall_bi == 0:
        f1_bi = 0
    else:
        f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)

    bi_multi_preds= [x*y for x,y in zip(bi_preds,multi_preds)]
    bi_multi_labels= [x*y for x,y in zip(bi_labels,multi_labels)]
    bi_multi_res= [1 if bi_multi_preds[i]>0 and bi_multi_preds[i]==bi_multi_labels[i] else 0 for i in range(len(bi_multi_preds))]
    cnt_cor_multi=sum(bi_multi_res)
    cnt_pred,cnt_golden=0,0
    for i in range(len(bi_multi_preds)):
        if bi_preds[i]==1 and multi_preds[i]>0:
            cnt_pred+=1

    for i in range(len(bi_multi_labels)):
        if bi_multi_labels[i]!=0:
            cnt_golden+=1

    if cnt_pred!=0:
        prec_multi = cnt_cor_multi * 1. / cnt_pred
    else:
        prec_multi=0

    if cnt_golden!=0:
        recall_multi=cnt_cor_multi * 1. / cnt_golden
    else:
        recall_multi=0
    if prec_multi == 0 or recall_multi ==0:
        f1_multi = 0
    else:
        f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    # prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
    # f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    # prec_multi, recall_multi = cnt_cor_multi * 1. / cnt_pred, cnt_cor_multi * 1. / cnt_golden
    # f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return f1_bi, f1_multi, prec_bi,recall_bi,prec_multi,recall_multi


# 不依赖于link predition
# def direct_precision_recall_f1_metric(multi_preds,multi_labels, type2id):
#     """
#        bi_preds = [0, 1, 1, 0]
#        bi_labels = [1, 1, 1, 0]
#        multi_preds =[1, 2, 3, 0]
#        multi_labels = [1, 2, 4, 0]
#     """
#     id2type = {value:key for key, value in type2id.items()}
#     # return  <跟你那边两个baseline的评价指标>
#     bi_preds=[1 if x>0 else 0 for x in multi_preds ]
#     bi_labels=[1 if x>0 else 0 for x in multi_labels]
#     # bi_preds,bi_labels=multi_preds&,multi_labels
#     cnt_cor_bi=sum([x*y for x,y in zip(bi_preds,bi_labels)])
#     cnt_pred=sum(bi_preds)
#     cnt_golden=sum(bi_labels)
#     if cnt_pred!=0:
#         prec_bi = cnt_cor_bi * 1. / cnt_pred
#     else:
#         prec_bi=0
#
#     if cnt_golden!=0:
#         recall_bi=cnt_cor_bi * 1. / cnt_golden
#     else:
#         recall_bi=0
#     if prec_bi!=0 and recall_bi!=0:
#         f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
#     else:
#         f1_bi=0
#     bi_multi_preds=[x*y for x,y in zip(bi_preds,multi_preds)]
#     bi_multi_labels=[x*y for x,y in zip(bi_labels,multi_labels)]
#     bi_multi_res=[1 if bi_multi_preds[i]>0 and bi_multi_preds[i]==bi_multi_labels[i] else 0 for i in range(len(bi_multi_preds)) ]
#     cnt_cor_multi=sum(bi_multi_res)
#     cnt_pred,cnt_golden=0,0
#     for i in range(len(bi_multi_preds)):
#         if bi_preds[i]==1 and multi_preds[i]>0:
#             cnt_pred+=1
#
#     for i in range(len(bi_multi_labels)):
#         if bi_multi_labels[i]!=0:
#             cnt_golden+=1
#
#     if cnt_pred!=0:
#         prec_multi = cnt_cor_multi * 1. / cnt_pred
#     else:
#         prec_multi=0
#
#     if cnt_golden!=0:
#         recall_multi=cnt_cor_multi * 1. / cnt_golden
#     else:
#         recall_multi=0
#     if prec_multi!=0 and recall_multi!=0:
#         f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
#     else:
#         f1_multi=0
#
#     precision_rate,recall_rate,f1_score=detail_precision_recall_f1_metric(bi_multi_preds,bi_multi_labels,type2id)
#
#     # return f1_bi, f1_multi,prec_bi,recall_bi,prec_multi,recall_multi
#     # result=[]
#     # result.append([prec_bi,recall_bi,f1_bi])
#     # result.append([prec_multi,recall_multi,f1_multi])
#     #
#     # for i in range(len(precision_rate)):
#     #     result.append([precision_rate[i],recall_rate[i],f1_score[i]])
#
#     result = {}
#     for i in range(len(precision_rate)):
#         result[id2type[i]] = {"P": precision_rate[i-1], "R": recall_rate[i-1], "F1": f1_score[i-1]}
#
#     return result

# 依赖于link prediction
def simul_precision_recall_f1_metric(bi_preds, bi_labels, multi_preds, multi_labels, type2id):
    """
       bi_preds = [0, 1, 1, 0]
       bi_labels = [1, 1, 1, 0]
       multi_preds =[1, 2, 3, 0]
       multi_labels = [1, 2, 4, 0]
    """
    id2type = {value:key for key, value in type2id.items()}

    # return  <跟你那边两个baseline的评价指标>
    cnt_cor_bi = sum([x * y for x, y in zip(bi_preds, bi_labels)])
    cnt_pred = sum(bi_preds)
    cnt_golden = sum(bi_labels)
    if cnt_pred != 0:
        prec_bi = cnt_cor_bi * 1. / cnt_pred
    else:
        prec_bi = 0

    if cnt_golden != 0:
        recall_bi = cnt_cor_bi * 1. / cnt_golden
    else:
        recall_bi = 0
    if prec_bi != 0 and recall_bi != 0:
        f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    else:
        f1_bi = 0
    bi_multi_preds = [x * y for x, y in zip(bi_preds, multi_preds)]
    bi_multi_labels = [x * y for x, y in zip(bi_labels, multi_labels)]
    bi_multi_res = [1 if bi_multi_preds[i] > 0 and bi_multi_preds[i] == bi_multi_labels[i] else 0 for i in
                    range(len(bi_multi_preds))]
    cnt_cor_multi = sum(bi_multi_res)
    cnt_pred, cnt_golden = 0, 0
    for i in range(len(bi_multi_preds)):
        if bi_preds[i] == 1 and multi_preds[i] > 0:
            cnt_pred += 1

    for i in range(len(bi_multi_labels)):
        if bi_multi_labels[i] != 0:
            cnt_golden += 1

    if cnt_pred != 0:
        prec_multi = cnt_cor_multi * 1. / cnt_pred
    else:
        prec_multi = 0

    if cnt_golden != 0:
        recall_multi = cnt_cor_multi * 1. / cnt_golden
    else:
        recall_multi = 0
    if prec_multi != 0 and recall_multi != 0:
        f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    else:
        f1_multi = 0

    precision_rate, recall_rate, f1_score = detail_precision_recall_f1_metric(bi_multi_preds, bi_multi_labels, type2id)

    # return f1_bi, f1_multi,prec_bi,recall_bi,prec_multi,recall_multi
    result = []
    result.append([prec_bi, recall_bi, f1_bi])
    result.append([prec_multi, recall_multi, f1_multi])

    for i in range(len(precision_rate)):
        result.append([precision_rate[i], recall_rate[i], f1_score[i]])

    result = {}
    for i in range(len(precision_rate)):
        # result[id2type[i]] = {"P": precision_rate[i-1], "R": recall_rate[i-1], "F1": f1_score[i-1]}
        result[id2type[i]] = {"P": precision_rate[i], "R": recall_rate[i], "F1": f1_score[i]}

    return result

def detail_precision_recall_f1_metric(bi_multi_preds, bi_multi_labels, type2id):
    '''
    multi_preds =[1, 2, 3, 0]
    multi_labels = [1, 0, 2, 0]
    '''

    # prediction_type = ["none","next_action", "sub_action", "supplement"]
    id2type = {}
    for key in type2id:
        id2type[type2id[key]] = key
    # print(id2type)
    TP_prediction = [0] * len(type2id)
    FN_prediction = [0] * len(type2id)
    FP_prediction = [0] * len(type2id)

    for i in range(len(bi_multi_preds)):
        if bi_multi_preds[i] == bi_multi_labels[i]:
            TP_prediction[bi_multi_preds[i]] += 1
        else:
            FP_prediction[bi_multi_preds[i]] += 1
    for i in range(len(bi_multi_labels)):
        if bi_multi_labels[i] != bi_multi_preds[i]:
            FN_prediction[bi_multi_labels[i]] += 1

    recall_rate = []
    precision_rate = []
    f1_score = []

    for i in range(len(TP_prediction)):
        sum = TP_prediction[i] + FN_prediction[i]
        if sum != 0:
            recall_rate.append(TP_prediction[i] / sum)
        else:
            recall_rate.append(0)

        sum = TP_prediction[i] + FP_prediction[i]
        if sum != 0:
            precision_rate.append(TP_prediction[i] / sum)
        else:
            precision_rate.append(0)

        sum = recall_rate[i] + precision_rate[i]
        if sum != 0:
            f1_score.append(2 * recall_rate[i] * precision_rate[i] / (sum))
        else:
            f1_score.append(0)

    # for i in range(len(type2id)):
    #     # print(i)
    #     print(id2type[i],
    #           ":\nprecision rate:", precision_rate[i],
    #           "\nrecall rate:", recall_rate[i],
    #           "\nf1 score:", f1_score[i], "\n\n")
    # print(bi_multi_preds, bi_multi_labels)
    # print(TP_prediction,FN_prediction,FP_prediction)
    # print(prediction_type,precision_rate,recall_rate,f1_score)
    # print(precision_rate, recall_rate, f1_score)
    return precision_rate, recall_rate, f1_score


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

        
logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def evaluate(model, data_loader, args, mode):
    
    total_loss = 0
    total_accuracy = 0

    # 存储所有的文本
    total_samples = []
    # senType classification
    total_senType_preds = []
    total_senType_labels = []

    # link prediction
    total_link_preds = []
    total_y_link_labels = []
    # link type classifcation
    total_link_type_preds = []
    total_y_link_type_labels = []

    total_steps = 0
    
    print("***** Running evaluation {} ***** %s", mode)
    print("  Num examples = %d", len(data_loader))
    print("  Batch size = %d", args.eval_batch_size)
    
    for step, batch in enumerate(data_loader):
        model.eval()
        batch_data = batch       # No need for sending data to device
        with torch.no_grad():
            loss, sent_class_prob, y_sentType_label, link_probs, y_link_labels, link_type_probs, y_linkType_label, link_tag = model(batch_data)

            if args.n_gpu > 1:
                loss = loss.mean()
            # senType prediction
            senType_pred = sent_class_prob.argmax(1).cpu().numpy().tolist()
            # link prediction
            link_pred = threshold_binary_pred(link_probs, args.threshold, link_tag)
            link_pred = list(link_pred)
            # link_pred = link_probs.argmax(1).cpu().numpy().tolist()
            # link type prediction
            link_type_pred = link_type_probs.argmax(-1).cpu().numpy().tolist()

            total_loss += loss.item()

        for sam_list in batch_data:
            total_samples.extend(sam_list.txt_tokens)

        total_steps += 1

        # senType classification
        total_senType_preds += senType_pred
        total_senType_labels += y_sentType_label.detach().cpu().numpy().tolist()
        total_link_preds += link_pred
        total_y_link_labels += y_link_labels.detach().cpu().numpy().tolist()
        total_link_type_preds += link_type_pred
        total_y_link_type_labels += y_linkType_label.detach().cpu().numpy().tolist()

    link_f1, link_type_f1, link_prec, link_recall, link_type_prec, link_type_recall = bi_multi_metric(total_link_preds, total_y_link_labels, total_link_type_preds, total_y_link_type_labels)
    senType_metric, senType_metric_report = precision_recall_f1_metric(total_senType_preds, total_senType_labels, average_type="micro")

    avg_loss = total_loss/total_steps

    type2id = json_util.load("./data/wikiHow/linkType2id.json")
    link_type_result = simul_precision_recall_f1_metric(total_link_preds, total_y_link_labels, total_link_type_preds, total_y_link_type_labels, type2id)

    return senType_metric, senType_metric_report, link_f1, link_type_f1, link_prec, link_recall, link_type_prec, link_type_recall, avg_loss, link_type_result

def search_idx(arr, search_value):
    match_idx_arr = []
    for idx, value in enumerate(arr):
        if value == search_value:
            match_idx_arr.append(idx)
    start_idx = match_idx_arr[0]
    end_idx = match_idx_arr[-1]

    return start_idx, end_idx+1

def threshold_binary_pred(link_probs, threshold, link_tags):

    pred_labels = link_probs.argmax(-1).detach().cpu().numpy().tolist()

    # link_probs = link_probs[:,1].detach().cpu().numpy().tolist()
    # link_tags = link_tags.detach().cpu().numpy().tolist()
    # pred_labels = []
    #
    # for v in range(link_tags[-1]):
    #     if v not in link_tags:
    #         continue
    #     start_idx, end_idx = search_idx(link_tags, v)
    #     if (end_idx - start_idx) > 1:
    #         prob_block = link_probs[start_idx:end_idx]
    #         sorted_id = sorted(range(len(prob_block)), key=lambda k: prob_block[k], reverse=True)
    #         corrected_preds = [0] * len(prob_block)
    #         corrected_preds[sorted_id[0]] = 1
    #         pred_labels.extend(corrected_preds)
    #     else:
    #         pred_labels.extend([1])

    # for v in range(link_tags[-1]):
    #     if v not in link_tags:
    #         continue
    #     start_idx, end_idx = search_idx(link_tags, v)
    #     if (end_idx - start_idx) > 2:
    #         pred_block = link_preds[start_idx:end_idx]
    #         prob_block = link_probs[start_idx:end_idx]
    #         if np.array(pred_block).sum() > 2:
    #             sorted_id = sorted(range(len(prob_block)), key=lambda k: prob_block[k], reverse=True)
    #             corrected_preds = [0] * len(prob_block)
    #             corrected_preds[sorted_id[0]] = 1
    #             corrected_preds[sorted_id[1]] = 1
    #             pred_labels.extend(corrected_preds)
    #         else:
    #             pred_labels.extend(pred_block)
    #     else:
    #         pred_labels.extend(link_preds[start_idx:end_idx])

    pred_labels = np.array(pred_labels).astype(dtype=int)

    return pred_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datafile", default="data/wikiHow/", type=str)
    parser.add_argument("--domain", default="wikiHow", type=str)
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--model", default="bert-base-uncased", type=str, help="google/electra-small-discriminator, bert-base-uncased")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--max_seq_len", default=80, type=int)   # 此参数决定了构建语法树的大小
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_node_classes", default=4, type=int, help="i.e., 4 classes: none, declaration, action, both")
    parser.add_argument("--num_link_classes", default=4, type=int, help="i.e., 4 classes: none, next_action, sub_action, supplement")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--output_dir", default="", type=str, help="Path where model needs to be saved.")
    # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")
    # parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to run training.")
    parser.add_argument("--custom_model", default="multi_grained_syntatic_enhanced", type=str)
    parser.add_argument("--window_size", default=10, type=int, help="0-Any pairs of nodes, n-edges between nodes no more than n distances apart")
    parser.add_argument("--do_oversample_data", action="store_true", help="Inflate Training positive labels")
    parser.add_argument("--use_pretrained_weights", action="store_true", help="Whether to load pre-trained weights")
    parser.add_argument("--gnn_layer1_out_dim", default=128, type=int, help="The dimension of Graph Neural Network Layer 1 output dimension")
    parser.add_argument("--gnn_layer2_out_dim", default=64, type=int, help="The dimension of Graph Neural Network Layer 2 output dimension")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout for each layers.")
    parser.add_argument("--graph_connection_type", default="complete", type=str, help="Type of graph connection you need to learn features.")
    # syn_knowledge_rgcn
    parser.add_argument("--rgcn_n_bases", type=int, default=6, help="the number of the gcn layer")
    parser.add_argument("--rgcn_dropout", type=float, default=0.0, help="dropout rate for the rgcn encoder")
    # parser.add_argument("--entity_embed_size", type=int, default=100, help="the size of the syn entity embedding")
    # pos embedding
    parser.add_argument("--pos_embedding_size",  default=128, type=int, help="the size of the pos embedding ")
    parser.add_argument("--dist_embeding_size",  default=50, type=int, help="the size of the pos embedding ")

    # structure-awre attention
    parser.add_argument("--head_num", type=int, default=4, help="the head num of the structure-awre attention module")
    parser.add_argument("--att_hidden_size", type=int, default=256, help="the hidden size of the structure-aware attention module")
    parser.add_argument("--att_dropout", type=float, default=0.0, help="the dropout rate of the structure-aware attention layer")
    parser.add_argument("--layer_num", type=int, default=3, help="the number layer of the attention layer")

    parser.add_argument("--threshold", type=float, default=0.2, help="the threshold value of the binary classification")
    parser.add_argument("--negative_loss_weight", type=float, default=0.2, help="the negative sample")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    
    # set_seed(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)

    print("=======================> ")
    print("This is threshold is {}".format(args.threshold))
    print("=======================> ")

    data_train, data_val, data_test, edge_percent, max_node_num = prepare_data(args.datafile, args.domain, args.max_seq_len, tokenizer,
                                                                 args.window_size, args.do_oversample_data,
                                                                 args.graph_connection_type, args.model)
    syn_rel2id = json_util.load(os.path.join(args.datafile, 'syn_rel2id.json'))
    senType2id = json_util.load(os.path.join(
        args.datafile, 'link2id.json'))
    pos2id = json_util.load(os.path.join(args.datafile, 'pos2id.json'))
    args.edge_percent = edge_percent
    args.max_node_num = max_node_num
    args.num_syn_rel = len(syn_rel2id)
    args.num_pos_type = len(pos2id)

    model = ""
    if args.custom_model == "multi_grained_syntatic_enhanced":
        model = MultiGrainedAndSynticEnhancedModel(config, args).to(device)
    else:
        logger.info("Invalid Custom Model name")
        exit(0)
    
    model = DataParallel(model)

    train_loader = DataListLoader(data_train, batch_size=args.batch_size, shuffle=False)
    val_loader = DataListLoader(data_val, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataListLoader(data_test, batch_size=args.eval_batch_size, shuffle=False)

    # Prepare optimizer and schedule (decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{ "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                      "weight_decay": args.weight_decay,},
                                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
                                    ]
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    print("num_training_steps:",len(train_loader)*args.num_epochs)
    # Scheduler causing the loss to hover around same value
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_loader)*args.num_epochs)
    
    # Start Training

    if args.do_train:
        print("start training")
        best_loss = 1000
        best_epoch = -1
        best_accuracy = 0
        best_senType_f1 = 0
        best_link_f1 = 0
        best_link_type_f1 = 0

        model.zero_grad()
        for epoch in tqdm(range(args.num_epochs)):
            global_loss = 0
            # senType Classfication
            epoch_senType_preds = []
            epoch_y_senType_labels = []
            # Link Prediction
            epoch_link_preds = []
            epoch_y_link_labels = []
            # Link Type Classification
            epoch_link_type_preds = []
            epoch_y_link_type_labels = []

            global_step = 0
            for step, batch in enumerate(train_loader):

                model.train()
                batch_data = batch
                optimizer.zero_grad()

                # This values is a aggregated n losses,probs,ylabels from n gpus aggregated in GPU 0
                loss, sentType_probs, y_sentType_labels, link_probs, y_link_labels, link_type_probs, y_linkType_label, link_tag = model(batch_data)

                if args.n_gpu > 1:
                    loss = loss.mean()

                # Predict Result
                sentType_preds = sentType_probs.argmax(-1)
                # link_preds= thredhod
                link_preds = threshold_binary_pred(link_probs, args.threshold, link_tag)
                # link_preds = link_probs.argmax(-1)
                link_type_preds = link_type_probs.argmax(1)

                global_loss += loss.item()
                global_step += 1

                # Calculate predictions
                if len(epoch_link_preds)== 0:
                    epoch_senType_preds = sentType_preds.detach().cpu().numpy()
                    epoch_y_senType_labels = y_sentType_labels.detach().cpu().numpy()
                    epoch_link_preds = link_preds#.detach().cpu().numpy()
                    epoch_y_link_labels = y_link_labels.detach().cpu().numpy()
                    epoch_link_type_preds = link_type_preds.detach().cpu().numpy()
                    epoch_y_link_type_labels = y_linkType_label.detach().cpu().numpy()
                else:
                    # senType classification
                    epoch_senType_preds = np.append(epoch_senType_preds, sentType_preds.detach().cpu().numpy(), axis=0)
                    epoch_y_senType_labels = np.append(epoch_y_senType_labels, y_sentType_labels.detach().cpu().numpy(), axis=0)
                    # link Prediction
                    epoch_link_preds = np.append(epoch_link_preds, link_preds, axis=0)
                    epoch_y_link_labels = np.append(epoch_y_link_labels, y_link_labels.detach().cpu().numpy(), axis=0)
                    # link type classification
                    epoch_link_type_preds = np.append(epoch_link_type_preds, link_type_preds.detach().cpu().numpy(), axis=0)
                    epoch_y_link_type_labels = np.append(epoch_y_link_type_labels, y_linkType_label.detach().cpu().numpy(), axis=0)

                # Back Propagate
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

            link_pred_f1_train, link_type_pred_f1_train, link_pred_prec_train, link_pred_recall_train, link_type_pred_prec_train, link_type_pred_recall_train = bi_multi_metric(epoch_link_preds, epoch_y_link_labels, epoch_link_type_preds, epoch_y_link_type_labels)
            train_avg_loss = global_loss / global_step
            senType_metric_val, senType_metric_report_val, link_f1_val, link_type_f1_val, link_prec_val, link_recall_val, link_type_prec_val, link_type_recall_val, avg_loss_val, link_type_result = evaluate(model, val_loader, args, "val")

            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            print("Training Epoch: %d, Loss: %f", epoch, train_avg_loss)
            print("Validation Epoch: %d, Loss: %f", epoch, avg_loss_val)

            if senType_metric_val['f1'] > best_senType_f1:
                best_senType_f1 = senType_metric_val['f1']
                val_loss = avg_loss_val
                # if not os.path.exists(os.path.join(args.output_dir, "senType")):
                #     os.makedirs(os.path.join(args.output_dir, "senType"))

                print("===================senType========================")
                print(senType_metric_report_val)
                print("Best Performances in senType: {}, Loss: {}, #SenType: |-> F1: {} #LinkPred: P: {}, R: {}, |-> "
                      "F1: {}; #LinkType: P: {}, R: {}, |-> F1: {}".format(epoch, val_loss, senType_metric_val['f1'],
                                                                           link_prec_val, link_recall_val, link_f1_val,
                                                                           link_type_prec_val, link_type_recall_val,
                                                                           link_type_f1_val))
                print("===================senType========================")
                model_to_save = os.path.join(args.output_dir, "senType", "best_model.ckpt")

            if link_f1_val > best_link_f1:
                best_link_f1 = link_f1_val
                val_loss = avg_loss_val
                # if not os.path.exists(os.path.join(args.output_dir, "senType")):
                #     os.makedirs(os.path.join(args.output_dir, "senType"))

                print("===================Link========================")
                print("Best Performances in senType: {}, Loss: {}, #SenType: |-> F1: {} #LinkPred: P: {}, R: {}, |-> "
                      "F1: {}; #LinkType: P: {}, R: {}, |-> F1: {}".format(epoch, val_loss, senType_metric_val['f1'],
                                                                           link_prec_val, link_recall_val, link_f1_val,
                                                                           link_type_prec_val, link_type_recall_val,
                                                                           link_type_f1_val))
                print("Fine_grained: {}".format(link_type_result))
                print("===================Link========================")
                model_to_save = os.path.join(args.output_dir, "senType", "best_model.ckpt")

            if link_type_f1_val > best_link_type_f1:
                best_link_type_f1 = link_type_f1_val
                val_loss = avg_loss_val
                # if not os.path.exists(os.path.join(args.output_dir, "senType")):
                #     os.makedirs(os.path.join(args.output_dir, "senType"))

                print("===================LinkType========================")
                print("Best Performances in senType: {}, Loss: {}, #SenType: |-> F1: {} #LinkPred: P: {}, R: {}, |-> "
                      "F1: {}; #LinkType: P: {}, R: {}, |-> F1: {}".format(epoch, val_loss, senType_metric_val['f1'],
                                                                           link_prec_val, link_recall_val, link_f1_val,
                                                                           link_type_prec_val, link_type_recall_val,
                                                                           link_type_f1_val))
                print("Fine_grained: {}".format(link_type_result))
                print("===================LinkType========================")
                model_to_save = os.path.join(args.output_dir, "senType", "best_model.ckpt")

    # if args.do_eval:
    #     """
    #         SenType Classification Task
    #     """
    #     model_to_load = os.path.join(args.output_dir, "senType", "best_model.ckpt")
    #     checkpoint = torch.load(model_to_load)
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #
    #     ### Evaluate on val data
    #     senType_metric, senType_metric_report, avg_loss = evaluate(model, test_loader, args, "test")
    #
    #     logger.info("=================Test SenType==========================")
    #     # logger.info("Test Performance: Loss: %f, senTypeCL: F1: %f | linkPrecition: F1: %f | linkTypeCL: F1: %f",
    #     #             avg_loss,  senType_metric["f1"], link_metric["f1"], linkType_metric["f1"])
    #     logger.info("Test Performance: senTypeCL: F1: %f", senType_metric["f1"])
    #     logger.info(senType_metric_report)
