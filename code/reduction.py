import sys
sys.path.append('..')
import random
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader

from deepctr_torch.models.nfm import NFM
from deepctr_torch.models.fm import Factorization_Machine
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, build_input_features
from data_utils import get_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kuairand')
    parser.add_argument('--model', type=str, default='nfm')
    parser.add_argument('--alpha', type=float, default=0)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# -------------------------------------------------- #

args = parse_args()
linear_feature_columns, dnn_feature_columns, _, _, _, _, testx, testy, key2index = get_data(args.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
key = '_'.join([args.model, args.dataset])

if args.model == 'nfm':
    model = NFM(linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-3, seed=0, \
        bi_dropout=0.2, dnn_dropout=0.2, task='regression', device=device, use_bn=1)
elif args.model == 'fm':
    model = Factorization_Machine(linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-3, \
        seed=0, task='regression', device=device)
model.compile("adam", 'bce', metrics=['bce', 'auc'])
model.load_state_dict(torch.load(f"../model/{key}.pth.tar"))
setup_seed(0)

if args.dataset == 'ml-1m':
    if isinstance(testx, dict):
        testx = [testx[feature] for feature in model.feature_index]
    for i in range(len(testx)):
        if len(testx[i].shape) == 1:
            testx[i] = np.expand_dims(testx[i], axis=1)
    testx = np.concatenate(testx, axis=-1)
    testx, testy = torch.from_numpy(testx), torch.tensor(testy)

tensor_data = Data.TensorDataset(testx)
test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=256)
pred_list = []

if args.dataset == 'book':
    cate_indices, cate_arr = [2,3,4,5,6], np.arange(486+1)
elif args.dataset == 'ml-1m':
    cate_indices, cate_arr = [5,6,7,8,9,10], np.arange(18+1)
else:   # kuairand
    cate_indices, cate_arr = [2,3,4], np.arange(46+1)

for name, param in model.named_parameters():
    if args.alpha == 0 and \
         name in {'linear_model.embedding_dict.categories.weight', 'linear_model.embedding_dict.genres.weight', 'linear_model.embedding_dict.tags.weight'}:
        nn.init.zeros_(param)
        break
sup_arr = np.load(f'../data/{args.dataset}/posratio.npy')[1:]     # (num_category,)
if args.dataset == 'book':
    # For Book dataset, we use 50 most frequent categories in the training set to calculate REO@5 
    # for the total number of groups is too large and most categories appear rarely.
    num_arr = np.load(f'../data/{args.dataset}/train_cate_arr.npy'.format(args.dataset))[:50] - 1
else:
    num_arr = np.arange(len(sup_arr))
top_sup_arr = sup_arr[num_arr]

# -------------------------------------------------- #

def UAUC(ut_dict, pt_dict): 
    uauc = 0.0
    u_size = 0.0
    for k in ut_dict:
        if (1 in ut_dict[k]) and (0 in ut_dict[k]):
            uauc_one = roc_auc_score(np.array(ut_dict[k]),np.array(pt_dict[k]))
            uauc += uauc_one
            u_size += 1.0
    return uauc/ u_size

def calc_ndcg(gt,pred,k,logsum):
    idcg = logsum[min(k,sum(gt))-1]
    dcg = 0
    k = int(min(k,len(gt)))
    max_indices = np.argsort(pred)[::-1]
    for i,idx in enumerate(max_indices[:k]):
        dcg += gt[int(idx)]/np.log2(i+2)
    return dcg/idcg

def NDCG(ut_dict, up_dict, atk_list):
    ndcg_list = [0 for i in range(len(atk_list))]
    usize = 0
    logsum = 1 / np.log2(np.arange(atk_list[-1] + 2)[2:])
    logsum = np.cumsum(logsum)
    for u in list(ut_dict.keys()):
        if (1 in ut_dict[u]) and (0 in ut_dict[u]):
            usize += 1
            for idx,k in enumerate(atk_list):
                ndcg_atk = calc_ndcg(ut_dict[u],up_dict[u],k,logsum)
                ndcg_list[idx] += ndcg_atk
    return [sum_ndcg/usize for sum_ndcg in ndcg_list]
print('Preparation Done.')

# -------------------------------------------------- #

model.eval()
with torch.no_grad():
    for _, x_test in enumerate(test_loader):
        X_ = x_test[0].to(device).float()
        y_pred = model(X_)
        pred_list.append(y_pred.cpu().data.numpy())
pred_arr = np.concatenate(pred_list).astype("float64").squeeze()
gt_arr = testy.numpy().squeeze()
testx = testx.numpy()

gt_dict, pred_dict, test_idx_dict = {}, {}, {}
for i in range(len(testx)):
    pred, gt, user = pred_arr[i], gt_arr[i], testx[i,0]
    try:
        gt_dict[user].append(gt)
        pred_dict[user].append(pred)
        test_idx_dict[user].append(i)
    except:
        gt_dict[user] = [gt]
        pred_dict[user] = [pred]
        test_idx_dict[user] = [i]
        
for user in gt_dict:
    gt_dict[user] = np.array(gt_dict[user])
    pred_dict[user] = np.array(pred_dict[user])
    test_idx_dict[user] = np.array(test_idx_dict[user])

test_cate_gt_dict, test_cate_pred_dict, test_cate_ratio_dict = {}, {}, {}
for user in pred_dict.keys():
    user_pos = int(sum(gt_dict[user]))
    if user_pos == 0 or user_pos == len(gt_dict[user]):
        continue
    rec_indices = test_idx_dict[user][np.argsort(pred_dict[user])[::-1][:user_pos]]
    gt_indices = test_idx_dict[user][gt_dict[user]==1]
    assert len(rec_indices) == len(gt_indices)

    for idx in gt_indices:
        for i in cate_indices:
            cate = int(testx[idx,i])
            if cate == 0:
                break
            try:
                test_cate_gt_dict[cate] += 1
            except:
                test_cate_gt_dict[cate] = 1
                
    for idx in rec_indices:
        for i in cate_indices:
            cate = int(testx[idx,i])
            if cate == 0:
                break
            try:
                test_cate_pred_dict[cate] += 1
            except:
                test_cate_pred_dict[cate] = 1

for cate in cate_arr:
    try:
        test_cate_ratio_dict[cate] = test_cate_pred_dict[cate] / test_cate_gt_dict[cate]
    except:
        test_cate_ratio_dict[cate] = 0
rec_ratio_arr = np.zeros(len(cate_arr))
for i,cate in enumerate(cate_arr):
    try:
        rec_ratio_arr[i] = test_cate_ratio_dict[cate]
    except:
        rec_ratio_arr[i] = 0
rec_ratio_arr = rec_ratio_arr[1:]
rec_ratio_arr = rec_ratio_arr[num_arr]
print('Construct Dict Done.')

# -------------------------------------------------- #

test_cate_top5_dict = {}
for user in pred_dict.keys():
    test_idx_dict[user], gt_dict[user] = np.array(test_idx_dict[user]), np.array(gt_dict[user])
    user_pos = min(int(sum(gt_dict[user])), 5)
    if user_pos == 0 or int(sum(gt_dict[user])) == len(gt_dict[user]):
        continue
    rec_indices_top5 = test_idx_dict[user][np.argsort(pred_dict[user])[::-1][:user_pos]]
            
    for idx in rec_indices_top5:
        for j in cate_indices:
            cate, gt = int(testx[idx,j]), int(gt_arr[idx])
            if cate == 0 or gt == 0:
                break
            try:
                test_cate_top5_dict[cate] += 1
            except:
                test_cate_top5_dict[cate] = 1

rec_top5_arr = np.zeros(len(sup_arr))
for j,cate in enumerate(np.arange(1,1+len(sup_arr))):
    try:
        rec_top5_arr[j] = test_cate_top5_dict[cate] / test_cate_gt_dict[cate]
    except:
        rec_top5_arr[j] = 0
rec_top5_arr = rec_top5_arr[num_arr]
reo5 = np.std(rec_top5_arr) / np.mean(rec_top5_arr)

uauc, ndcg = UAUC(gt_dict, pred_dict), NDCG(gt_dict, pred_dict, [5])[0]
print(uauc, ndcg, reo5)