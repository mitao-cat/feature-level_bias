import sys
sys.path.append('..')
import random
import argparse

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader

from deepctr_torch.models.nfm import NFM
from deepctr_torch.models.fm import Factorization_Machine
from data_utils import get_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kuairand')
    parser.add_argument('--model', type=str, default='nfm')
    parser.add_argument('--reconstruction', type=int, default=1)
    # parser.add_argument('--beta', type=float, default=20)
    # parser.add_argument('--gamma', type=float, default=4)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# -------------------------------------------------- #

args = parse_args()
assert args.dataset == 'kuairand'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
key = '_'.join([args.model, args.dataset])
linear_feature_columns, dnn_feature_columns, _, train_loader, _, _, _, _, _ = get_data(args.dataset)

if args.model == 'nfm':
    model = NFM(linear_feature_columns, dnn_feature_columns, seed=0, bi_dropout=0.5, dnn_dropout=0.3, task='regression', device=device, use_bn=1)
elif args.model == 'fm':
    model = Factorization_Machine(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
model.compile("adam", 'bce', metrics=['bce', 'auc'])
model.load_state_dict(torch.load(f"../model/{key}.pth.tar"))
setup_seed(0)

for name, param in model.named_parameters():
    if name in {'linear_model.embedding_dict.categories.weight', 'linear_model.embedding_dict.genres.weight', 'linear_model.embedding_dict.tags.weight'}:
        linear_weights = list(param.data.cpu().numpy().squeeze()[1:])   # len: 486/18/46
        break
linear_weights = np.array(linear_weights)
train_sup_arr = np.load(f'../data/{args.dataset}/posratio.npy')[1:]

linregress = stats.linregress(train_sup_arr, linear_weights)
lw_sup_slope = linregress.slope
avg_lw, avg_sup = np.mean(linear_weights), np.mean(train_sup_arr)

# -------------------------------------------------- #

# debiased test for kuairand
args.dataset = 'KuaiRand_DT'
_, _, _, _, valx, valy, _, testx, testy, _ = get_data(args.dataset)
sampled_indices = np.array(random.sample(range(len(valy)),10000))
valx, valy = valx[sampled_indices], valy[sampled_indices]

val_gt_arr, valx = valy.numpy().squeeze(), valx.numpy()
maxlen, cate_indices = 3, [2,3,4]
val_pos_arr, val_neg_arr, val_sup_arr = np.zeros(len(train_sup_arr)+1), np.zeros(len(train_sup_arr)+1), np.zeros(len(train_sup_arr)+1)

for i in range(len(valx)):
    gt, user = val_gt_arr[i], valx[i,0]
    for j in cate_indices:
        cate = int(valx[i,j])
        if cate == 0:
            break
        if gt == 1:
            val_pos_arr[cate] += 1
        elif gt == 0:
            val_neg_arr[cate] += 1

for cate in np.arange(len(val_sup_arr)):
    if val_pos_arr[cate] + val_neg_arr[cate] != 0:
        val_sup_arr[cate] = val_pos_arr[cate] / (val_pos_arr[cate] + val_neg_arr[cate])
lw_list, train_sup_list, val_sup_list = [0] + list(linear_weights), [0] + list(train_sup_arr), list(val_sup_arr)

valx = torch.from_numpy(valx)
tensor_data = Data.TensorDataset(testx)
test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=256)
tensor_data = Data.TensorDataset(valx)
val_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=256)
test_pred_list, val_pred_list = [], []

# -------------------------------------------------- #

model.eval()
with torch.no_grad():
    for _, x_test in enumerate(test_loader):
        X_ = x_test[0].to(device).float()
        y_pred = model(X_)
        test_pred_list.append(y_pred.cpu().data.numpy())        
test_pred_arr = np.concatenate(test_pred_list).astype("float64").squeeze()
test_gt_arr = testy.numpy().squeeze()
testx = testx.numpy()

test_gt_dict, test_pred_dict, test_idx_dict, test_res_dict, test_bias_dict, test_sup_dict = {}, {}, {}, {}, {}, {}
for i in range(len(testx)):
    pred, gt, user = test_pred_arr[i], test_gt_arr[i], testx[i,0]
    res_temp, bias_temp, sup_temp, count = 0, 0, 0, 0
    for j in cate_indices:
        cate = int(testx[i,j])
        if cate == 0:
            break
        bias_temp += lw_list[cate]
        res_temp += (lw_list[cate] - train_sup_list[cate]*lw_sup_slope)
        sup_temp += val_sup_list[cate]
        count += 1
    bias_temp, res_temp, sup_temp = bias_temp / count, res_temp / count, sup_temp / count
    try:
        test_gt_dict[user].append(gt)
        test_pred_dict[user].append(pred)
        test_idx_dict[user].append(i)
        test_res_dict[user].append(res_temp)
        test_bias_dict[user].append(bias_temp)
        test_sup_dict[user].append(sup_temp)
    except:
        test_gt_dict[user] = [gt]
        test_pred_dict[user] = [pred]
        test_idx_dict[user] = [i]
        test_res_dict[user] = [res_temp]
        test_bias_dict[user] = [bias_temp]
        test_sup_dict[user] = [sup_temp]
        
for user in test_gt_dict:
    test_gt_dict[user] = np.array(test_gt_dict[user])
    test_pred_dict[user] = np.array(test_pred_dict[user])
    test_idx_dict[user] = np.array(test_idx_dict[user])
    test_res_dict[user] = np.array(test_res_dict[user])
    test_bias_dict[user] = np.array(test_bias_dict[user])
    test_sup_dict[user] = np.array(test_sup_dict[user])

# -------------------------------------------------- #

model.eval()
with torch.no_grad():
    for _, x_val in enumerate(val_loader):
        X_ = x_val[0].to(device).float()
        y_pred = model(X_)
        val_pred_list.append(y_pred.cpu().data.numpy())        
val_pred_arr = np.concatenate(val_pred_list).astype("float64").squeeze()
val_gt_arr = valy.numpy().squeeze()
valx = valx.numpy()

val_gt_dict, val_pred_dict, val_idx_dict, val_res_dict, val_bias_dict, val_sup_dict = {}, {}, {}, {}, {}, {}
for i in range(len(valx)):
    pred, gt, user = val_pred_arr[i], val_gt_arr[i], valx[i,0]
    res_temp, bias_temp, sup_temp, count = 0, 0, 0, 0
    for j in cate_indices:
        cate = int(valx[i,j])
        if cate == 0:
            break
        bias_temp += lw_list[cate]
        res_temp += (lw_list[cate] - train_sup_list[cate]*lw_sup_slope)
        sup_temp += val_sup_list[cate]
        count += 1
    bias_temp, res_temp, sup_temp = bias_temp / count, res_temp / count, sup_temp / count

    try:
        val_gt_dict[user].append(gt)
        val_pred_dict[user].append(pred)
        val_idx_dict[user].append(i)
        val_res_dict[user].append(res_temp)
        val_bias_dict[user].append(bias_temp)
        val_sup_dict[user].append(sup_temp)
    except:
        val_gt_dict[user] = [gt]
        val_pred_dict[user] = [pred]
        val_idx_dict[user] = [i]
        val_res_dict[user] = [res_temp]
        val_bias_dict[user] = [bias_temp]
        val_sup_dict[user] = [sup_temp]
        
for user in val_gt_dict:
    val_gt_dict[user] = np.array(val_gt_dict[user])
    val_pred_dict[user] = np.array(val_pred_dict[user])
    val_idx_dict[user] = np.array(val_idx_dict[user])
    val_res_dict[user] = np.array(val_res_dict[user])
    val_bias_dict[user] = np.array(val_bias_dict[user])
    val_sup_dict[user] = np.array(val_sup_dict[user])

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

# -------------------------------------------------- #

user_tag_pos_arr, user_tag_all_arr = -2*np.ones((16748,46+1)), -np.ones((16748,46+1))
for (x_train,y_train) in train_loader:
    x = x_train.cpu().data.numpy()
    y = y_train.cpu().data.numpy().squeeze()
    for i in range(len(y)):
        user, gt = x[i,0], y[i]
        for j in [2,3,4]:
            cate = x[i,j]
            if cate == 0:
                break
            if user_tag_all_arr[user,cate] < 0:
                user_tag_all_arr[user,cate] = 0
                user_tag_pos_arr[user,cate] = 0
            user_tag_pos_arr[user,cate] += gt
            user_tag_all_arr[user,cate] += 1

user_tag_sup_arr = user_tag_pos_arr / user_tag_all_arr
delta_sup = val_sup_arr - np.array(train_sup_list)
delta_sorted_indices = np.argsort(delta_sup)[:-1]
top_pos_tags, top_neg_tags = delta_sorted_indices[-3:], delta_sorted_indices[:3]

# -------------------------------------------------- #

if args.reconstruction == 1:
    beta, gamma = 20, 4
else:
    beta, gamma = 0, 0
val_zero_pred_dict, test_zero_pred_dict = dict(), dict()
for user in val_gt_dict:
    val_zero_pred_dict[user] = val_pred_dict[user] - val_bias_dict[user]*args.reconstruction + gamma * val_res_dict[user] + beta * val_sup_dict[user]
val_uauc, val_ndcg = UAUC(val_gt_dict, val_zero_pred_dict), NDCG(val_gt_dict, val_zero_pred_dict, [5])[0]
for user in test_gt_dict:
    test_zero_pred_dict[user] = test_pred_dict[user] - test_bias_dict[user]*args.reconstruction + gamma * test_res_dict[user] + beta * test_sup_dict[user]
test_uauc, test_ndcg = UAUC(test_gt_dict, test_zero_pred_dict), NDCG(test_gt_dict, test_zero_pred_dict, [5])[0]

print(beta, gamma, val_uauc, val_ndcg, test_uauc, test_ndcg)