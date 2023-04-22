from __future__ import print_function

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import os

from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm


from ..inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays


def UAUC(ut_dict, pt_dict): 
    uauc = 0.0
    u_size = 0.0
    for k in ut_dict:
        if (1 in ut_dict[k]) and (0 in ut_dict[k]):
            uauc_one = roc_auc_score(np.array(ut_dict[k]),np.array(pt_dict[k]))
            uauc += uauc_one
            u_size += 1.0
    return uauc/ u_size


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


def calc_ndcg(gt,pred,k,logsum):
    idcg = logsum[min(k,sum(gt))-1]
    dcg = 0
    k = int(min(k,len(gt)))
    max_indices = np.argsort(pred)[::-1]
    for i,idx in enumerate(max_indices[:k]):
        dcg += gt[int(idx)]/np.log2(i+2)
    return dcg/idcg


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
            # nn.init.uniform_(tensor.weight, a=-1.0, b=1.0)
            # nn.init.constant_(tensor.weight, 0)
            # nn.init.xavier_uniform_(tensor.weight, gain=1.0)
        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)
            # nn.init.uniform_(self.weight, a=-1.0, b=1.0)
            # nn.init.constant_(self.weight, 0)
            # nn.init.xavier_uniform_(self.weight, gain=1.0)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(sparse_embedding_list[0].device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):

        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)


    def fit(self, x=None, y=None, batch_size=256, verbose=1, initial_epoch=0, validation_split=0.2,
            validation_data=None, test_split=0.1, args=None, presplit_data=None, epoch_log=False, prepared_tensor=False):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks(not used): List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return(TODO): A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """

        if presplit_data is None and prepared_tensor is False:
            if isinstance(x, dict):
                x = [x[feature] for feature in self.feature_index]

            do_validation = False
            if validation_data:
                do_validation = True
                if len(validation_data) == 2:
                    val_x, val_y = validation_data
                    val_sample_weight = None
                elif len(validation_data) == 3:
                    val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
                else:
                    raise ValueError(
                        'When passing a `validation_data` argument, '
                        'it must contain either 2 items (x_val, y_val), '
                        'or 3 items (x_val, y_val, val_sample_weights), '
                        'or alternatively it could be a dataset or a '
                        'dataset or a dataset iterator. '
                        'However we received `validation_data=%s`' % validation_data)
                if isinstance(val_x, dict):
                    val_x = [val_x[feature] for feature in self.feature_index]

            elif validation_split+test_split and 0. < validation_split+test_split < 1.:
                do_validation = True
                if hasattr(x[0], 'shape'):
                    split_at = round(x[0].shape[0] * (1. - validation_split-test_split))
                else:
                    split_at = round(len(x[0]) * (1. - validation_split-test_split))
                x, val_x = (slice_arrays(x, 0, split_at), slice_arrays(x, split_at))
                y, val_y = (slice_arrays(y, 0, split_at), slice_arrays(y, split_at))
                
                if hasattr(val_x[0], 'shape'):
                    new_split_at = round(val_x[0].shape[0] * validation_split/(validation_split+test_split))
                else:
                    new_split_at = round(len(val_x[0]) * validation_split/(validation_split+test_split))
                val_x, test_x = (slice_arrays(val_x, 0, new_split_at), slice_arrays(val_x, new_split_at))
                val_y, test_y = (slice_arrays(val_y, 0, new_split_at), slice_arrays(val_y, new_split_at))

            else:
                val_x = []
                val_y = []
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)

            train_tensor_data = Data.TensorDataset(
                torch.from_numpy(
                    np.concatenate(x, axis=-1)),
                torch.from_numpy(y))
        else:
            train_tensor_data, train_loader, val_x, val_y, test_x, test_y = presplit_data
            do_validation = True

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        # train_loader = DataLoader(
        #     dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size, num_workers=2)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # Train
        patience = 3
        earlystopper = EarlyStopping(patience=patience)

        # print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
        #     len(train_tensor_data), len(val_y), steps_per_epoch))

        val_pred, filename = [], '_'.join([args.model, args.dataset, args.loss])
        if epoch_log is True and prepared_tensor is False:  # not used
            trainx, trainy, train_pred, valx_list, valy_list = [], [], [], [], []
            with torch.no_grad():
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x, y = x_train.numpy(), np.squeeze(y_train.numpy()) # (256?,11), (256?,)
                        trainx.append(x)
                        trainy.append(y)
            trainx, trainy = np.concatenate(trainx, axis=0), np.concatenate(trainy, axis=0)
            print(trainx.shape, trainy.shape)
            if do_validation:
                '''
                if isinstance(val_x, dict):
                    val_x = [val_x[feature] for feature in self.feature_index]
                for i in range(len(val_x)):
                    if len(val_x[i].shape) == 1:
                        val_x[i] = np.expand_dims(val_x[i], axis=1)
                val_data = Data.TensorDataset(torch.from_numpy(np.concatenate(val_x, axis=-1)))
                val_loader = DataLoader(dataset=val_data, shuffle=False, batch_size=batch_size)
                '''
                for i in range(len(val_x)):
                    if len(val_x[i].shape) == 1:
                        val_x[i] = np.expand_dims(val_x[i], axis=1)
                val_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(val_x, axis=-1)), torch.from_numpy(val_y))
                val_loader = DataLoader(dataset=val_tensor_data, shuffle=False, batch_size=batch_size, num_workers=2)
                with torch.no_grad():
                    with tqdm(enumerate(val_loader), disable=verbose != 1) as t:
                        for _, (x_val, y_val) in t:
                            x, y = x_val.numpy(), np.squeeze(y_val.numpy()) # (256?,11), (256?,)
                            valx_list.append(x)
                            valy_list.append(y)
                valx_arr, valy_arr = np.concatenate(valx_list, axis=0), np.concatenate(valy_list, axis=0)
                print(valx_arr.shape, valy_arr.shape)
                filename = '_'.join([args.model, args.dataset, args.loss])

        for epoch in range(initial_epoch, args.max_epoch):
            epoch_logs, train_result = {}, {}
            start_time = time.time()
            total_loss_epoch = 0
            if epoch_log is True:
                train_pred_epoch = []
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x)

                        optim.zero_grad()
                        loss = loss_func(y_pred, y)
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        # loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        # if verbose > 0:
                        #     for name, metric_fun in self.metrics.items():
                        #         if name not in train_result:
                        #             train_result[name] = []
                        #         train_result[name].append(metric_fun(
                        #             y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                        if epoch_log is True and prepared_tensor is False:
                            train_pred_epoch.append(y_pred.cpu().data.numpy())
                if epoch_log is True and prepared_tensor is False:
                    train_pred_epoch = np.concatenate(train_pred_epoch, axis=0)
                    train_pred.append(train_pred_epoch)

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                if epoch_log is False:
                    eval_result = self.evaluate(val_x, val_y, batch_size, epoch_log=False, prepared_tensor=prepared_tensor)
                else:
                    eval_result, val_pred_epoch = self.evaluate(val_x, val_y, batch_size, epoch_log=True, prepared_tensor=prepared_tensor)
                    val_pred.append(val_pred_epoch)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, args.max_epoch))

                eval_str = "{0}s - ".format(epoch_time)

                for name in list(epoch_logs.keys()):
                    eval_str += name + ": {0: .4f}".format(epoch_logs[name])

                # if do_validation:
                #     for name in self.metrics:
                #         eval_str += " - " + "val_" + name + ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            if earlystopper(epoch_logs, epoch_logs['val_uauc'], model) is True:
                break

        if epoch_log is True:
            if prepared_tensor is False:
                train_pred = np.concatenate(train_pred, axis=1)
            if do_validation:
                val_pred = np.concatenate(val_pred, axis=1)

        print('Loading {}th epoch'.format(min(epoch-patience,args.max_epoch)))
        model.load_state_dict(earlystopper.best_state)

        test_result = self.evaluate(test_x, test_y, batch_size, prepared_tensor=prepared_tensor)
        for name, result in test_result.items():
            epoch_logs["test_" + name] = result
        test_str = ''
        for name in list(test_result.keys()):
            test_str += "test_" + name + ": {0: .4f}".format(epoch_logs["test_" + name])
        print(test_str)
        

    def evaluate(self, x, y, batch_size=256, uauc_ndcg=True, epoch_log=False, prepared_tensor=False):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size, prepared_tensor=prepared_tensor)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(np.array(y), pred_ans)
        if uauc_ndcg:
            if prepared_tensor is False:
                if isinstance(x, dict):
                    x = [x[feature] for feature in self.feature_index]
                for i in range(len(x)):
                    if len(x[i].shape) == 1:
                        x[i] = np.expand_dims(x[i], axis=1)
                x = np.concatenate(x, axis=-1)
            else:
                x = np.array(x)
            # construct dict
            users, utdict, updict = x[:,0], dict(), dict()
            for user,gt,pred in zip(users,y,pred_ans):
                try:
                    utdict[user].append(gt[0])
                    updict[user].append(pred[0])
                except:
                    utdict[user] = [gt[0]]
                    updict[user] = [pred[0]]
            # calculate uauc
            eval_result['uauc'] = UAUC(utdict,updict)
            atk_list = [5,10]
            ndcg_list = NDCG(utdict, updict, atk_list)
            for k, ndcg_atk in zip(atk_list, ndcg_list):
                eval_result['ndcg@'+str(k)] = ndcg_atk
        if epoch_log:
            return eval_result, pred_ans
        return eval_result

    def predict(self, x, batch_size=256, prepared_tensor=False):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if prepared_tensor is False:
            if isinstance(x, dict):
                x = [x[feature] for feature in self.feature_index]
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)

            tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)))
            test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=batch_size)
        else:
            tensor_data = Data.TensorDataset(x)
            test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer, loss=None, metrics=None, lr=0.001):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer, lr)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer, lr):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "bce":
                # loss_func = F.binary_cross_entropy
                loss_func = nn.BCEWithLogitsLoss(reduction='sum')
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "bce" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_state = None
        self.best_epoch_log = dict()

    def __call__(self, epoch_logs, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, epoch_logs)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, epoch_logs)
            self.counter = 0
        return self.early_stop
        

    def save_checkpoint(self, val_score, model, epoch_logs):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        self.best_state = {key: value.cpu() for key, value in model.state_dict().items()}                
        self.val_score_min = val_score
        self.best_epoch_log = epoch_logs