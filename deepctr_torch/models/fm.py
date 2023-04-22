import torch

from .basemodel import BaseModel
from ..layers import FM

class Factorization_Machine(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_embedding=1e-5, l2_reg_linear=1e-5, \
        init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(Factorization_Machine, self).__init__(\
            linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, \
                init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.fm = FM()
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        linear_logit = self.linear_model(X)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        second_order_logit = self.fm(fm_input)
        logit = linear_logit + second_order_logit
        y_pred = self.out(logit)    # out: basemodel (+bias)
        return y_pred