import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import os
import transformers as ppb
from transformers import RobertaModel, AutoConfig
import warnings
from sentence_transformers import SentenceTransformer
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')
from xclib.data.data_utils import read_sparse_file
import copy
from transformers import get_scheduler,AdamW
from transformers import AutoTokenizer
import nlpaug.augmenter.word as naw
import nlpaug
import nltk
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
import scipy.sparse as sp
from nns import exact_search



# class BERTModel(torch.nn.Module):
#     def __init__(self,gamma):
#         super().__init__()
#         self.encoder = SentenceTransformer('msmarco-distilbert-base-v3')
#         self.rep_dim = self.encoder.get_sentence_embedding_dimension()
#         self.hidden_dim = 8192
#         self.predictor = torch.nn.Sequential(torch.nn.Linear(self.rep_dim,self.hidden_dim),
#                                              torch.nn.BatchNorm1d(self.hidden_dim),
#                                              torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.hidden_dim),
#                                              torch.nn.BatchNorm1d(self.hidden_dim),
#                                              torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.hidden_dim),
#                                              torch.nn.BatchNorm1d(self.hidden_dim),
#                                              torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.hidden_dim))
#         self.lambda_ = 1e-2
        
#     def forward(self,x,y):
#         device = self.predictor[0].weight.device
#         x = self.to_gpu(x,device)
#         y = self.to_gpu(y,device)

#         x_rep = torch.nn.functional.normalize(self.encoder(x)['sentence_embedding'],dim=-1)
#         y_rep = torch.nn.functional.normalize(self.encoder(y)['sentence_embedding'],dim=-1)
#         # x_rep = self.encoder(x)['sentence_embedding']
#         # y_rep = self.encoder(y)['sentence_embedding']

#         x_embeds = self.predictor(x_rep)
#         y_embeds = self.predictor(y_rep)
#         x_embeds = (x_embeds-x_embeds.mean(dim=0,keepdims=True))/x_embeds.std(dim=0,keepdims=True)
#         y_embeds = (y_embeds-y_embeds.mean(dim=0,keepdims=True))/y_embeds.std(dim=0,keepdims=True)

#         correlations = (x_embeds.T@y_embeds)/x_embeds.shape[0]
#         iden_matrix = torch.eye(correlations.shape[0]).to(correlations.device)
#         errors = (correlations-iden_matrix)**2
#         total_error = errors.sum()
#         diags_error =  torch.diag(errors).sum()
#         offdiags_error =  total_error-diags_error
#         loss = diags_error+self.lambda_*offdiags_error

#         with torch.no_grad():
#             scores = x_rep@y_rep.T
#             positives = torch.diag(scores).sum()
#             negatives = scores.sum()-positives
#             positives /= scores.shape[0]
#             negatives /= ((scores.shape[0]-1)*(scores.shape[0]))
        
#         return positives,negatives,loss
    
#     def get_embed(self,x):
#         return torch.nn.functional.normalize(self.encoder(x)['sentence_embedding'],dim=-1)

#     def to_gpu(self,x,device):
#         return {'input_ids':x['input_ids'].to(device),'attention_mask':x['attention_mask'].to(device)}

class BERTModel(torch.nn.Module):
    def __init__(self,gamma):
        super().__init__()
        self.encoder = SentenceTransformer('msmarco-distilbert-base-v3')
        self.rep_dim = self.encoder.get_sentence_embedding_dimension()
        self.hidden_dim = 8192
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = torch.nn.Sequential(torch.nn.Linear(self.rep_dim,self.hidden_dim),
                                             torch.nn.BatchNorm1d(self.hidden_dim),
                                             torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                                             torch.nn.BatchNorm1d(self.hidden_dim),
                                             torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                                             torch.nn.BatchNorm1d(self.hidden_dim),
                                            #  torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                                             torch.nn.ReLU(),torch.nn.Linear(self.hidden_dim,self.rep_dim))
        self.gamma = gamma
        
    def forward(self,x,y):
        device = self.predictor[0].weight.device
        x = self.to_gpu(x,device)
        y = self.to_gpu(y,device)
        # y_embeds = torch.nn.functional.normalize(self.predictor(torch.nn.functional.normalize(self.encoder(y)['sentence_embedding'],dim=-1)),dim=-1)
        # x_embeds = torch.nn.functional.normalize(self.target_encoder(x)['sentence_embedding'],dim=-1).clone().detach()
        flip = np.random.binomial(1,0.5)
        if (flip == 0):
            x_embeds = torch.nn.functional.normalize(self.predictor(torch.nn.functional.normalize(self.encoder(x)['sentence_embedding'],dim=-1)),dim=-1)
            # x_embeds = torch.nn.functional.normalize(self.predictor(self.encoder(x)['sentence_embedding']),dim=-1)
            y_embeds = torch.nn.functional.normalize(self.target_encoder(y)['sentence_embedding'],dim=-1).clone().detach()
        else:
            y_embeds = torch.nn.functional.normalize(self.predictor(torch.nn.functional.normalize(self.encoder(y)['sentence_embedding'],dim=-1)),dim=-1)
            # y_embeds = torch.nn.functional.normalize(self.predictor(self.encoder(y)['sentence_embedding']),dim=-1)
            x_embeds = torch.nn.functional.normalize(self.target_encoder(x)['sentence_embedding'],dim=-1).clone().detach()
        scores = x_embeds@y_embeds.T
        positives = torch.diag(scores).sum()
        negatives = scores.sum()-positives
        positives /= scores.shape[0]
        negatives /= ((scores.shape[0]-1)*(scores.shape[0]))
        self.update_target()
        return positives,negatives,-positives
    
    def get_embed(self,x):
        return torch.nn.functional.normalize(self.encoder(x)['sentence_embedding'],dim=-1)

    def to_gpu(self,x,device):
        return {'input_ids':x['input_ids'].to(device),'attention_mask':x['attention_mask'].to(device)}
    
    def update_target(self):
        target_dict = self.target_encoder.state_dict()
        online_dict = self.encoder.state_dict()
        for key in online_dict.keys():
            target_dict[key] = target_dict[key]*self.gamma + online_dict[key]*(1-self.gamma)
        self.target_encoder.load_state_dict(target_dict)
