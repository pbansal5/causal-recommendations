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

class PrecEvaluator():
    def __init__(self, datadir, dataset, device,inv_propen,filter_mat,batch_size):
        self.K = 5
        self.metric = "P"
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        # self.filter_mat = dataset.tst_filter_mat
        self.best_score = -9999999

        # temp = np.fromfile('%s/tst_filter_labels.txt'%datadir, sep=' ').astype(int)
        # temp = temp.reshape(-1, 2).T
        # tst_X_Y = read_sparse_file('%s/tst_X_Y.txt'%datadir)
        # trn_X_Y = read_sparse_file('%s/trn_X_Y.txt'%datadir)
        self.inv_propen = inv_propen

        self.filter_mat = filter_mat

    def __call__(self,model):
        xembs,yembs = self.dataset.get_embeds(model,self.batch_size,self.device)
        torch.cuda.empty_cache()
        es = exact_search({'data': yembs.cpu().numpy(), 'query': xembs.cpu().numpy(), 'K': 100, 'device': self.device})
        score_mat = es.getnns_gpu()
        if self.filter_mat is not None:
            self._filter(score_mat)
        res = self.printacc(score_mat, X_Y=self.dataset.mat_mapping, K=self.K)
        recall = xc_metrics.recall(score_mat, self.dataset.mat_mapping, k=100)*100
        print(f'Recall@100: {"%.2f"%recall[99]}')        
        score = res[str(self.K)][self.metric]
        return score,res,recall[99]
    
    def _filter(self,score_mat):
        temp = self.filter_mat.tocoo()
        score_mat[temp.row, temp.col] = 0
        del temp
        score_mat = score_mat.tocsr()
        score_mat.eliminate_zeros()
        return score_mat

            
    def printacc(self,score_mat, K = 5, X_Y = None, disp = True):
        if X_Y is None: X_Y = tst_X_Y
        

        acc = xc_metrics.Metrics(X_Y.tocsr().astype(np.bool), self.inv_propen)
        metrics = np.array(acc.eval(score_mat, K))*100
        df = pd.DataFrame(metrics)

        df.index = ['P', 'nDCG', 'PSP', 'PSnDCG']
        
        df.columns = [str(i+1) for i in range(K)]
        return df

    def get_scores(self,model):
        xembs,yembs = self.dataset.get_embeds(model,self.batch_size,self.device)
        torch.cuda.empty_cache()
        es = exact_search({'data': yembs.cpu().numpy(), 'query': xembs.cpu().numpy(), 'K': 100, 'device': self.device})
        score_mat = es.getnns_gpu()
        if self.filter_mat is not None:
            self._filter(score_mat)
        res = self.printacc(score_mat, X_Y=self.dataset.mat_mapping, K=self.K)
        recall = xc_metrics.recall(score_mat, self.dataset.mat_mapping, k=100)*100
        print(f'Recall@100: {"%.2f"%recall[99]}')        
        score = res[str(self.K)][self.metric]
        return score,res,score_mat
    