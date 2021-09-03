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
from random import randint

  

# class BertDataset(torch.utils.data.Dataset):

#     def __init__(self, datadir,mapping, device,split):
#         self.lbl_size = 131073
#         self.datadir = datadir
#         self.device=device
#         self.size = 294805 if split =='trn' else 134835
#         if split == 'trn':
#             self.point_text_files = [x.strip() for x in open('%s/raw/trn_X.title.txt'%self.datadir).readlines()]
#         else :
#             self.point_text_files = [x.strip() for x in open('%s/raw/tst_X.title.txt'%self.datadir).readlines()]
#         self.label_text_files = [x.strip() for x in open('%s/raw/Y.title.txt'%self.datadir).readlines()]

#         if split == 'trn':
#             self.point_text_files = self.point_text_files[:mapping.shape[0]]
#             self.label_text_files = self.label_text_files[:mapping.shape[1]]
#         else : 
#             self.point_text_files = self.point_text_files[:mapping.shape[0]]
#             self.label_text_files = self.label_text_files[-mapping.shape[1]:]

#         self.mat_mapping = mapping.tocsc()
#         self.num_labels = mapping.shape[1]
#         self.mapping = mapping.nonzero()
#         self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#         self.maxsize=32
#         self.aug = naw.SynonymAug(aug_src='wordnet',aug_max=5)
#         self.dropout = 1.0
#         # aug = naw.WordEmbsAug(model_type='word2vec',model_path = './wiki-news-300d-1M.vec')

#     def __getitem__(self,index,augment=False):
#         if (index>=self.num_labels):
#             label_index = index-self.num_labels
#             text = self.label_text_files[label_index]
#             partial_text = text.split(' ')
#             left = randint(0, len(partial_text)-1)
#             right = randint(left+1, len(partial_text))
#             partial_text = partial_text[left:right]
#             point_data,label_data = self.convert_joint(text,partial_text,augment=augment)
#         else : 
#             label_array = (self.mat_mapping[:,index]).nonzero()[0]  
#             label_index = index
#             # point_index1 = np.random.choice(label_array)
#             # point_index2 = np.random.choice(label_array)
#             # point_data,label_data = self.convert_joint(self.point_text_files[point_index1]+' '+self.point_text_files[point_index2],self.label_text_files[label_index],augment=augment)
#             point_index = np.random.choice(label_array)
#             point_data,label_data = self.convert_joint(self.point_text_files[point_index],self.label_text_files[label_index],augment=augment)
#     #         point_data = self.convert(self.point_text_files[self.mapping[0][index]],augment=augment)
#     #         label_data = self.convert(self.label_text_files[self.mapping[1][index]],augment=augment)
#         mask1 = torch.bernoulli(torch.ones(point_data['input_ids'][0].shape)*self.dropout)
#         mask2 = torch.bernoulli(torch.ones(label_data['input_ids'][0].shape)*self.dropout)

#         return (torch.Tensor(point_data['input_ids'][0])*mask1,
#                torch.Tensor(label_data['input_ids'][0])*mask2,
#                torch.Tensor(point_data['attention_mask'][0]),
#                torch.Tensor(label_data['attention_mask'][0]))

# #     def __getitem__(self,index,augment=False):
# #         label_array = (self.mat_mapping[:,index]).nonzero()[0]  
# #         point_index = np.random.choice(label_array)
# #         label_index = index
# #         point_data,label_data = self.convert_joint(self.point_text_files[point_index],self.label_text_files[label_index],augment=augment)
# # #         point_data = self.convert(self.point_text_files[self.mapping[0][index]],augment=augment)
# # #         label_data = self.convert(self.label_text_files[self.mapping[1][index]],augment=augment)
# #         mask1 = torch.bernoulli(torch.ones(point_data['input_ids'][0].shape)*self.dropout)
# #         mask2 = torch.bernoulli(torch.ones(label_data['input_ids'][0].shape)*self.dropout)

# #         return (torch.Tensor(point_data['input_ids'][0])*mask1,
# #                torch.Tensor(label_data['input_ids'][0])*mask2,
# #                torch.Tensor(point_data['attention_mask'][0]),
# #                torch.Tensor(label_data['attention_mask'][0]))
    
#     def convert_joint(self,textp,textl,augment):
#         if augment : 
#             textp=self.aug.augment(textp,n=1)
#             textl=self.aug.augment(textl,n=1)

#         # combined = textp.split(' ')+textl.split(' ')
#         # split_point = int(np.random.uniform(1,len(combined)))
#         # textp = ' '.join(combined[:split_point])
#         # textl = ' '.join(combined[split_point:])
        
#         return (self.tokenizer(textp,add_special_tokens = True,
#                          truncation=True,return_tensors = 'np',
#                          return_attention_mask = True,padding = 'max_length',max_length=self.maxsize),
#                 self.tokenizer(textl,add_special_tokens = True,
#                          truncation=True,return_tensors = 'np',
#                          return_attention_mask = True,padding = 'max_length',max_length=self.maxsize))
    
#     def convert(self,text,augment):
#         if augment : 
#             text=self.aug.augment(text,n=1)
#         return self.tokenizer(text,add_special_tokens = True,
#                          truncation=True,return_tensors = 'np',
#                          return_attention_mask = True,padding = 'max_length',max_length=self.maxsize)
    
#     def get_embeds(self,model,batch_size,device):
#         labels = self.convert(self.label_text_files,augment=False)
#         points = self.convert(self.point_text_files,augment=False)
#         num_labels = labels['input_ids'].shape[0]
#         num_points = points['input_ids'].shape[0]
#         with torch.no_grad():
#             label_embeds = []
#             for i in range(0,num_labels,batch_size):
#                 label_embeds.append(model.get_embed({'input_ids':torch.LongTensor(labels['input_ids'][i:i+batch_size]).to(device),
#                                           'attention_mask':torch.LongTensor(labels['attention_mask'][i:i+batch_size]).to(device)}).cpu())
#             label_embeds = torch.cat(label_embeds,dim=0)

#             point_embeds = []
#             for i in range(0,num_points,batch_size):
#                 point_embeds.append(model.get_embed({'input_ids':torch.LongTensor(points['input_ids'][i:i+batch_size]).to(device),
#                                           'attention_mask':torch.LongTensor(points['attention_mask'][i:i+batch_size]).to(device)}).cpu())
#             point_embeds = torch.cat(point_embeds,dim=0)
#         return point_embeds,label_embeds

#     # def __len__(self):
#     #     return self.num_labels
#     def __len__(self):
#         return self.num_labels*2

# class BertDataset(torch.utils.data.Dataset):

#     def __init__(self, datadir,mapping, device,split):
#         self.lbl_size = 131073
#         self.datadir = datadir
#         self.device=device
#         self.size = 294805 if split =='trn' else 134835
#         if split == 'trn':
#             self.point_text_files = [x.strip() for x in open('%s/raw/trn_X.title.txt'%self.datadir).readlines()]
#         else :
#             self.point_text_files = [x.strip() for x in open('%s/raw/tst_X.title.txt'%self.datadir).readlines()]
#         self.label_text_files = [x.strip() for x in open('%s/raw/Y.title.txt'%self.datadir).readlines()]
#         self.mat_mapping = mapping
#         self.mapping = mapping.nonzero()
#         self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#         self.maxsize=32
#         self.aug = naw.SynonymAug(aug_src='wordnet',aug_max=5)
#         # aug = naw.WordEmbsAug(model_type='word2vec',model_path = './wiki-news-300d-1M.vec')

#     def __getitem__(self,index,augment=False):
#         point_data,label_data = self.convert_joint(self.point_text_files[self.mapping[0][index]],self.label_text_files[self.mapping[1][index]],augment=augment)
# #         point_data = self.convert(self.point_text_files[self.mapping[0][index]],augment=augment)
# #         label_data = self.convert(self.label_text_files[self.mapping[1][index]],augment=augment)
#         return (torch.Tensor(point_data['input_ids'][0]),
#                torch.Tensor(label_data['input_ids'][0]),
#                torch.Tensor(point_data['attention_mask'][0]),
#                torch.Tensor(label_data['attention_mask'][0]))
    
#     def convert_joint(self,textp,textl,augment):
#         if augment : 
#             textp=self.aug.augment(textp,n=1)
#             textl=self.aug.augment(textl,n=1)

#         # combined = textp.split(' ')+textl.split(' ')
#         # split_point = int(np.random.uniform(1,len(combined)))
#         # textp = ' '.join(combined[:split_point])
#         # textl = ' '.join(combined[split_point:])
        
#         return (self.tokenizer(textp,add_special_tokens = True,
#                          truncation=True,return_tensors = 'np',
#                          return_attention_mask = True,padding = 'max_length',max_length=self.maxsize),
#                 self.tokenizer(textl,add_special_tokens = True,
#                          truncation=True,return_tensors = 'np',
#                          return_attention_mask = True,padding = 'max_length',max_length=self.maxsize))
    
#     def convert(self,text,augment):
#         if augment : 
#             text=self.aug.augment(text,n=1)
#         return self.tokenizer(text,add_special_tokens = True,
#                          truncation=True,return_tensors = 'np',
#                          return_attention_mask = True,padding = 'max_length',max_length=self.maxsize)
    
#     def get_embeds(self,model,batch_size,device):
#         labels = self.convert(self.label_text_files,augment=False)
#         points = self.convert(self.point_text_files,augment=False)
#         num_labels = labels['input_ids'].shape[0]
#         num_points = points['input_ids'].shape[0]
#         with torch.no_grad():
#             label_embeds = []
#             for i in range(0,num_labels,batch_size):
#                 label_embeds.append(model.get_embed({'input_ids':torch.LongTensor(labels['input_ids'][i:i+batch_size]).to(device),
#                                           'attention_mask':torch.LongTensor(labels['attention_mask'][i:i+batch_size]).to(device)}).cpu())
#             label_embeds = torch.cat(label_embeds,dim=0)

#             point_embeds = []
#             for i in range(0,num_points,batch_size):
#                 point_embeds.append(model.get_embed({'input_ids':torch.LongTensor(points['input_ids'][i:i+batch_size]).to(device),
#                                           'attention_mask':torch.LongTensor(points['attention_mask'][i:i+batch_size]).to(device)}).cpu())
#             point_embeds = torch.cat(point_embeds,dim=0)
#         return point_embeds,label_embeds

#     def __len__(self):
#         return len(self.mapping[0])

class BertDataset(torch.utils.data.Dataset):

    def __init__(self, datadir,point_text_files,label_text_files,mapping, device):
        self.datadir = datadir
        self.device=device
        self.point_text_files = point_text_files
        self.label_text_files = label_text_files
        self.lbl_size = len(label_text_files)
        self.mat_mapping = mapping
        self.mapping = mapping.nonzero()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.maxsize=32
        self.aug = naw.SynonymAug(aug_src='wordnet',aug_max=5)
        # aug = naw.WordEmbsAug(model_type='word2vec',model_path = './wiki-news-300d-1M.vec')
        self.dropout = 1.0

    def __getitem__(self,index,augment=False):
        point_data,label_data = self.convert_joint(self.point_text_files[self.mapping[0][index]],self.label_text_files[self.mapping[1][index]],augment=augment)
#         point_data = self.convert(self.point_text_files[self.mapping[0][index]],augment=augment)
#         label_data = self.convert(self.label_text_files[self.mapping[1][index]],augment=augment)
        mask1 = torch.bernoulli(torch.ones(point_data['input_ids'][0].shape)*self.dropout)
        mask2 = torch.bernoulli(torch.ones(label_data['input_ids'][0].shape)*self.dropout)

        return (torch.Tensor(point_data['input_ids'][0])*mask1,
               torch.Tensor(label_data['input_ids'][0])*mask2,
               torch.Tensor(point_data['attention_mask'][0]),
               torch.Tensor(label_data['attention_mask'][0]))
    
    def convert_joint(self,textp,textl,augment):
        if augment : 
            textp=self.aug.augment(textp,n=1)
            textl=self.aug.augment(textl,n=1)

        # combined = textp.split(' ')+textl.split(' ')
        # split_point = int(np.random.uniform(1,len(combined)))
        # textp = ' '.join(combined[:split_point])
        # textl = ' '.join(combined[split_point:])
        
        return (self.tokenizer(textp,add_special_tokens = True,
                         truncation=True,return_tensors = 'np',
                         return_attention_mask = True,padding = 'max_length',max_length=self.maxsize),
                self.tokenizer(textl,add_special_tokens = True,
                         truncation=True,return_tensors = 'np',
                         return_attention_mask = True,padding = 'max_length',max_length=self.maxsize))
    
    def convert(self,text,augment):
        if augment : 
            text=self.aug.augment(text,n=1)
        return self.tokenizer(text,add_special_tokens = True,
                         truncation=True,return_tensors = 'np',
                         return_attention_mask = True,padding = 'max_length',max_length=self.maxsize)
    
    def get_embeds(self,model,batch_size,device):
        labels = self.convert(self.label_text_files,augment=False)
        points = self.convert(self.point_text_files,augment=False)
        num_labels = labels['input_ids'].shape[0]
        num_points = points['input_ids'].shape[0]
        with torch.no_grad():
            label_embeds = []
            for i in range(0,num_labels,batch_size):
                label_embeds.append(model.get_embed({'input_ids':torch.LongTensor(labels['input_ids'][i:i+batch_size]).to(device),
                                          'attention_mask':torch.LongTensor(labels['attention_mask'][i:i+batch_size]).to(device)}).cpu())
            label_embeds = torch.cat(label_embeds,dim=0)

            point_embeds = []
            for i in range(0,num_points,batch_size):
                point_embeds.append(model.get_embed({'input_ids':torch.LongTensor(points['input_ids'][i:i+batch_size]).to(device),
                                          'attention_mask':torch.LongTensor(points['attention_mask'][i:i+batch_size]).to(device)}).cpu())
            point_embeds = torch.cat(point_embeds,dim=0)
        return point_embeds,label_embeds

    def __len__(self):
        return len(self.mapping[0])

# # class SimpleCustomBatch:
# #     def __init__(self, batch):
# #         point_ids,label_ids,point_masks,label_masks = zip(*batch)
# #         self.points = {'input_ids' : torch.stack(point_ids,dim=0).long(),'attention_mask':torch.stack(point_masks,dim=0).long()}
# #         self.labels = {'input_ids' : torch.stack(label_ids,dim=0).long(),'attention_mask':torch.stack(label_masks,dim=0).long()}

# #     # custom memory pinning method on custom type
# #     def pin_memory(self):
# #         self.points['input_ids'] = self.points['input_ids'].pin_memory() 
# #         self.points['attention_mask'] = self.points['attention_mask'].pin_memory() 
# #         self.labels['input_ids'] = self.labels['input_ids'].pin_memory() 
# #         self.labels['attention_mask'] = self.labels['attention_mask'].pin_memory() 
# #         return self.points,self.labels

# # def collate_fn(batch):
# #     SimpleCustomBatch(batch)

def collate_fn(batch):
    point_ids,label_ids,point_masks,label_masks = zip(*batch)
    return {'input_ids' : torch.stack(point_ids,dim=0).long(),'attention_mask':torch.stack(point_masks,dim=0).long()},{'input_ids' : torch.stack(label_ids,dim=0).long(),'attention_mask':torch.stack(label_masks,dim=0).long()}
