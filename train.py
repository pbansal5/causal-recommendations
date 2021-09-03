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
from model import *
from loader import *
from utils import *
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# torch.multiprocessing.set_start_method('spawn')# good solution !!!!

expname = 'WordNetEvenBiggestNoCutRandomMaskOtherModelOOD'
# expname = 'BERT'
print (expname)
logdir = '/home/t-pbansal'
datadir='/ecstorage/bert-opt/datasets'
# logdir = '/home/t-pbansal/logs'
# datadir='/home/t-pbansal/datasets'
datadir='%s/LF-AmazonTitles-131K'%datadir
trn_X_Y = read_sparse_file('%s/trn_X_Y.txt'%datadir)
tst_X_Y = read_sparse_file('%s/tst_X_Y.txt'%datadir)
trn_point_text_files = [x.strip() for x in open('%s/raw/trn_X.title.txt'%datadir).readlines()]
tst_point_text_files = [x.strip() for x in open('%s/raw/tst_X.title.txt'%datadir).readlines()]
label_text_files = [x.strip() for x in open('%s/raw/Y.title.txt'%datadir).readlines()]

temp = np.fromfile('%s/tst_filter_labels.txt'%datadir, sep=' ').astype(int)
temp = temp.reshape(-1, 2).T
inv_propen = xc_metrics.compute_inv_propesity(read_sparse_file('%s/trn_X_Y.txt'%datadir), 0.6, 2.6)
filter_mat = sp.coo_matrix((np.ones(temp.shape[1]), (temp[0], temp[1])), tst_X_Y.shape).tocsr()
# inv_propen = xc_metrics.compute_inv_propesity(read_sparse_file('%s/trn_X_Y.txt'%datadir), 0.6, 2.6)[-int(total_lbls/5):]
# filter_mat = filter_mat[:,-int(total_lbls/5):]

num_train = int(len(label_text_files)*0.5)
np.random.seed(0)
trn_label_indices = np.random.choice(len(label_text_files),num_train,replace=False)
tst_label_indices = np.setdiff1d(np.arange(len(label_text_files)), trn_label_indices)

trn_X_Y = trn_X_Y.tocsc()[:,trn_label_indices]
ood_tst_X_Y = tst_X_Y.tocsc()[:,tst_label_indices]
ood_label_text_files = np.array(label_text_files)[tst_label_indices].tolist()
ood_inv_propen = inv_propen[tst_label_indices]
ood_filter_mat = filter_mat[:,tst_label_indices]
iid_tst_X_Y = tst_X_Y.tocsc()[:,trn_label_indices]
iid_label_text_files = np.array(label_text_files)[trn_label_indices].tolist()
iid_inv_propen = inv_propen[trn_label_indices]
iid_filter_mat = filter_mat[:,trn_label_indices]

total_lbls = trn_X_Y.shape[1]
org_tst_X_Y_shape = tst_X_Y.shape

# trn_X_Y = trn_X_Y.tocsc()[:,:int(4*total_lbls/5)]
# tst_X_Y = tst_X_Y.tocsc()[:,-int(total_lbls/5):]

device = torch.device('cuda:0')
batch_size = 256
max_epoch = 5000
gamma = 0.995
iteration = 0

model = BERTModel(gamma = gamma).to(device)
results_dir = '%s/tensorboard/%s_%fgamma_%dbs'%(logdir,expname,gamma,batch_size)
writer = SummaryWriter(log_dir=results_dir)

# prefetch_factor = 2 
# num_workers = 0
prefetch_factor = 8 
num_workers = 4
pin_memory = False
train_dataloader = torch.utils.data.DataLoader(BertDataset(datadir=datadir,point_text_files=trn_point_text_files,label_text_files=iid_label_text_files,mapping=trn_X_Y,device=device),num_workers=num_workers,prefetch_factor=prefetch_factor,batch_size=batch_size,drop_last=True,shuffle=True,collate_fn=collate_fn,pin_memory=pin_memory)
iid_test_dataloader = torch.utils.data.DataLoader(BertDataset(datadir=datadir,point_text_files=tst_point_text_files,label_text_files=iid_label_text_files,mapping=iid_tst_X_Y,device=device),num_workers=num_workers,prefetch_factor=prefetch_factor,batch_size=batch_size,drop_last=True,shuffle=True,collate_fn=collate_fn,pin_memory=pin_memory)
ood_test_dataloader = torch.utils.data.DataLoader(BertDataset(datadir=datadir,point_text_files=tst_point_text_files,label_text_files=ood_label_text_files,mapping=ood_tst_X_Y,device=device),num_workers=num_workers,prefetch_factor=prefetch_factor,batch_size=batch_size,drop_last=True,shuffle=True,collate_fn=collate_fn,pin_memory=pin_memory)
iid_evaluator = PrecEvaluator(datadir, iid_test_dataloader.dataset,device,iid_inv_propen,iid_filter_mat,100)
ood_evaluator = PrecEvaluator(datadir, ood_test_dataloader.dataset,device,ood_inv_propen,ood_filter_mat,100)

optimizer = AdamW(model.parameters(), lr=1e-4,weight_decay=1e-6)
num_training_steps = int(max_epoch * len(train_dataloader))
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_training_steps=num_training_steps
)


best_score = float(0)

# torch.save(model.state_dict(),os.path.join(results_dir,'checkpoint.pt'))
# print ('saved checkpoint for epoch %d'%epoch)
# exit()

for epoch in range(max_epoch):
    print ("starting epoch %d at iteration %d"%(epoch,iteration))
    for _,(x,y) in enumerate(train_dataloader):
        positives,negatives,loss = model(x,y)
        if (iteration %10  == 0):
            writer.add_scalar('train/loss',loss,iteration)
            writer.add_scalar('train/positives',positives,iteration)
            writer.add_scalar('train/negatives',negatives,iteration)
            writer.add_scalar('train/lr',lr_scheduler.get_lr()[0],iteration)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        iteration += 1
    if(epoch %5 == 0):
        positives,negatives,loss,count = 0,0,0,0
        with torch.no_grad():
            for _,(x,y) in enumerate(iid_test_dataloader):
                positives_,negatives_,loss_ = model(x,y)
                loss += loss_
                positives += positives_
                negatives += negatives_
                count += 1
        score,res,recall = iid_evaluator.__call__(model)
        writer.add_scalar('val/P@1',res['1']['P'],iteration)
        writer.add_scalar('val/P@3',res['3']['P'],iteration)
        writer.add_scalar('val/P@5',res['5']['P'],iteration)
        writer.add_scalar('val/recall',recall,iteration)
        writer.add_scalar('val/loss',loss/count,iteration)
        writer.add_scalar('val/positives',positives/count,iteration)
        writer.add_scalar('val/negatives',negatives/count,iteration)
        print ('==> Epoch %d, P@1 : %f, P@3 : %f, P@5 : %f'%(epoch,res['1']['P'],res['3']['P'],res['5']['P']))


        positives,negatives,loss,count = 0,0,0,0
        with torch.no_grad():
            for _,(x,y) in enumerate(ood_test_dataloader):
                positives_,negatives_,loss_ = model(x,y)
                loss += loss_
                positives += positives_
                negatives += negatives_
                count += 1
        score,res,recall = ood_evaluator.__call__(model)
        writer.add_scalar('ood_val/P@1',res['1']['P'],iteration)
        writer.add_scalar('ood_val/P@3',res['3']['P'],iteration)
        writer.add_scalar('ood_val/P@5',res['5']['P'],iteration)
        writer.add_scalar('ood_val/recall',recall,iteration)
        writer.add_scalar('ood_val/loss',loss/count,iteration)
        writer.add_scalar('ood_val/positives',positives/count,iteration)
        writer.add_scalar('ood_val/negatives',negatives/count,iteration)
        print ('==> Epoch %d, P@1 : %f, P@3 : %f, P@5 : %f'%(epoch,res['1']['P'],res['3']['P'],res['5']['P']))
        if (score > best_score):
            best_score = score
            torch.save(model.state_dict(),os.path.join(results_dir,'checkpoint.pt'))
            print ('saved checkpoint for epoch %d'%epoch)
