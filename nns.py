import numpy as np
import xclib
import torch
import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from tqdm.autonotebook import tqdm, trange

def getnns_cpu(self):
    if self.hp['metric'] != 'ip':
        print('%s not supported.'%self.hp['metric'])
        return 
    
    self.initialize()
    start = time.time()
    for i in tqdm(range(0, self.num_query, self.batch_size)):
        query_slice = self.hp['query'][i:i+self.batch_size]
        prod = np.dot(query_slice, self.w)
        self.indices[i:i+self.batch_size] = np.argsort(prod, axis=1)[:, -self.K:]
        self.data[i:i+self.batch_size] = np.take_along_axis(prod, self.indices[i:i+self.batch_size], axis=1)
    end = time.time()

    print('Total time, time per point : %.2fs, %.4fms/pt'%(end-start, (end-start)*1000/self.num_query))
    return self.getnns(self.hp['data'].shape[0])


def getnns_shorty_gpu(self, shorty):
    if self.hp['metric'] != 'ip':
        print('%s not supported.'%self.hp['metric'])
        return 
    
    self.K = self.hp['K']
    device = self.device
    res = shorty.copy().tocoo()
    
    with torch.no_grad():
        w_gpu = torch.from_numpy(self.hp['data']).float().to(device)
        query_gpu = torch.from_numpy(self.hp['query']).float().to(device)
        rows = res.row
        cols = res.col
        bsz = self.batch_size

        start = time.time()
        for i in tqdm(range(0, res.nnz, bsz)):
            prod_gpu = (query_gpu[rows[i:i+bsz]] * w_gpu[cols[i:i+bsz]]).sum(dim=1)
            res.data[i:i+bsz] = prod_gpu.detach().cpu().numpy()                              
        end = time.time()

        print('Total time, time per point : %.2fs, %.4f ms/pt'%(end-start, (end-start)*1000/self.num_query))
        del w_gpu, query_gpu, prod_gpu
        torch.cuda.empty_cache()
        return res.tocsr()

class exact_search:
    def __init__(self, hp):
        self.hp = {
            'batch_size' : 512,
            'data' : None,
            'query' : None,
            'K' : 10,
            'sim' : True,
            'metric' : 'ip',
            'device': 'cuda:0'
          }
        for k, v in hp.items():
            self.hp[k] = v
        self.initialize()
        
    def initialize(self):
        self.num_query = self.hp['query'].shape[0]
        self.num_base = self.hp['data'].shape[0]
        self.batch_size = self.hp['batch_size']
        self.device = self.hp['device']
        
        if self.hp['metric'] == 'cosine':
            self.hp['query'] = normalize(self.hp['query'], axis=1)
            self.hp['data'] = normalize(self.hp['data'], axis=1)
            self.hp['metric'] = 'ip'
        
        if self.hp['metric'] == 'ip':
            self.w = self.hp['data']
            self.hp['sim'] = True
        elif self.hp['metric'] == 'euclid':
            self.w = self.hp['data']
            self.hp['sim'] = False
            
        self.K = self.hp['K']
        self.data = np.zeros((self.num_query, self.K))
        self.indices = np.zeros((self.num_query, self.K), dtype=int)
        self.indptr = range(0, self.data.shape[0]*self.data.shape[1]+1, self.data.shape[1])
        
    def getnns(self, nc, save = False):
        score_mat = csr_matrix((self.data.ravel(), self.indices.ravel(), self.indptr), (self.num_query, nc))
        if save: 
            sp.save_npz(self.hp['score_mat'], score_mat)
#         del self.data, self.indptr, self.indices
        return score_mat

    def getnns_gpu(self, shard_size=1000000):
        if self.w.shape[0] > shard_size:
            print(f'Doing nns in {(self.w.shape[0]+shard_size-1)//shard_size} shards')
            score_mat = csr_matrix((self.num_query, 0))

            for ctr in tqdm(range(0, self.w.shape[0], shard_size)):
                temp = self.getnns_gpu_shard(range(ctr, min(ctr+shard_size, self.w.shape[0])))
                score_mat = xclib.utils.sparse.retain_topk(sp.hstack((score_mat, temp)).tocsr(), k=self.K)
            return score_mat
        else:
            return self.getnns_gpu_shard(range(self.w.shape[0]))

    def getnns_gpu_shard(self, shard=None):
        device = self.device
        with torch.no_grad():
            w_gpu = torch.from_numpy(self.w[shard]).float().to(device)

            start = time.time()
            for i in tqdm(range(0, self.num_query, self.batch_size)):
                query_slice_gpu = torch.from_numpy(self.hp['query'][i:i+self.batch_size]).float().to(device)

                prod_gpu = None
                if self.hp['metric'] == 'ip':
                    prod_gpu = torch.matmul(w_gpu, query_slice_gpu.T).T
                elif self.hp['metric'] == 'euclid':
                    prod_gpu = torch.cdist(query_slice_gpu, w_gpu)

                batch_data_gpu, batch_indices_gpu = torch.topk(prod_gpu, k=self.K, sorted=True, largest=self.hp['sim'])
                self.data[i:i+self.batch_size], self.indices[i:i+self.batch_size] = batch_data_gpu.cpu().numpy(), batch_indices_gpu.cpu().numpy()
            end = time.time()

            print('Total time, time per point : %.2fs, %.4f ms/pt'%(end-start, (end-start)*1000/self.num_query))
            del w_gpu, query_slice_gpu, prod_gpu, batch_data_gpu, batch_indices_gpu
            torch.cuda.empty_cache()
            return self.getnns(len(shard))
