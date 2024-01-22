from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from absl import app
# from absl import flags

import math
import numpy as np

import torch

np.random.seed(10)

def eigen_by_lanczos(mat, proj_dims):
        
        T, V = Lanczos(mat, 128)
        T_evals, T_evecs = np.linalg.eig(T)
        idx = T_evals.argsort()[-1 : -(proj_dims+1) : -1]
        Vk = np.dot(V.T, T_evecs[:,idx])

        # T, V = Lanczos_torch(mat, 128)
        # T_evecs, T_evals, _  = torch.svd(T)
        # # print("T_evecs.shape", T_evecs.shape)   #128*128
        # # print("T_evals.shape", T_evals.shape)   # 128 
        # idx = T_evals.argsort(descending=True)[0:proj_dims-1]
        # Vk = torch.matmul(V.T, T_evecs[:,idx])
        return Vk

def Lanczos_torch(mat, m=128):
    
    # reference: https://en.wikipedia.org/wiki/Lanczos_algorithm
    n = mat[0].shape[0]
    v0 = torch.rand(n,1).cuda()
    v0 /= torch.sqrt(torch.mm(v0.T,v0))

    # v0 = np.random.rand(n)
    # v0 /= np.sqrt(np.dot(v0,v0))

    V = torch.zeros( (m,n) ).cuda()
    T = torch.zeros( (m,m) ).cuda()
    V[0] = v0.squeeze()
    # V = np.zeros( (m,n) )
    # T = np.zeros( (m,m) )
    # V[0, :] = v0

    # step 2.1 - 2.3   //A=mat.T*mat  := dot(dot(matT,mat),v0)
    #print("V[0,:]",V[0,:].shape)
    temp = [torch.matmul(col.reshape(-1,1), torch.matmul(col.reshape(1,-1), V[0].reshape(-1,1))) for col in mat]
    temp = torch.tensor([item.cpu().detach().numpy() for item in temp]).cuda()
    w = torch.sum(temp, dim=0)
    alfa = torch.matmul(w.T, V[0].reshape(-1,1))
    w = w - (alfa * V[0,:]).T
    #print("w",w.shape)
    T[0,0] = alfa

    for j in range(1, m-1):
        beta = torch.sqrt(torch.matmul(w.T,w))
        #beta = np.sqrt( np.dot( w, w ) )
        # print("w",w.shape)
        # print("V[j,:]",V[j,:].shape)
        V[j,:] = (w/beta).squeeze()

        # This performs some rediagonalization to make sure all the vectors
        # are orthogonal to eachother
        for i in range(j-1):
            V[j, :] = V[j,:] - torch.matmul(torch.conj(V[j,:]).reshape(1,-1), V[i, :].reshape(-1,1))*V[i,:]
            # V[j, :] = V[j,:] - np.dot(np.conj(V[j,:]), V[i, :])*V[i,:] #斯密特正交化 v0为向量 分母因为原本dot（v0，v0）ot np.linalg.norm(v0)=||v0||2 or v0.T,v0  or  <v0,v0> = 1
        V[j, :] = V[j, :]/torch.norm(V[j, :])
        # V[j, :] = V[j, :]/np.linalg.norm(V[j, :])
        temp = [torch.matmul(col.reshape(-1,1), torch.matmul(col.reshape(1,-1), V[j,:].reshape(-1,1))) for col in mat]
        temp = torch.tensor([item.cpu().detach().numpy() for item in temp]).cuda()
        w = torch.sum(temp, 0)
        alfa = torch.matmul(w.T, V[j, :].reshape(-1,1))
        w = w - (alfa * V[j, :]).T - (beta*V[j-1, :]).T
        # w = np.sum([np.dot(col, np.dot(col.T, V[j,:])) for col in mat], 0)
        # alfa = np.dot(w, V[j, :])
        # w = w - alfa * V[j, :] - beta*V[j-1, :]

        T[j,j  ] = alfa
        T[j-1,j] = beta
        T[j,j-1] = beta
    
    return T, V


def Lanczos(mat, m=128):

    mat = mat.cpu()
    # reference: https://en.wikipedia.org/wiki/Lanczos_algorithm
    n = mat[0].shape[0]
    v0 = np.random.rand(n)
    v0 /= np.sqrt(np.dot(v0,v0))
    
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    V[0, :] = v0
    
    # step 2.1 - 2.3   //A=mat.T*mat  := dot(dot(matT,mat),v0)
    w = np.sum([np.dot(col, np.dot(col.T, V[0,:])) for col in mat], 0)
    alfa = np.dot(w, V[0,:])
    w = w - alfa * V[0,:]
    T[0,0] = alfa

    # needs to start the iterations from indices 1
    for j in range(1, m-1):
        
        beta = np.sqrt( np.dot( w, w ) )
        V[j,:] = w/beta

        # This performs some rediagonalization to make sure all the vectors
        # are orthogonal to eachother
        for i in range(j-1):
            V[j, :] = V[j,:] - np.dot(np.conj(V[j,:]), V[i, :])*V[i,:] #斯密特正交化 v0为向量 分母因为原本dot（v0，v0）ot np.linalg.norm(v0)=||v0||2 or v0.T,v0  or  <v0,v0> = 1
        V[j, :] = V[j, :]/np.linalg.norm(V[j, :])

        w = np.sum([np.dot(col, np.dot(col.T, V[j,:])) for col in mat], 0)
        alfa = np.dot(w, V[j, :])
        w = w - alfa * V[j, :] - beta*V[j-1, :]

        T[j,j  ] = alfa
        T[j-1,j] = beta
        T[j,j-1] = beta
    
    return T, V


    