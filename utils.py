
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os

import models


import numpy as np
#from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

from rdp_accountant import compute_rdp, get_privacy_spent

from lanczos import Lanczos

def process_grad_batch(params, clipping=0.01, gamma=0.0001,dp_type='ours', norm_scale = 0.7):
    n = params[0].grad_batch.shape[0]
    #print("n:", n)
    #print("len(params):", len(params))
    grad_norm_list = torch.zeros(n).cuda()
    for p in params:  #every layer
        flat_g = p.grad_batch.reshape(n, -1)
        #print("flat_g:", flat_g.shape)
        current_norm_list = torch.norm(flat_g, dim=1)
        #print("current_norm_list.shape:", current_norm_list.shape)
        grad_norm_list += torch.square(current_norm_list)
        #print("grad_norm_list",grad_norm_list.shape)
    grad_norm_list_sq = torch.sqrt(grad_norm_list)
    #print("grad_norm_list.shape", grad_norm_list.shape)
    #print("grad_norm_list-10:",grad_norm_list[0:10])

    #abadi
    if (dp_type == "abadi"):
        scaling = clipping/grad_norm_list_sq
        # scaling[scaling>1] = 1

    # auto clip - bu
    if (dp_type == "bu"):
        scaling = clipping/(grad_norm_list_sq + gamma)

    # auto clip - xia
    if (dp_type == "xia"):
        scaling = clipping/(grad_norm_list_sq + gamma / (grad_norm_list_sq + gamma) )

    # clip-new - huangtao
    if (dp_type == "ours"):
        # scaling =(clipping) / ((grad_norm_list + gamma) +
        #                       (gamma) / (grad_norm_list + gamma))
        scaling = clipping/(norm_scale*grad_norm_list_sq + gamma / (grad_norm_list_sq + gamma) )

    for p in params:
        p_dim = len(p.shape)
        #print("scaling:",scaling.shape)
        scaling = scaling.view([n] + [1]*p_dim)
        #print("scaling-a:",scaling.shape)
        p.grad_batch *= scaling
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)

    return torch.sum(grad_norm_list)/n



def process_layer_grad_batch(params, batch_idx, Vk, clipping=1):
    n = params[0].grad_batch.shape[0]
    #print("n:", n)
    #print("len(params):", len(params))
    grad_norm_list = torch.zeros(len(params), n).cuda()
    idx_layer = 0
    for p in params:  #every layer
        flat_g = p.grad_batch.reshape(n, -1)
        mean_batch = torch.mean(flat_g, dim=0)

        ### random proj
        # if batch_idx == 0:
        #     random_p = np.random.random(size=(flat_g[0].cpu().numpy().size, 1))
        #     Vk_layer, _ = np.linalg.qr(random_p)
        #     Vk.append(Vk_layer)
        # Vk_layer = torch.from_numpy(Vk[idx_layer]).float().cuda()
        # flat_g = torch.matmul(Vk_layer, torch.matmul(Vk_layer.T, flat_g.T)).T
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)

        ### random vec
        # Vk.append(torch.randn(flat_g.shape[1], 1,dtype=torch.float32).cuda())
        # Vk /= torch.norm(Vk)
        # flat_g = torch.matmul(Vk, torch.matmul(Vk.T, flat_g.T)).T
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)
        
        ### sparsify
        # Vk = sparsify(flat_g.shape[1], 0.5)
        # flat_g = torch.mul(flat_g.T, Vk).T
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)
        #print("Vk:", Vk)
        #print("flat_g:", flat_g)

        ### topk- sparsification
        # Vk = topk_sparsify(flat_g, k).float().cuda()
        # flat_g = torch.mul(Vk, flat_g)
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)
        # print("Vk:", Vk.shape)
        # print("flat_g:", flat_g.shape)

        ### pca-lanczos
        # if batch_idx == 0:
        #     Vk_layer = eigen_by_lanczos((flat_g - mean_batch).cpu().numpy(), 1)
        #     Vk.append(Vk_layer)
        # Vk_layer = torch.from_numpy(Vk[idx_layer]).float().cuda()
        # flat_g = torch.matmul(Vk_layer, torch.matmul(Vk_layer.T, flat_g.T)).T + mean_batch
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)
        #print("flat_g:", flat_g.shape)

        ### pca-torch
        # if batch_idx == 0:
        #     Vk_layer, _, _ = torch.linalg.svd( (flat_g - mean_batch).T, full_matrices=False)
        #     Vk.append(Vk_layer[:,0:1])
        #     print("Vk_layer:", Vk_layer[:,0:1].shape)
        # Vk_layer = Vk[idx_layer]
        # flat_g = torch.matmul(Vk_layer, torch.matmul(Vk_layer.T, flat_g.T)).T + mean_batch
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)
        
        ### classic
        current_norm_list = torch.norm(flat_g, dim=1)
        #print("current_norm_layer_list:", current_norm_list[0:10])
        grad_norm_list[idx_layer] += current_norm_list
        idx_layer += 1 
    print("grad_norm_layer_list-10:",grad_norm_list[15,0:10])
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1
    #print("scaling.shape:", scaling.shape)
    
    idx_layer = 0
    for p in params:
        p_dim = len(p.shape)
        #print("scaling:",scaling.shape)
        scaling_layer = scaling[idx_layer].view([n] + [1]*p_dim)
        #print("scaling_layer.shape:", scaling_layer.shape)
        idx_layer += 1
        #print("p.grad_batch:", p.grad_batch)
        p.grad_batch *= scaling_layer
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)

    return grad_norm_list[15,0], Vk


def get_data_loader(dataset, batchsize):
    if(dataset == 'svhn'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.SVHN(root='./data/SVHN',split='train', download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=73257, shuffle=True, num_workers=0) #load full btach into memory, to concatenate with extra data

        extraset = torchvision.datasets.SVHN(root='./data/SVHN',split='extra', download=False, transform=transform)
        extraloader = torch.utils.data.DataLoader(extraset, batch_size=531131, shuffle=True, num_workers=0) #load full btach into memory

        testset = torchvision.datasets.SVHN(root='./data/SVHN',split='test', download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)
        return trainloader, extraloader, testloader, len(trainset)+len(extraset), len(testset)

    if(dataset == 'mnist'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.MNIST(root='./data',train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2) #load full btach into memory, to concatenate with extra data

        testset = torchvision.datasets.MNIST(root='./data',train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)

    elif(dataset == 'fmnist'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data',train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2) #load full btach into memory, to concatenate with extra data

        testset = torchvision.datasets.FashionMNIST(root='./data',train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)

    else:
        transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=False, transform=transform_train) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=False, transform=transform_test) 
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)



def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    previous_eps = eps
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if(rgp):
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break    
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    cur_sigma = init_sigma
    
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps


def restore_param(cur_state, state_dict):
    own_state = cur_state
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)

def sum_list_tensor(tensor_list, dim=0):
    return torch.sum(torch.cat(tensor_list, dim=dim), dim=dim)

def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def checkpoint(net, acc, epoch, sess):
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + sess  + '.ckpt')

def adjust_learning_rate(optimizer, init_lr, epoch, all_epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    decay = 1.0
    if(epoch<all_epoch*0.5):
        decay = 1.
    elif(epoch<all_epoch*0.75):
        decay = 10.
    else:
        decay = 100.

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / decay
    return init_lr / decay
