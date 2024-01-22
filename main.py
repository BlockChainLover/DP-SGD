import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

import os
import argparse
import csv
import random
import time
import numpy as np
#from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

from models import resnet20
from models.cnn import CNN, LeNet
from utils import get_data_loader, get_sigma, restore_param, checkpoint, adjust_learning_rate, process_grad_batch
from main_utils import save_pro

from lanczos import eigen_by_lanczos, Lanczos

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

## general arguments
parser.add_argument('--dataset', default='fmnist', type=str, help='dataset name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='CNN_fmnist', type=str, help='session name')
parser.add_argument('--seed', default=6, type=int, help='random seed')
parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
parser.add_argument('--n_training', default=10000, type=int, help='used training num')
parser.add_argument('--batchsize', default=250, type=int, help='batch size (default=1000)')
parser.add_argument('--n_epoch', default=80, type=int, help='total number of epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')
parser.add_argument('--time', default='03092235', type=str, help='time')
parser.add_argument('--save_dir', default='results/', type=str, help='save path')



## arguments for learning with differential privacy
parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
parser.add_argument('--pcdp', default=False, type=bool, help='enable pcdp-sgd')
parser.add_argument('--spar', default=False, type=bool, help='sparsity')
parser.add_argument('--clip', default=0.01, type=float, help='gradient clipping bound') #0.001
parser.add_argument('--gamma', default=0.0001, type=float, help='value of gamma') #0.001
parser.add_argument('--lr', default= 0.2, type=float, help='base learning rate (default=0.1)')
parser.add_argument('--eps', default=3, type=float, help='privacy parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
parser.add_argument('--dp_type', default='ours', type=str, help='choose types of scaling method')
parser.add_argument('--norm_scale', default=0.7, type=float, help='gradient norm scaling')

parser.add_argument('--proj_k', default=100, type=int, help='top-k dimension for projection')
parser.add_argument('--real_labels', action='store_true', help='use real labels for auxiliary dataset')
parser.add_argument('--aux_dataset', default='fmnist', type=str, help='name of the public dataset, [cifar10, cifar100, imagenet]')
parser.add_argument('--aux_data_size', default=100, type=int, help='size of the auxiliary dataset') #1000
parser.add_argument('--aux_batch_size', default=100, type=int, help='size of the auxiliary dataset')


args = parser.parse_args()

assert args.dataset in ['cifar10', 'svhn', 'mnist', 'fmnist']
assert args.aux_dataset in ['cifar10', 'cifar100', 'imagenet', 'mnist', 'fmnist']
print("real_labels", args.real_labels)
if(args.real_labels):
    assert args.aux_dataset == 'cifar10'

use_cuda = True
best_acc = 0  
accuracy_accountant = []
grad_norm_accountant = []

iter_orggrad_accountant = []
iter_stograd_accountant = []
start_epoch = 0  
batch_size = args.batchsize

if(args.seed != -1): 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

print('==> Preparing data..')
## preparing data for trainning && testing
print("args.dataset:",args.dataset)
if(args.dataset == 'svhn'):  ## For SVHN, we concatenate training samples and extra samples to build the training set.
    trainloader, extraloader, testloader, n_training, n_test = get_data_loader('svhn', batchsize = args.batchsize)
    for train_samples, train_labels in trainloader:
        break
    for extra_samples, extra_labels in extraloader:
        break
    train_samples = torch.cat([train_samples, extra_samples], dim=0)
    train_labels = torch.cat([train_labels, extra_labels], dim=0)

elif(args.dataset == 'cifar10'):
    trainloader, testloader, n_training, n_test = get_data_loader('cifar10', batchsize = args.batchsize)
    train_samples, train_labels = None, None

elif(args.dataset == 'mnist'):
    trainloader, testloader, n_training, n_test = get_data_loader('mnist', batchsize = args.batchsize)
    train_samples, train_labels = None, None
elif(args.dataset == 'fmnist'):
    trainloader, testloader, n_training, n_test = get_data_loader('fmnist', batchsize = args.batchsize)
    train_samples, train_labels = None, None

### public data
args.real_labels = True
print("args.aux_dataset", args.aux_dataset)
print("real_labels", args.real_labels)
num_public_examples = args.aux_data_size
if('cifar' in args.aux_dataset):
    if(args.aux_dataset == 'cifar100'):
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=False, download=False, transform=transform_test)
        public_data_loader = torch.utils.data.DataLoader(testset, batch_size=num_public_examples, shuffle=False, num_workers=2) #
        for public_inputs, public_targets in public_data_loader:
            break
    # cifar10
    else:   
        # testloader 
        # for batch_idx, (inputs, targets) in enumerate(testloader):
        #     if(batch_idx == 0):
        #         public_inputs = inputs
        #         public_targets = targets
    ### trainloader -> public data
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if(batch_idx == 50):
                public_inputs = inputs
                public_targets = targets
            # elif(batch_idx in [1,2,3,4]):
            #     public_inputs = torch.cat((public_inputs, inputs), dim=0)
            #     public_targets = torch.cat((public_targets, targets), dim=0)
elif(args.aux_dataset == 'mnist' or args.aux_dataset == 'fmnist'):
    # for batch_idx, (inputs, targets) in enumerate(testloader):
    #     if(batch_idx == 0):
    #         public_inputs = inputs
    #         public_targets = targets
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if(batch_idx == 50):  #40
            public_inputs = inputs
            public_targets = targets
else:
    public_inputs = torch.load('./data/IMAGENET/imagenet_examples_2000')[:num_public_examples]


if(not args.real_labels):
    public_targets = torch.randint(high=10, size=(num_public_examples,))
public_inputs, public_targets = public_inputs.cuda(), public_targets.cuda()
n_training = args.n_training
print('# of training examples: ', n_training, '# of testing examples: ', n_test, '# of auxiliary examples: ', num_public_examples)
#print("public_targets", public_targets.shape)

print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.eps, args.delta))
sampling_prob=args.batchsize/n_training
n_steps = int(args.n_epoch/sampling_prob)

sigma = 30
noise_multiplier = sigma
print('noise scale: ', noise_multiplier, 'privacy guarantee: ', args.eps)

print('\n==> Creating CNN model instance')
if(args.resume):
    try:
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_file = './checkpoint/' + args.sess  + '.ckpt'
        checkpoint = torch.load(checkpoint_file)
        net = CNN(input_dim=1, output_dim=10)
        #net = LeNet(input_dim=3, output_dim=10)
        net.cuda()
        restore_param(net.state_dict(), checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    except:
        print('resume from checkpoint failed')
else:
    net = CNN(input_dim=1, output_dim=10) 
    #net = LeNet(input_dim=3, output_dim=10)
    net.cuda()

net = extend(net)

num_params = 0
for p in net.parameters():
    num_params += p.numel()

print('total number of parameters: ', num_params/(10**6), 'M')

if(args.private):
    #loss_func = nn.CrossEntropyLoss(reduction='sum')
    loss_func = nn.CrossEntropyLoss()
else:
    loss_func = nn.CrossEntropyLoss(reduction='mean')

loss_func = extend(loss_func)

num_params = 0
np_list = []
for p in net.parameters():
    num_params += p.numel()
    np_list.append(p.numel())

optimizer = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

def train(epoch, Vk):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    #Vk = []
    Vk_rp = []
    pub_grad = []
    sym_accountant = []
    sym_pro_accountant = []

    iter_sto_grad = []
    iter_org_grad = []

    t0 = time.time()
    steps = n_training//args.batchsize

    if(train_samples == None): # using pytorch data loader for CIFAR10
        loader = iter(trainloader)
    else: # manually sample minibatchs for SVHN
        sample_idxes = np.arange(n_training)
        np.random.shuffle(sample_idxes)

    for batch_idx in range(steps):
        # print("batch_idx:", batch_idx)
        if(args.dataset=='svhn'):
            current_batch_idxes = sample_idxes[batch_idx*args.batchsize : (batch_idx+1)*args.batchsize]
            inputs, targets = train_samples[current_batch_idxes], train_labels[current_batch_idxes]
        else:
            inputs, targets = next(loader)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if(args.private):
            logging = batch_idx % 20 == 0
            gap = 1
            ### public projection 
            if batch_idx == 0 and (epoch % gap  == 0 or epoch == 0):
                optimizer.zero_grad()

                # dependent: public data index
                pub_sample_idxes = np.arange(args.aux_data_size)
                np.random.shuffle(pub_sample_idxes)
                pub_batch_idxes = pub_sample_idxes[0 : args.aux_batch_size]
                pub_inputs, pub_targets = public_inputs[pub_batch_idxes], public_targets[pub_batch_idxes]

                outputs = net(pub_inputs)
                loss = loss_func(outputs, pub_targets) 

                with backpack(BatchGrad()):
                    loss.backward()
                    with torch.no_grad():
                        for p in net.parameters():
                            flat_g = p.grad_batch.reshape(args.aux_batch_size, -1) 
                            mean_batch = torch.mean(flat_g, dim=0)
                            #print("mean_batch",mean_batch.shape)
                            pub_grad.append(mean_batch)

            ### origin gradient descent
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            #print("loss-train",loss)
            with backpack(BatchGrad()):
                loss.backward()
                grad_norm_sample = process_grad_batch(list(net.parameters()), args.clip, args.gamma, args.dp_type, args.norm_scale) # clip gradients and sum clipped gradients
                grad_norm_accountant.append(float(grad_norm_sample))

                ## add noise to gradient
                idx_layer = 0
                for p in net.parameters():
                    shape = p.grad.shape
                    #print("shape:", shape)
                    numel = p.grad.numel()

                    #grad_noise = 0
                    #print("p.grad.shape", p.grad.shape)
                    if args.dp_type == 'ours':
                        grad_noise = torch.normal(0, noise_multiplier*args.clip/(args.norm_scale*args.batchsize), size=p.grad.shape, device=p.grad.device)
                    else:
                        grad_noise = torch.normal(0, noise_multiplier*args.clip/args.batchsize, size=p.grad.shape, device=p.grad.device)
                    p.grad.data += grad_noise.reshape(p.grad.shape)

                    idx_layer += 1

            #if args.pcdp: 
            grad_norm_accountant.append(float(grad_norm_sample))    #6.1

        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            try:
                for p in net.parameters():
                    del p.grad_batch
            except:
                pass
        optimizer.step()
        step_loss = loss.item()
        if(args.private):
            step_loss /= inputs.shape[0]
            #print("input.shape:",inputs.shape)
        train_loss += step_loss
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
    t1 = time.time()
    print('Train loss:%.5f'%(train_loss/(batch_idx+1)), 'time: %d s'%(t1-t0), 'train acc:', acc, end=' ')
    return (train_loss/batch_idx, acc, Vk)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            step_loss = loss.item()
            if(args.private):
                step_loss /= inputs.shape[0]

            test_loss += step_loss 
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)
        accuracy_accountant.append(acc)
        print('test loss:%.5f'%(test_loss/(batch_idx+1)), 'test acc:', acc)
        ## Save checkpoint.
        if acc > best_acc:
            best_acc = acc
            checkpoint(net, acc, epoch, args.sess)

    return (test_loss/batch_idx, acc)


print('\n==> Strat training')
Vk = []
for epoch in range(start_epoch, args.n_epoch):
    lr = adjust_learning_rate(optimizer, args.lr, epoch, all_epoch=args.n_epoch)
    train_loss, train_acc, Vk_e = train(epoch, Vk)
    Vk = Vk_e
    test_loss, test_acc = test(epoch)
    save_pro.save_progress(args, accuracy_accountant, grad_norm_accountant)


