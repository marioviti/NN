import torch
import numpy as np
from tme6 import CirclesData
from torch.autograd import Variable

def loss_accuracy(Yhat, Y):
    L = - torch.mean(Y * torch.log(Yhat))

    _, indYhat = torch.max(Yhat, 1)
    _, indY = torch.max(Y, 1)

    acc = torch.sum(indY == indYhat) #* 100 / indY.size(0);
    acc = float(acc.data[0]) * 100./indY.size(0)

    return L, acc

def init_params_auto(nx, nh, ny):
    params = {}
    params['Wh'] = Variable(torch.randn(nh, nx), requires_grad=True)
    params['bh'] = Variable(torch.zeros(nh, 1), requires_grad=True)
    params['Wy'] = Variable(torch.randn(ny, nh) * 0.3, requires_grad=True)
    params['by'] = Variable(torch.zeros(ny, 1),requires_grad=True)
    return params

def forward(params, X):
    bsize = X.size(0)
    nh = params['Wh'].size(0)
    ny = params['Wy'].size(0)
    outputs = {}
    outputs['X'] = X
    outputs['htilde'] = torch.mm(X, params['Wh'].t()) + params['bh'].t().expand(bsize, nh)
    outputs['h'] = torch.tanh(outputs['htilde'])
    outputs['ytilde'] = torch.mm(outputs['h'], params['Wy'].t()) + params['by'].t().expand(bsize, ny)
    outputs['yhat'] = torch.exp(outputs['ytilde'])
    outputs['yhat'] = outputs['yhat'] / (outputs['yhat'].sum(1, keepdim=True)).expand_as(outputs['yhat'])
    return outputs['yhat'], outputs

def sgd(params, eta=0.05):
    params['Wy'].data -= eta * params['Wy'].grad.data
    params['Wh'].data -= eta * params['Wh'].grad.data
    params['by'].data -= eta * params['by'].grad.data
    params['bh'].data -= eta * params['bh'].grad.data
    return params

data = CirclesData()
data.plot_data()

# init
N = data.Xtrain.shape[0]
Nbatch = 20
nx = data.Xtrain.shape[1]
nh = 10
ny = data.Ytrain.shape[1]
eps = 30
params = init_params_auto(nx, nh, ny)


for iteration in range(20):

    perm = torch.randperm(N)
    Xtrain = data.Xtrain[perm]
    Ytrain = data.Ytrain[perm]

    # batches
    for j in range(N // Nbatch):
        X = Xtrain[perm[j * Nbatch:(j+1) * Nbatch]]
        Y = Ytrain[perm[j * Nbatch:(j+1) * Nbatch]]
        Yhat, outputs = forward(params, Variable(X, requires_grad=False))
        L, _ = loss_accuracy(Yhat, Variable(Y, requires_grad=False))
        L.backward()
        params = sgd(params, 0.03)

    Yhat_train, _ = forward(params, Variable(data.Xtrain, requires_grad=False))
    Yhat_test, _ = forward(params, Variable(data.Xtest, requires_grad=False))
    Ltrain, acctrain = loss_accuracy(Yhat_train, Variable(data.Ytrain, requires_grad=False))
    Ltest, acctest = loss_accuracy(Yhat_test, Variable(data.Ytest, requires_grad=False))
    Ygrid, _ = forward(params, Variable(data.Xgrid, requires_grad=False))

    title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain.data[0], acctest, Ltest.data[0])
    #print(title)
    data.plot_data_with_grid(Ygrid.data, title)
    L_train = Ltrain.data[0]
    L_test = Ltest.data[0]
    data.plot_loss((Ltrain.data[0]), (Ltest.data[0]), acctrain, acctest)































#
