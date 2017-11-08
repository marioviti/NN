import torch
import numpy as np
from tme6 import CirclesData
from torch.autograd import Variable

def loss_accuracy(Yhat, Y):
    L = - torch.mean(Y * torch.log(Yhat))

    _, indYhat = torch.max(Yhat, 1)
    _, indY = torch.max(Y, 1)

    acc = torch.sum(indY == indYhat)
    acc = float(acc.data[0]) * 100./indY.size(0)

    return L, acc

def init_model(nx,nh,ny):
    model = torch.nn.Sequential(
        torch.nn.Linear(nx,nh),
        torch.nn.Tanh(),
        torch.nn.Linear(nh,ny)
    )
    loss = torch.nn.CrossEntropyLoss()
    return model, loss

def sgd(model,eta=0.03):
    for param in model.parameters():
        param.data -= eta*param.grad.data
    model.zero_grad()

data = CirclesData()
Y = data.Ytrain

# init
N = data.Xtrain.shape[0]
Nbatch = 20
nx = data.Xtrain.shape[1]
nh = 5
ny = data.Ytrain.shape[1]
model,loss = init_model(nx,nh,ny)
softmax = torch.nn.Softmax()

for iteration in range(20):

    perm = torch.randperm(N)
    Xtrain = data.Xtrain[perm]
    Ytrain = data.Ytrain[perm]

    # batches
    for j in range(N // Nbatch):
        X = Xtrain[perm[j * Nbatch:(j+1) * Nbatch]]
        Y = Ytrain[perm[j * Nbatch:(j+1) * Nbatch]]
        _,Yind = torch.max(Y, 1)
        Yhat = model(Variable(X, requires_grad=False))
        L = loss(Yhat, Variable(Yind, requires_grad=False))
        L.backward()
        sgd(model)

    Yhat_train = model(Variable(data.Xtrain, requires_grad=False))
    Yhat_test = model(Variable(data.Xtest, requires_grad=False))
    Ltrain, acctrain = loss_accuracy(Yhat_train, Variable(data.Ytrain, requires_grad=False))
    Ltest, acctest = loss_accuracy(Yhat_test, Variable(data.Ytest, requires_grad=False))
    Ygrid = model(Variable(data.Xgrid, requires_grad=False))
    Ygrid = softmax(Ygrid)

    title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain.data[0], acctest, Ltest.data[0])
    data.plot_data_with_grid(Ygrid.data, title)
    data.plot_loss((Ltrain.data[0]), (Ltest.data[0]), acctrain, acctest)
