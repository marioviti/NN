import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

datasets.CIFAR10.url = "http://webia.lip6.fr/~robert/cours/rdfia/cifar-10-python.tar.gz"
path = "../Datasets/"


def cov_mat(X):
    N,H,W,C = X.shape
    X_reshape = X.reshape(N*H*W,C)
    mu = np.mean(X_reshape,0)
    xmu = X_reshape - mu
    sigma = np.dot(np.transpose(xmu),xmu)/float(N*H*W)
    return sigma,mu

def sample_base(X,n_samples=100):
    N,H,W,C = X.shape
    sampling = np.random.permutation(range(0,N))[:n_samples]
    x_sample = X[sampling]
    return x_sample

def ZCA_withening_matrix(X,n_samples=4500):
    N,H,W,C = X.shape
    n_samples = N if N<n_samples else n_samples
    x_sample = sample_base(X,n_samples)
    sigma,mu = cov_mat(x_sample)
    U,S,V = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZCA_mat = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    return ZCA_mat

def ZCA_withening_dataset(X,ZCA_mat):
    N,H,W,C = X.shape
    X_ravel = X.reshape(N*H*W,C)
    X_withened = np.dot(X_ravel,ZCA_mat).reshape(N,H,W,C)
    return X_withened

def ZCA_withen_image(img,ZCA_mat):
    H,W,C = img.shape
    img_reshaped = img.reshape(H*W,C).transpose()
    image_withened = np.dot(ZCA_mat,img_reshaped).transpose()
    return image_withened.reshape(H,W,C)

def ZCA_whitening_functional(tensor,ZCA_mat):
    img = tensor.numpy().transpose()
    whitened = ZCA_withen_image(img,ZCA_mat)
    whitened_tensor = torch.from_numpy(whitened.transpose()).type(torch.FloatTensor)
    return whitened_tensor

def toImage(data):
    min_ = np.min(data)
    max_ = np.max(data)
    return (data-min_)/(max_ - min_)

class ZCA_whitening(object):

    def __init__(self, ZCA_mat):
        self.ZCA_mat = ZCA_mat

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Whithened Tensor image.
        """
        return ZCA_whitening_functional(tensor, self.ZCA_mat)

# %% Compute ZCA_mat

def test():
    train_dataset = datasets.CIFAR10(path, train=True, download=True)
    X = train_dataset.train_data/255.
    print(X.shape)

    x_sample = sample_base(X)

    print(x_sample.shape)
    ZCA_mat = ZCA_withening_matrix(X)

    print(ZCA_mat.shape)


    # %% Withen image with ZCA_mat

    #x_withened = ZCA_withening(x_sample,ZCA_mat)
    x_withened = ZCA_withening_dataset(X,ZCA_mat)
    x_withened = toImage(x_withened[9])
    plt.imshow(x_withened)
    plt.show()

if __name__ == '__main__':

    train_dataset = datasets.CIFAR10(path, train=True, download=True)
    X = train_dataset.train_data/255.
    print(X.shape)

    x_sample = sample_base(X)

    print(x_sample.shape)
    ZCA_mat = ZCA_withening_matrix(X)

    print(ZCA_mat.shape)


    # %% Withen image with ZCA_mat

    x_withened = ZCA_withening_dataset(X,ZCA_mat)
    x_withened_img = toImage(x_withened[10])
    print(x_withened_img)
    plt.imshow(x_withened_img)
    plt.show()
    cov_mat = cov_mat(x_withened)
    x_withened_img = toImage(x_withened[10])
    plt.imshow(x_withened_img)
    plt.show()






















#
