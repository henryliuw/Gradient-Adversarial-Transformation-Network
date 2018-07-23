# print('import tool success')

import torch as tc
import torchvision as tv
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def load_train_data():
    '''
    USAGE: 直接使用，返回MNIST的训练集
    CONTRIBUTOR: 张博皓,2018/7/21
    '''
    train_data = tv.datasets.MNIST(root='data/mnist', train=True,
                                            transform=tv.transforms.ToTensor(),
                                            download=True)
    return train_data
    
def load_test_data():
    '''
    USAGE: 直接使用，返回MNIST的验证集
    CONTRIBUTOR: 张博皓,2018/7/21
    '''
    test_data = tv.datasets.MNIST(root='data/mnist', train=False,
                                            transform=tv.transforms.ToTensor(),
                                            download=True)
    return test_data

def load_cnn():
    '''
    USAGE: 直接使用，返回已经训练好的神经网络
    CONTRIBUTOR: 张博皓,2018/7/23
    '''
    return tc.load('mnist_cnn_model.pkl')


def imshow(instance, label):
    '''
    INPUT: 包含图片信息的一个numpy数组（28*28）,图片对应的数字
    USAGE: 输出MNIST的一张图片
    CONTRIBUTOR: 张博皓,2018/7/21
    '''
    plt.imshow(instance.reshape(28,28), cmap='gray')
    plt.title(label, fontsize = 20)
    plt.show()

