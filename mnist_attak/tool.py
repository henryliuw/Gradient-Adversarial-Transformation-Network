# print('import tool success')

import torch as tc
import torchvision as tv
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def load_train_data():
    '''
    USAGE: 直接使用，返回MNIST的原始训练集
    CONTRIBUTOR: 张博皓,2018/7/21
    '''
    train_data = tv.datasets.MNIST(root='data/mnist', train=True,
                                            transform=tv.transforms.ToTensor(),
                                            download=True)
    return train_data
    
def load_test_data():
    '''
    USAGE: 直接使用，返回MNIST的原始验证集
    CONTRIBUTOR: 张博皓,2018/7/21
    '''
    test_data = tv.datasets.MNIST(root='data/mnist', train=False,
                                            transform=tv.transforms.ToTensor(),
                                            download=True)
    return test_data

def load_test_image():
    '''
    USAGE: 直接使用，返回MNIST的验证集中的图片数组，维度为(10000,1,28,28)，每个像素点取值为[0,1]
    CONTRIBUTOR: 张博皓,2018/7/23
    '''
    return tc.tensor(load_test_data().test_data.reshape(10000,1,28,28),dtype=tc.float) / 255

def load_test_label():
    '''
    USAGE: 直接使用，返回MNIST的验证集中的图片对应的真实标签
    CONTRIBUTOR: 张博皓,2018/7/23
    '''
    return load_test_data().test_labels

def imshow(instance, label):
    '''
    INPUT: 包含图片信息的一个numpy数组（28*28）,图片对应的数字
    USAGE: 输出MNIST的一张图片
    CONTRIBUTOR: 张博皓,2018/7/21
    '''
    plt.imshow(instance.reshape(28,28), cmap='gray')
    plt.title(label, fontsize = 20)

