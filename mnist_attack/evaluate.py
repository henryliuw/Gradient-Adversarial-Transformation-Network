#print('import evaluate success')

import torch as tc
import torchvision as tv
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def outputPoss(cnn, adverserial):
    '''
    INPUT: 网络，攻击样本
    USAGE: 返回预测攻击样本的结果中，十个数字的占比
    CONTRIBUTOR: 张博皓,2018/7/21
    '''
    outputs = cnn(adverserial)
    _, predicted = tc.max(outputs.data, 1)
    poss = np.zeros(10)
    total = adverserial.size(0)
    for i in range(10):
        equal = (predicted == i).sum().item()
        poss[i] = equal/total
    return poss
