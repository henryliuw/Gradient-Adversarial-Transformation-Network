#import('import onePixel success')

import torch as tc
import torchvision as tv
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import tool
import sys

def perturbe(X_test, y_test, scale=0.5, population=400, iteration=100):
    '''
    INPUT: 1.一张需要攻击的图像，维度必须如（1, 1，28，28）
           2.这组图像的真实类别，
           3.缩放因子scale，默认为0.5
           4.演化种群数量，默认为400
           5.迭代次数，默认为100
    USAGE: 对输入的样本进行单像素攻击，返回攻击后的图片
    CONTRIBUTOR: 张博皓,2018/7/23
    '''
    cnn = tool.load_cnn()
    
    total      = X_test.size(0)
    iteration  = 200
    population = 400
    maxnorm    = -1
    survivor   = [0,0,0]
    
    candidates = np.hstack((np.random.randint(low=0, high=27, size=(population,2)),np.random.rand(population,1)))
    
    np.random.seed(8888)
    for num in range(total):
        for generation in range(iteration):
            r = np.random.randint(low=0, high=population-1, size=(population,3))
            next_candidates = np.zeros((population,3))
            
            for i in range(population):
                next_candidates[i] = candidates[r[i][0]] + scale * (candidates[r[i][1]] + candidates[r[i][2]])
            
            for i in range(population):
                if(0<=next_candidates[i][0]<=27 and 0<=next_candidates[i][1]<=27):
                    pixel = X_test[0][0][int(candidates[i][0])][int(candidates[i][1])]
                    X = X_test.clone()
                    X[0][0][int(candidates[i][0])][int(candidates[i][1])] = tc.min(tc.tensor([tc.max(tc.tensor([pixel + candidates[i][2], 0])), 1]))
                    
                    pixel = X_test[0][0][int(next_candidates[i][0])][int(next_candidates[i][1])]
                    X_next = X_test.clone()
                    X_next[0][0][int(next_candidates[i][0])][int(next_candidates[i][1])] = tc.min(tc.tensor([tc.max(tc.tensor([pixel + next_candidates[i][2], 0])), 1]))
    
                    y = cnn(X)
                    y = y - tc.min(y.data)
                    y = y / tc.sum(y.data)
                    
                    y_next = cnn(X_next)
                    y_next = y_next - tc.min(y_next.data)
                    y_next = y_next / tc.sum(y_next.data)
                    
                    y_norm = tc.norm(y[0][y_test[0].item()]-1)
                    y_next_norm = tc.norm(y_next[0][y_test[0].item()]-1)
                    
                    if(y_norm < y_next_norm):
                        candidates[i] = next_candidates[i]
                        y_norm = y_next_norm
                        
                    if(y_norm > maxnorm):
                        maxnorm = y_norm
                        survivor = candidates[i]

        X_adversarial = X_test.clone()
        X_adversarial[0][0][int(survivor[0])][int(survivor[1])] = tc.min(tc.tensor([tc.max(tc.tensor([X_test[0][0][int(survivor[0])][int(survivor[1])] + survivor[2], 0])), 1]))
        outputs = cnn(X_adversarial)
        _, predicted = tc.max(outputs.data, 1)
    
        fig = plt.figure()
        plt.subplot(1,2,1)
        tool.imshow(X_test[0].detach().numpy(),'orignial '+str(y_test[0].item()))
        plt.subplot(1,2,2)
        tool.imshow(X_adversarial[0].detach().numpy(),'adversarial '+str(predicted[0].item()))
        plt.show()
    
    return X_adversarial

    

    