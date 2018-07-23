#print('import FGSM success')

import torch as tc
import torchvision as tv
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import tool
import evaluate

def perturbe(noise_rate=0.05, n=0):  
    '''
    INPUT: 输入噪声所占比率，FGSM算法中默认为0.05,输入需要输出图片的个数，默认为0
    USAGE: 在训练好的模型上进行FGSM攻击，输出MNIST验证集中所有图片对应的噪声图片，打印攻击后CNN网络的识别成功率和攻击成功率，并且随机输出n张攻击后的图片与原图片对比
    CONTRIBUTOR: 张博皓 2018/7/21
    '''
    
    test_data = tool.load_test_data()
    
    X_test = test_data.test_data.reshape(10000,1,28,28)
    X_test = tc.tensor(X_test,dtype=tc.float) / 255
    y_test = test_data.test_labels
    
    cnn = tool.load_cnn()

    X_test = tc.tensor(X_test,requires_grad=True)
    output = cnn(X_test)
    loss_func=tc.nn.CrossEntropyLoss()
    loss = loss_func(output,y_test)
    loss.backward()
    noise = noise_rate*tc.sign(X_test.grad.data)

    X_adversial_test = X_test + noise
    y_test = test_data.test_labels

    outputs = cnn(X_adversial_test)
    _, predicted = tc.max(outputs.data, 1)
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    accuracy = correct/total
    success = (0.9813-accuracy)/0.9813
    
    print('|after adversarial, test accuracy:%.4f%%' % (accuracy*100))
    print('|adversarial success rate:%.4f%%' % (success*100))
    
    imnum=np.random.randint(low=0, high=9999, size=n, dtype='l')
    
    for i in range(n):
        plt.subplot(n,2,2*i+1)
        tool.imshow(X_test[imnum[i]].detach().numpy(),'orignial')
        plt.subplot(n,2,2*i+2)
        tool.imshow(X_adversial_test[imnum[i]].detach().numpy(),'adversarial')

