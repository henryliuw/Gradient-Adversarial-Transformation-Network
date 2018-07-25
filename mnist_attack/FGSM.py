#print('import FGSM success')

import torch as tc
import torchvision as tv
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import tool
import train_cnn

'''
https://arxiv.org/abs/1312.6199
'''

def perturbe(X_test, y_test, noise_rate=0.1, printImage=True, saveImage=False, savePath=''):  
    '''
    INPUT: 1.一组需要攻击的图像，维度必须如（*，1，28，28）
           2.这组图像的真实类别，
           3.噪音比率参数noise_rate，默认为0.1
           4.是否打印本组图片printImage，默认为打印
           5.是否保存本组图片saveImage，默认为不保存
           6.保存图片的路径savePath，默认为当前文件夹，若输入请保证存在此路径并最后有'/'
    USAGE: 在训练好的模型上进行FGSM攻击，返回输入图片对应的噪声图片，打印攻击前后CNN网络的识别成功率，并打印输入图片与对应的攻击图片
    CONTRIBUTOR: 张博皓 2018/7/23
    '''

    total = y_test.size(0)
    cnn = tool.load_cnn_1()
    X_test = tc.tensor(X_test,requires_grad=True)
    output = cnn(X_test)
    _, predicted = tc.max(output.data, 1)
    
    correct = (predicted == y_test).sum().item()
    accuracy = correct/total
    print('|before adversarial, test accuracy: %.4f%%' % (accuracy*100))
    
    loss_func=tc.nn.CrossEntropyLoss()
    loss = loss_func(output,y_test)
    loss.backward()
    noise = noise_rate*tc.sign(X_test.grad.data)
    
    X_adversarial_test = X_test + noise
    outputs = cnn(X_adversarial_test)
    _, predicted = tc.max(outputs.data, 1)
    
    correct = (predicted == y_test).sum().item()
    accuracy = correct/total
    print('|after adversarial, test accuracy: %.4f%%' % (accuracy*100))
    
    if(printImage==True or saveImage==True):
        for i in range(total):
            fig = plt.figure()
            plt.subplot(1,2,1)
            tool.imshow(X_test[i].detach().numpy(),'orignial '+str(y_test[i].item()))
            plt.subplot(1,2,2)
            tool.imshow(X_adversarial_test[i].detach().numpy(),'adversarial '+str(predicted[i].item()))
            if(saveImage==True):
                plt.savefig(savePath+'adversarial_'+str(i+1)+'.png')
            if(printImage==True):
                plt.show()
            
    return X_adversarial_test

if __name__ == '__main__':
    X_test = tool.load_test_image()
    y_test = tool.load_test_label()
    perturbe(X_test,y_test,printImage=False)