# print('import train_cnn success')

import torch as tc
import torchvision as tv
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class CNN_1(nn.Module):
    '''
    定义了一个识别MNIST的CNN网络
    网络结构:
        5*5 Conv -> 3*3 Conv -> 3*3 Conv -> FC -> FC -> FC
    最终识别正确率：98.16%
    '''
    def __init__(self):
        super(CNN_1, self).__init__()
        self.layer1_conv = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (20, 14, 14)
        )
        self.layer2_conv = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=16,      # input height
                out_channels=32,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.layer3_conv = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=32,      # input height
                out_channels=64,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (64, 7, 7)
            nn.ReLU(),  # activation
        )
        self.layer4_linear = nn.Linear(64 * 7 * 7, 32 * 3 * 3)   # fully connected layer, output 10 classes
        self.layer5_linear = nn.Linear(32 * 3 * 3, 64)
        self.layer6_linear = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.layer1_conv(x)
        x = self.layer2_conv(x)
        x = self.layer3_conv(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.layer4_linear(x)
        x = x.sigmoid()
        x = self.layer5_linear(x)
        x = x.sigmoid()
        output = self.layer6_linear(x)
        output = output.sigmoid()
        output = output / output.sum(1).view(output.size(0), -1)
        return output

class CNN_2(nn.Module):
    '''
    定义了一个识别MNIST的CNN网络
    网络结构:
        3*3 Conv -> FC -> FC -> FC
    最终识别正确率：95.53%
    '''
    def __init__(self):
        super(CNN_2, self).__init__()
        self.layer1_conv = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (20, 14, 14)
        )
        self.layer2_linear = nn.Linear(16 * 14 * 14, 32 * 32)   # fully connected layer, output 10 classes
        self.layer3_linear = nn.Linear(32 * 32, 16 * 16)
        self.layer4_linear = nn.Linear(16 * 16, 10)
        
    def forward(self, x):
        x = self.layer1_conv(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.layer2_linear(x)
        x = x.sigmoid()
        x = self.layer3_linear(x)
        x = x.sigmoid()
        output = self.layer4_linear(x)
        output = output.sigmoid()
        output = output / output.sum(1).view(output.size(0), -1)
        return output
    
def train_cnn(CNN, batch_size=100, show_step=100):
    '''
    INPUT: 
        1.输入一个需要训练的网络结构
        2.每次批量梯度下降的样本尺寸，默认为100
        3.多少次迭代中显示一步损失函数与测试准确率，默认为100步
    USAGE: 自动训练CNN网络，并保存整个模型到同级文件夹中，名称为：mnist_CNN名字_model.pkl
    CONTRIBUTOR: 张博皓 2018/7/21
    '''
    
    train_data = tv.datasets.MNIST(root='data/mnist', train=True,
                                            transform=tv.transforms.ToTensor(),
                                            download=True)
    test_data = tv.datasets.MNIST(root='data/mnist', train=False,
                                            transform=tv.transforms.ToTensor(),
                                            download=True)
    X_test = tc.tensor(test_data.test_data.reshape(10000,1,28,28),dtype=tc.float) / 255
    y_test = test_data.test_labels
    train_batch = Data.DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
    
    print('=====Train Network '+CNN.__name__+'=====')
    cnn=CNN()
    loss_func=tc.nn.CrossEntropyLoss()
    optimizer=tc.optim.Adam(cnn.parameters(),lr=0.001)
    
    for step, (x, y) in enumerate(train_batch):
        output = cnn(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % show_step == 0:
            outputs = cnn(X_test)
            _, predicted = tc.max(outputs.data, 1)
            total = y_test.size(0)
            correct = (predicted == y_test).sum().item()
            accuracy = correct/total
            print('|Step:', step,'|train loss:%.4f' % loss.data.item(), '|test accuracy:%.4f' % accuracy)
        
    print('=====Train Complete=====')
    
    tc.save(cnn.state_dict(), 'mnist_'+CNN.__name__+'_model_params.pkl')

if __name__ == '__main__' :
    train_cnn(CNN_1)
    train_cnn(CNN_2)
