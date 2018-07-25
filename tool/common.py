''' this file contains functions that may be used by many people '''

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
tv = torchvision
tc = torch

def load_data(dataset_name ):
    ''' 
    USAGE:  x_data, y_train, x_test, y_test = load_data('mnist')
    INPUT:      a string, naming 'mnist' or 'cifar10'  and a 
                        a self.batch_size, defaultly 100
    RETURN:  X_train, y_train, X_test, y_test
                        returned X_train and X_test are 4D-torch-tensor in shape [ data_size, channel, width, height ]

    This function loads the original data and returns the four variables as shown above.
    Note that user need to put the the dataset under './data/' as done before
    
    CONTRIBUTER: henryliu,07.20
    ''' 
    if dataset_name != 'mnist' and dataset_name != 'cifar10':
        print("unrecognized dataset, da cuo le ba ?")
        return
    directory = 'data/' + dataset_name 
   #  print(directory)
    if dataset_name == 'mnist':
        train_data = tv.datasets.MNIST(root=directory, train=True,transform=tv.transforms.ToTensor(),download=False)
        test_data = tv.datasets.MNIST(root=directory, train=False,transform=tv.transforms.ToTensor(),download=False)
    else:
        train_data = tv.datasets.CIFAR10(root=directory, train=True,transform=tv.transforms.ToTensor(),download=False)
        test_data = tv.datasets.CIFAR10(root=directory, train=False,transform=tv.transforms.ToTensor(),download=False)
    if dataset_name == 'mnist':
        X_train = train_data.train_data
        X_train = X_train.reshape(X_train.shape[0],1, 28, 28)
        y_train = train_data.train_labels
        X_test = test_data.test_data
        X_test = X_test.reshape(X_test.shape[0],1, 28, 28)
        y_test = test_data.test_labels
        X_test = tc.tensor(X_test,dtype=tc.float) / 255
        X_train = tc.tensor(X_train,dtype=tc.float) / 255
        return  X_train,  y_train, X_test, y_test
    elif dataset_name == 'cifar10':
        X_train = train_data.train_data
        y_train = train_data.train_labels
        X_test = test_data.test_data
        y_test = test_data.test_labels
        X_train = X_train.transpose([0,3,1,2])
        X_test = X_test.transpose([0,3,1,2])
        X_test = tc.tensor(X_test,dtype=tc.float) / 255
        X_train = tc.tensor(X_train,dtype=tc.float) / 255
        return X_train,  y_train, X_test, y_test
    return

def imshow(image, label):
    '''
    USAGE: imshow(X_train[0], y_train[0])
    INPUT:  X_train is a 3-D torch-tensor in the shape of [channel, width, height]
                    y_train is a number/str
                    Can be used on both mnist and cifar10
    RETURN: plt the graph with label

    This function plot a single graph of mnist or cifar10

    CONTRIBUTER: henryliu, 07.20
    '''
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    plt.axis('off')
    if image.shape[2] == 28: # if mnist
        plt.imshow(image.reshape(28,28), cmap='gray')
        plt.title('%i' % label, fontsize = 20)
    else: # if cifar10    
        plt.imshow( image.permute(1,2,0) )
        plt.title(classes[label])
    plt.show()

class batch_generator():
    '''
    USAGE: generator = batch_generator( batch_size = 50): 
                    x_batch, y_batch = generator.next_batch(X_train, y_train)
    INPUT: when initialize, takes an int
                    when generating batch, takes X and y
    RETURNS: x_batch 4D-tensor [self.batch_size, channel, width, height]
    
    This function uses static method to count for next batch
    Note that caller is responsible to determine that how many rounds there should be in each epoch!
    For example:
        for epoch in range(total_epoch) :
            for small_round in range( len(y_train)/ self.batch_size )
                X_batch, y_batch = next_batch(X,y, self.batch_size)
                do something here

    CONTRIBUTER: henryliu, 07.20
    '''
    def __init__(self,batch_size=100):
        self.batch_size = batch_size
        self.static_counter = 0
    def next_batch(self, X, y):
        if self.static_counter==None:
            self.static_counter = 0
        data_size = len(y)
        if ( self.static_counter+1 ) * self.batch_size >= data_size:
            self.static_counter = 0
            return X[ data_size - self.batch_size: ], y[data_size - self.batch_size : ]
        else:
            self.static_counter += 1
            start, end = self.batch_size * ( self.static_counter -1 ) , self.batch_size * self.static_counter
            return X[ start: end], y[start: end]
