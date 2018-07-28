import torch
import matplotlib.pyplot as plt
import atn.ATN as ATN
import tool.common as cm
import mnist_attak.train_cnn
import atn_train
import numpy as np
sigmoid_norm = atn_train.sigmoid_norm
CNN_1 = mnist_attak.train_cnn.CNN_1
CNN_2 = mnist_attak.train_cnn.CNN_2
nn = torch.nn
_, __, x_test, y_test = cm.load_data('mnist')
del _
del __
import gc
gc.collect()


def mnist_find(target, origin, num_pic, batchsize, attacker, mnist_net):
    '''
    This function will search in a number of batchsize of pictures started from random place in the 10000 test samples
    for a number of num_pic pictures who was originally classified as origin but classified as target latter on

    :param target: the target class
    :param origin: the original class
    :param num_pic: number of pictures you want to show at the same time
    :param batchsize: this is to prevent your computer from exploding
    :param attacker: the attacker atn net, can only be 'GatnFC' or 'GatnConv'
    :param mnist_net: the net you want to attack, can only be 'CNN_1' or 'CNN_2'
    :return: an array of index of the shown pictures
    '''

    # loading 
    CNN_FILE_PATH = 'data/mnist_'+mnist_net+'_model_params.pkl'
    if mnist_net == 'CNN_1':
        cnn_mnist = CNN_1()
    elif mnist_net == 'CNN_2':
        cnn_mnist = CNN_2()
    else:
        print('No such net for recognition on MNIST')
        print('Possible choices: CNN_1, CNN_2')
        return

    cnn_mnist.load_state_dict(torch.load(CNN_FILE_PATH))
    
    ATN_FILE_PATH = 'data/'+attacker+'_mnist'+mnist_net+'_target'+str(target)+'.parameter'
    if attacker == 'GatnFC':
        atn = ATN.GATN_FC()
    elif attacker == 'GatnConv':
        atn = ATN.GATN_Conv()
    else:
        print('No such atn')
        print('Possible choices: GatnFC, GatnConv')
        return

    atn.load_state_dict(torch.load(ATN_FILE_PATH))

    start_idx = np.random.randint(0, 9500)
    y_origin = cnn_mnist(x_test[start_idx:start_idx+batchsize])

    idx = []
    select_num = 0
    for index, i in enumerate(range(start_idx, start_idx+batchsize)):
        y_label = torch.argmax(y_origin[index]).item()
        if select_num == num_pic:
            break
        if y_label == origin:
            idx.append(i)
            select_num += 1
            x_original = x_test[i].reshape(1, 1, 28, 28)

            plt.subplot(1, 3, 1)
            before_pro = cnn_mnist(x_original)
            imshow_nofig(x_original, torch.argmax(before_pro, dim=1).item())

            x_grad = atn_train.cal_grad_target(x_original, cnn_mnist, target)
            x_adv = atn(x_original, x_grad)
            x_adv_detach = x_adv.detach()
            after_pro = cnn_mnist(x_adv)
            plt.subplot(1, 3, 2)
            imshow_nofig(x_adv_detach, torch.argmax(after_pro, dim=1).item())

            plt.subplot(1, 3, 3)
            imshow_nofig(x_adv_detach - x_original, 'index: '+str(i))
            plt.show()

    return idx


def grid_plot(pos, attacker, mnist_net):
    CNN_FILE_PATH = 'data/mnist_' + mnist_net + '_model_params.pkl'
    if mnist_net == 'CNN_1':
        cnn_mnist = CNN_1()
    elif mnist_net == 'CNN_2':
        cnn_mnist = CNN_2()
    else:
        print('No such net for recognition on MNIST')
        print('Possible choices: CNN_1, CNN_2')
        return

    cnn_mnist.load_state_dict(torch.load(CNN_FILE_PATH))

    plt.axis('off')
    grid = np.zeros((10*28, 10*28))
    for i in range(10):
        ATN_FILE_PATH = 'data/' + attacker + '_mnist' + mnist_net + '_target' + str(i) + '.parameter'
        if attacker == 'GatnFC':
            atn = ATN.GATN_FC()
        elif attacker == 'GatnConv':
            atn = ATN.GATN_Conv()
        else:
            print('No such atn')
            print('Possible choices: GatnFC, GatnConv')
            return

        atn.load_state_dict(torch.load(ATN_FILE_PATH))
        for j in range(10):
            if i != j:
                x_original = x_test[pos[i, j]].reshape(1, 1, 28, 28)
                x_grad = atn_train.cal_grad_target(x_original, cnn_mnist, i)
                x_adv = atn(x_original, x_grad)
                x_adv_detach = x_adv.detach()
                grid[i*28:(i+1)*28, j*28:(j+1)*28] = x_adv_detach
    plt.imshow(grid, cmap='gray')
    plt.show()


def imshow_nofig(image, label):
    '''
    USAGE: imshow(X_train[0], y_train[0])
    INPUT:  X_train is a 3-D or 4-D torch-tensor in the shape of [channel, width, height] or [1, channel, width, height]
                    y_train is a number/str
                    Can be used on both mnist and cifar10
    RETURN: plt the graph with label

    This function plot a single graph of mnist or cifar10 with out plt.show()
    '''
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    plt.axis('off')
    image = image.reshape(-1, image.size(2), image.size(2) )
    if image.shape[1] == 28:  # if mnist
        plt.imshow(image.reshape(28, 28), cmap='gray')
        if isinstance(label, int):
            plt.title('%s' % str(label), fontsize=20)
        else:
            plt.title('%s' % label, fontsize=20)
    else:  # if cifar10
        plt.imshow(image.permute(1, 2, 0))
        plt.title(classes[label])
    #plt.show()


if __name__ == '__main__':
    print(mnist_find(1, 5, 10, 200, 'GatnFC', 'CNN_2'))

    #ind = np.random.randint(0, 10000, (10, 10))
    #grid_plot(ind, 'GatnFC', 'CNN_1')
