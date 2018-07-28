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
            x_grad = atn_train.cal_grad_target(x_original, cnn_mnist, target)
            x_adv = atn(x_original, x_grad)
            x_adv_detach = x_adv.detach()
            after_pro = cnn_mnist(x_adv)

            # You can show multiple results in the same picture to compare. You can change the shape yourself
            #plt.subplot(3, 3, select_num)
            #cm.imshow_nofig(x_adv_detach, torch.argmax(after_pro, dim=1).item())
            #plt.show()

            cm.imshow(x_adv_detach, torch.argmax(after_pro, dim=1).item())

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


if __name__ == '__main__':
    #mnist_find(3, 8, 10, 200, 'GatnFC', 'CNN_2')
    ind = np.random.randint(0, 10000, (10, 10))
    grid_plot(ind, 'GatnFC', 'CNN_1')
