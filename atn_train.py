'''
USAGE: python atn_train.py 
Note that you need to specify ALL the parameter for the main function!
Please refer to the parameter of main!
Please run this file after you understand all parameters in the main function

Training on 1060GTX takes about 15min/epoch
This file trains basic ATN. By now only GATN_FC on mnist is written.

CONTRIBUTER: henryliu, 07.23
'''

import torch
import atn.ATN as ATN
import tool.common as cm
import mnist_attak.train_cnn
CNN_1 = mnist_attak.train_cnn.CNN_1
CNN_2 = mnist_attak.train_cnn.CNN_2
traincnn = mnist_attak.train_cnn
nn = torch.nn


def lossY_fn(y_now,  target):
    '''
        USAGE: returns MSE loss between y_now and target
        all y should be [D, 10] tensor, target is proposed class
        target should be an int
    '''
    #y_now = sigmoid_norm(y_now)
    #y_origin = sigmoid_norm(y_origin)
    #y_reranked_target = reranking(y_origin, target, alpha)
    y_target = torch.zeros_like(y_now)
    y_target[:, target] = 1
    # print(y_target)
    #KLloss_fn = nn.KLDivLoss()
    #lossY = KLloss_fn(torch.log(y_now), y_target)
    MSELoss_fn = nn.MSELoss()
    lossY = MSELoss_fn(y_now, y_target)
    return lossY


def lossY_fn_untarget(y_now, y_origin):
    '''
    DO NOT USE THIS
    '''
    # xentropy_loss_fn = nn.CrossEntropyLoss()
    # xentropy = xentropy_loss_fn(y_now, y_label)
    #
    #y_now = sigmoid_norm(y_now)
    #y_origin = sigmoid_norm(y_origin)
    #KLloss_fn = nn.KLDivLoss()
    #lossY = KLloss_fn(torch.log(y_now), y_origin)
    MSEloss_fn = nn.MSELoss()
    lossY = MSEloss_fn(y_now, y_origin)
    return - lossY


def sigmoid_norm(y):
    '''
    USAGE: y must be a [D,class_size] size tensor
    This function gives whatever input and norm it to possibility using sigmoid_norm
    '''
    y = y.sigmoid()
    y = y / y.sum(1).view(y.size(0), -1)
    return y


def reranking(y_origin, target, alpha):
    '''
    DO NOT USE THIS
    original reranking function in paper
    use sum() to norm the probability distribution
    '''
    max_n, _ = torch.max(y_origin, 1)
    weight_mat = torch.ones_like(y_origin)
    weight_mat[:, target] = alpha
    y_origin[:, target] = max_n
    result = weight_mat * y_origin
    result = result / result.sum(1).view(y_origin.size(0), -1)
    return result


def accuracy(y_pred, y_label, target):
    '''
    USAGE: returns the accuracy/target_rate based on y_pred in shape [D, 10] and y_label in shape [D].
    '''
    _, predicted = torch.max(y_pred, 1)
    total = y_label.size(0)
    correct = (predicted == y_label).sum().item()
    accuracy = correct/total
    non_target_idx = (y_label != target)
    targetotal = (predicted[non_target_idx] == target).sum().item()
    targetrate = targetotal / non_target_idx.sum().item()
    return accuracy, targetrate


def cal_grad_target(X, cnn_model, target):
    '''
    USAGE: This function calculate target gradient with respect to target output probability
    (instead of loss) 
    (they are actually the same)
    '''
    x_image = X.detach()
    x_image.requires_grad_(True)
    out = cnn_model(x_image)
    target_out = out[:, target]
    target_out.backward(torch.ones_like(target_out))
    return x_image.grad


def cal_grad_untarget(X, cnn_model, y_label):
    '''
    DO NOT USE THIS
    '''
    x_image = X.detach()
    x_image.requires_grad_(True)
    y_pred = cnn_model(x_image)
    xentropy_loss_fn = nn.CrossEntropyLoss(reduce=False)
    xentropy = xentropy_loss_fn(y_pred, y_label)
    xentropy.backward(torch.ones_like(xentropy))
    return x_image.grad


def main(beta=3, SAVE_PATH='data/GatnFC_mnistConv_target6.parameter', atn='fc', classifier='1' ,target=6, classifier_PATH='data/mnist_cnn_model_lhy.pkl', epoch_n=1, batch_size=50, continue_training=False):
    '''
    USAGE: all paremeter must be specified, the training use GPU if possible
        atn: the GATN network type you wish to train, should be one of 'fc' or 'conv'
        classifer: the target CNN we with to attach, should be '1' or '2'
        target: the target class, an int
        classifier_PATH: the classifier model path, note the file should be parameter save
        SAVE_PATH: saved file name, note the model is saved as parameter
        beta: loss weight between lossX and lossY
        epoch_n: total epoch, 1 is generally enough
        batch_size: the batch_size
        continue_training: whether we wish to train the model based on model. Note that the checkpoint model should be the SAVE_PATH model
    '''
    # loading data, architecture
    # beta=3 is good for fc on CNN1
    # beta=5 is good for fc on CNN2
    # beta=1 is good for Conv on CNN1
    # beta=0.3  is good for Conv on CNN2
    print("beta=", beta)
    x_train, y_train, x_test, y_test = cm.load_data('mnist')
    # cleaning data
    idx = (y_train != target)
    x_train = x_train[idx]
    y_train = y_train[idx]  
    if atn == 'fc':
        atn = ATN.GATN_FC(beta=beta)
    if atn == 'conv':
        atn = ATN.GATN_Conv(beta=beta)
    lossX_fn = nn.MSELoss()  # loss between X_origin and X_after
    if classifier == '1':
        cnn_mnist = CNN_1()
    elif classifier == '2':
        cnn_mnist = CNN_2()
    cnn_mnist.load_state_dict(torch.load(classifier_PATH))

    #continue training
    if continue_training:
        print("continue training!")
        atn.load_state_dict(torch.load(SAVE_PATH))

    # training parameter/using GPU is possible
    device = device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    iteration_n = int(len(y_train)/batch_size)
    batch_generator = cm.batch_generator(batch_size)
    optimizer = torch.optim.Adam(atn.parameters(), lr=0.001)
    cnn_mnist.to(device)
    atn.to(device)

    x_test = x_test.to(device)
    y_test = y_test.to(device)

    print("original accuracy is %.3f" %
          (accuracy(cnn_mnist(x_test), y_test, target)[0]))
    for param in cnn_mnist.parameters():
        param.requires_grad = False

    # this is targeted white box attack
    for epoch in range(epoch_n):
        count = 0
        for iteration in range(iteration_n):
            x_batch, _ = batch_generator.next_batch(x_train, y_train)

            x_batch.to(device)

            # training
            x_grad = cal_grad_target(
                x_batch, cnn_mnist, target)  # target gradient
            x_adv = atn(x_batch, x_grad)    # the network output image
            y_now = cnn_mnist(x_adv)

            # calculating loss
            lossX = lossX_fn(x_adv, x_batch)
            lossY = lossY_fn(y_now, target)  # target loss Y
            loss = lossX * atn.beta + lossY

            # update weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            count += batch_size
            if count >= 1000:
                count -= 1000
                x_test_grad = cal_grad_target(x_test, cnn_mnist, target)
                x_adv_test = atn(x_test, x_test_grad)
                y_pred = cnn_mnist(x_adv_test)
                acc, targetrate = accuracy(y_pred, y_test, target)
                print('Epoch:', epoch, '|Step:', iteration, '|loss Y:%.4f' %
                      lossY.item(), '|image norm:%.4f' % lossX, '|test accuracy:%.4f' % acc, '|target rate:%.4f' % targetrate)
                if lossX < 0.011 and acc < 0.23 and targetrate > 0.86:
                    torch.save(atn.state_dict(), SAVE_PATH)
                    print('model saved')
                    return

                # self adjustment of parameter using threshold
                if lossX >= 0.03:
                    atn.beta *= 1.15
                elif lossX >= 0.015:
                    atn.beta *= 1.05
                elif lossX >= 0.01:
                    atn.beta *= 1.02

                if acc >= 0.4:
                    atn.beta /= 1.1
                elif acc >= 0.3:
                    atn.beta /= 1.05
                elif acc >= 0.20:
                    atn.beta /= 1.01

    torch.save(atn.state_dict(), SAVE_PATH)


def original_main():
    '''
    DO NOT USE THIS
    THIS IS TESTING ORIGINAL ATN
    '''
    # loading data, architecture
    x_train, y_train, x_test, y_test = cm.load_data('mnist')
    atn = ATN.ATN_a(beta=1.7)
    lossX_fn = nn.MSELoss()  # loss between X_origin and X_after
    FILE_PATH = 'data/mnist_cnn_model.pkl'
    cnn_mnist = torch.load(FILE_PATH)
    device = device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    # training parameter/using GPU is possible
    epoch_n, batch_size = 1, 100
    target = 3
    iteration_n = int(len(y_train)/batch_size)
    batch_generator = cm.batch_generator(batch_size)
    optimizer = torch.optim.Adam(atn.parameters(), lr=0.001)
    cnn_mnist.to(device)
    atn.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    print("original accuracy is %.3f" %
          (accuracy(cnn_mnist(x_test), y_test, target)[0]))
    for param in cnn_mnist.parameters():
        param.requires_grad = False

    for epoch in range(epoch_n):
        count = 0
        for iteration in range(iteration_n):
            x_batch, _ = batch_generator.next_batch(x_train, y_train)
            x_batch.to(device)

            # training
            x_adv = atn(x_batch)    # look at size!
            y_now = cnn_mnist(x_adv)
            y_origin = cnn_mnist(x_batch)

            y_origin.requires_grad_(False)
            lossX = lossX_fn(x_adv, x_batch)
            lossY = lossY_fn(y_now, target)
            loss = lossX * atn.beta + lossY

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            count += batch_size
            if count >= 1000:
                count -= 1000
                x_adv_test = atn(x_test)
                y_pred = cnn_mnist(x_adv_test)
                acc, targetrate = accuracy(y_pred, y_test, target)
                print('Epoch:', epoch, '|Step:', iteration, '|train loss:%.4f' %
                      loss.item(), '|image norm:%.4f' % lossX, '|test accuracy:%.4f' % acc, '|target rate:%.4f' % targetrate)
    SAVE_PATH = 'data/atn_a_mnist_model.ckpt'
    torch.save(atn, SAVE_PATH)


if __name__ == "__main__":
    i = 0
    save_path = 'data/GatnConv_mnistCNN_2_target' + str(i) + '.parameter'
    C_PATH='data/mnist_CNN_2_model_params.pkl'
    main(beta=0.6,target=i,SAVE_PATH=save_path, batch_size=5, atn='conv', classifier='2', classifier_PATH=C_PATH, epoch_n=1, continue_training=True)
