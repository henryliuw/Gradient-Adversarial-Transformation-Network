'''
This file is executable file tha train the CNN for cifar10
'''
import torch as tc
import cifar10
import torch.nn as nn
import tool.common as com

x_train, y_train, x_test, y_test = com.load_data('cifar10')
train_generator = com.batch_generator( batch_size=128)
test_generator = com.batch_generator( batch_size=100)

loss_func = nn.CrossEntropyLoss().cuda()
cnn_cifar = cifar10.VGG('VGG19').cuda()
optimizer = tc.optim.Adam(cnn_cifar.parameters(), lr=0.001)

for epoch in range(350):
    for step in range(len(y_train)//128):
        x, y = train_generator.next_batch(x_train, y_train)
        x = x.cuda()
        y = tc.tensor(y).cuda()
        output = cnn_cifar(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 150 == 0:
            correct = 0
            for stept in range(len(y_test) // 100):
                xt, yt = test_generator.next_batch(x_test, y_test)
                xt = xt.cuda()
                yt = tc.tensor(yt).cuda()
                outputs = cnn_cifar(xt)
                _, predicted = tc.max(outputs.data, 1)
                correct += (predicted == yt).sum().item()

            total = 10000
            accuracy = correct / total
            print('Epoch:', epoch, '|Step:', step,
                  '|train loss:%.4f' % loss.data.item(), '|test accuracy:%.4f' % accuracy)

tc.save(cnn_cifar, 'cifar_cnn')
