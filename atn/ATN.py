'''
This file includes ATNs as classes.
'''
import torch
nn = torch.nn

class GATN_FC(nn.Module):
    '''
    USAGE: atn = GATN_FC()

    This is FC GATN that takes also gradient as input and inner layer with low dimension

    CONTRIBUTER: henryliu, 07.25
    '''

    def __init__(self, beta=1.5, innerlayer=200, width=28, channel=1):
        '''
            The sturcture of this network is:
            INPUT: x_grad, x_image
            OUTPUT: x_adv
            STRUCTURE:
            (x_image, x_grad) -> layer1{
                Linear,
                SELU,
            }
            layer1 -> layer2{
                Linear,
                SELU,
            }
            (layer2 - x_grad) ->layer3{
                Linear
                SELU
            }
            layer3 -> norm_layer{
                sigmoid
            } 
        '''
        super().__init__()
        self.beta = beta
        self.channel = channel
        self.width = width
        self.layer1 = nn.Sequential(
            nn.Linear(2*width * width * channel, innerlayer),
            nn.SELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(innerlayer,width * width * channel),
            nn.SELU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(width * width * channel, width * width * channel),
            nn.SELU()
        )
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.torch.nn.init.xavier_normal_(m.weight)
        self.out_image = nn.Sigmoid()

    def forward(self, x_image, x_grad):
        self.batch_size = x_image.size(0)
        x_image = x_image.view(x_image.size(0), -1)
        x_grad = x_grad.view(x_grad.size(0), -1)
        x = torch.cat((x_image, x_grad), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x + 0.4 * x_grad)
        x = self.out_image( (x + x_image-0.5)*5 ) # adding pertubation and norm
        return x.view(x.size(0), self.channel, self.width, self.width)

class GATN_Conv(nn.Module):
    '''
    USAGE: atn = GATN_Conv()

    This is ATN_a that takes also gradient as input and use convolutional layer

    CONTRIBUTER: henryliu, 07.25
    '''

    def __init__(self, beta=1.5, innerlayer=100, width=28, channel=1):
        '''
        This is a simple net of (width * width)->FC( innerlayer )->FC(width * width)-> sigmoid(grad + output)
        '''
        super().__init__()
        self.beta = beta
        self.channel = channel
        self.width = width
        self.layer1_conv = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(1, 8, 5, 1), # use padding = 0
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (10, 14, 14)
        )
        self.layer2_conv = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(1, 8, 5, 1), # use padding = 0
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (10, 14, 14)
        )
        self.layer3 = nn.Sequential(
            nn.Linear( 8 * (width - 4)/2 * (width - 4)/2 * 2, width * width * channel),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(width * width * channel, width * width * channel)
        )
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.torch.nn.init.xavier_normal_(m.weight)
        self.out_image = nn.Sigmoid()

    def forward(self, x_image, x_grad):
        self.batch_size = x_image.size(0)
        x1 = self.layer1_conv(x_image).view(self.batch_size, -1)
        x2 = self.layer2_conv(x_grad).view(self.batch_size, -1)
        x = torch.cat((x1, x2), dim=1) # [D, 10 * width/4 *width /4 * 2]
        x_image = x_image.view(x_image.size(0), -1)
        x_grad = x_grad.view(x_grad.size(0), -1)        
        x = self.layer3(x)
        x = self.layer4(x + x_grad) # x is perturbation
        x = self.out_image( (x + x_image-0.5)*5 ) # adding pertubation and norm
        return x.view(x.size(0), self.channel, self.width, self.width)

class ATN_a(nn.Module):
    '''
   DO NOT USE THIS
   THIS IS FAILED TRY
    '''
    def __init__(self, alpha=3, beta=1.5, innerlayer=2000, width=28, channel=1):
        '''
            This is a simple net of (width * width)->FC( innerlayer )->FC(width * width)
        '''
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.channel = channel
        self.width = width
        self.layer1 = nn.Sequential(
            nn.Linear(width * width * channel, innerlayer),
            nn.SELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(innerlayer, innerlayer),
            nn.SELU()
        )
        self.layer3 = nn.Linear(innerlayer, width * width * channel)
        self.outlayer = nn.Sigmoid()
        self.weight_a = torch.tensor([1.5], requires_grad=False)
        self.weight_b = torch.tensor([0.5], requires_grad=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = (x - self.weight_b) * self.weight_a
        x = self.outlayer(x)
        return x.view(x.size(0), self.channel, self.width, self.width)


class ATN_b(nn.Module):
    '''
    DO NOT USE THIS
    THIS IS FAILED TRY
    '''
    def __init__(self, alpha=3, beta=1.5, innerlayer=1000, width=28, channel=1):
        '''
            The network is following
            layer1: {
                conv,
                ReLU,
                Maxpooling
            }
            layer2:{
                conv,
                ReLU,
                Maxpooling
            }
            layer:3 {
                FC,
                Sigmoid
            }
        '''
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.channel = channel
        self.width = width
        self.layer1_conv = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(1, 8, 3, 1, 1),  
            nn.ReLU(), 
            nn.MaxPool2d(2),  # output shape (8, 14, 14)
        )
        self.layer2_conv = nn.Sequential(  # input shape (8, 14, 14)
            nn.Conv2d(8, 16, 3, 1, 1),  
            nn.ReLU(), 
            nn.MaxPool2d(2),  # output shape (16, 7, 7 )
        )
        self.layer3_fc = nn.Sequential(
            nn.Linear(width/4 * width/4 * channel * 16, channel * width * width),
        )
        self.outlayer = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1_conv(x)
        x = self.layer2_conv(x)
        x = x.view(x.size(0), self.width/4 * self.width/4 * self.channel * 16 )
        x = self.layer3_fc(x)
        x = x - 0.5
        x = x.outlayer(x)
        return x.view(x.size(0), self.channel, self.width, self.width)


class GATN_a(nn.Module):
    ''' 
    DO NOT USE THIS
    THIS IS FAILED TRY
    '''

    def __init__(self, alpha=3, beta=1.5, innerlayer=2000, width=28, channel=1):
        '''
            This is a simple net of (width * width)->FC( innerlayer )->FC(width * width)-> sigmoid(grad + output)
        '''
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.channel = channel
        self.width = width
        self.layer1 = nn.Sequential(
            nn.Linear(2*width * width * channel, innerlayer),
            nn.SELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(innerlayer, innerlayer),
            nn.SELU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(innerlayer, width * width * channel),
            nn.SELU()
        )
        self.grad_linear = nn.Linear(width * width * channel, width*width*channel)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.torch.nn.init.xavier_normal_(m.weight)
        self.weight_v = torch.ones(channel * width * width, requires_grad=True)  # MSL_ave of x_grad is around 0.03 / abs is 0.1 
        self.weight_res = torch.tensor([0.02], requires_grad = False ) #0.6
        self.out_image = nn.Sigmoid()
    def forward(self, x_image, x_grad):
        self.batch_size = x_image.size(0)

        x_image = x_image.view(x_image.size(0), -1)
        x_grad = x_grad.view(x_grad.size(0), -1)
        x = torch.cat((x_image, x_grad), dim=1)
        x = self.layer1(x)
        #x = self.layer2(x)
        x = self.layer3(x)
        # should this grad be + or -?
        # MSE_ave of x is around 4.7 / abs is 0.7
        #x= self.out_perturb( ) #perturbation
        x = self.out_image( (x * self.weight_res + self.grad_linear(x_grad) +x_image-0.5)*5 ) # adding pertubation and norm
        return x.view(x.size(0), self.channel, self.width, self.width)


