# 经验教训
1、慎用连等，连等的几个变量即使以后分开赋值，也会被赋予相同的值  
2、慎用tensorflow读取图片，会大大的影响速度。用skimage  
3、保存模型 torch.save(model, PATH);载入模型 model = torch.load(PATH)，然后用model.train()载入成功，用model.eval()载入失败。    
# 定义网络-方法一：
#定义语句简单，但中间结果不能输出。  
nn.Sequential(   ) #每一行最后都必须有逗号。最后一行可有可无。中间过程不能输出。  
forward()  #如果nn.Sequential定义成类，forward只返回nn.Sequential等于的值即可。  
# 定义网络-方法二：  
#定义语句相对方法一复杂，但中间结果可以输出。  
#与nn.Sequential区别：每一行后面不能用逗号，可以输出中间过程。需要在forward()中将每层再调用一次，并返回给一个变量。  
# 方法一（例子）：  
class Gen(nn.Module):  

    def __init__(self):  
        super(Gen, self).__init__()  
        self.main = nn.Sequential(  # Generator  
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(64 * 2),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(64 * 4),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(64 * 8),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(64 * 4),  
            nn.ReLU(True),  
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(64 * 2),  
            nn.ReLU(True),  
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(64),  
            nn.ReLU(True),  
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  
            nn.Sigmoid()  
        )  
    def forward(self,input):  
        return self.main(input)  

# 方法二（例子）：          
class Gen(nn.Module):  

    def __init__(self):  
        super(Gen, self).__init__()  
        self.conv1=nn.Conv2d(3, 64, 4, 2, 1, bias=False)  
        self.relu1=nn.LeakyReLU(0.2, inplace=True)  
        self.conv2=nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)  
        self.bn2=nn.BatchNorm2d(64 * 2)  
        self.relu2=nn.LeakyReLU(0.2, inplace=True)  
        self.conv3=nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)  
        self.bn3=nn.BatchNorm2d(64 * 4)  
        self.relu3=nn.LeakyReLU(0.2, inplace=True)  
        self.conv4=nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)  
        self.bn4=nn.BatchNorm2d(64 * 8)  
        self.relu4=nn.LeakyReLU(0.2, inplace=True)  
        self.convT5=nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)  
        self.bn5=nn.BatchNorm2d(64 * 4)  
        self.relu5=nn.ReLU(True)  
        self.convT6=nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False)  
        self.bn6=nn.BatchNorm2d(64 * 2)  
        self.relu6=nn.ReLU(True)  
        self.convT7=nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False)  
        self.bn7=nn.BatchNorm2d(64)  
        self.relu7=nn.ReLU(True)  
        self.convT8=nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)  
    def forward(self,input):  
        conv1 = self.conv1(input)  
        relu1 = self.relu1(conv1)  
        conv2 = self.conv2(relu1)  
        bn2 = self.bn2(conv2)  
        relu2 = self.relu2(bn2)  
        conv3 = self.conv3(relu2)  
        bn3 = self.bn3(conv3)  
        relu3 = self.relu3(bn3)  
        conv4 = self.conv4(relu3)  
        bn4 = self.bn4(conv4)  
        relu4 = self.relu4(bn4)  
        convT5=self.convT5(relu4)  
        bn5 = self.bn5(convT5)  
        relu5 = self.relu5(bn5)  
        convT6 =self.convT6(relu5)  
        bn6 = self.bn6(convT6)  
        relu6 = self.relu6(bn6)  
        convT7= self.convT7(relu6)  
        bn7 = self.bn7(convT7)  
        relu7 = self.relu7(bn7)  
        convT8 = self.convT8(relu7)  
        Dout = torch.sigmoid(convT8)  
        return Dout  
    
