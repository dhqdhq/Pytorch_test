import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import torch.utils.data as Data


import torchvision
import torchvision.utils as vutils


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# ss = StandardScaler(with_mean = True , with_std = True)
# boston_Xs = ss.fit_transform(boston_X)
# #print(boston_Xs.shape)
# train_xt = torch.from_numpy(boston_Xs.astype(np.float32))
# train_yt = torch.from_numpy(boston_y.astype(np.float32))
# train_data = Data.TensorDataset(train_xt,train_yt)
# train_loader = Data.DataLoader(
#                     dataset = train_data,
#                     batch_size =128,
#                     shuffle = True,
#                     num_workers =1,
#                     )


# class MLPmodel(nn.Module):
#     def __init__(self):
#         super(MLPmodel,self) .__init__()
#         self.hidden1 = nn.Linear(
#             in_features = 13,
#             out_features = 10,
#             bias = True
#         )
#         self.active1= nn.ReLU()
#         self.hidden2 = nn.Linear(10,10)
#         self.active2 = nn.ReLU()
#         self.regression = nn.Linear(10,1)
#     def forward(self,x):
#         x = self.hidden1(x)
#         x=self.active1(x)
#         x = self.hidden2(x)
#         x = self.active2(x)
#         output = self.regression(x)
#         return output

# class MLPmodel2(nn.Module):
#     def __init__(self):
#         super(MLPmodel2,self) .__init__()
#         self.hidden = nn.Sequential(
#              nn.Linear(13,10),
#              nn.ReLU(),
#              nn.Linear(10,10),
#              nn.ReLU(),
#         )
#         self.regression = nn.Linear(10,1)
#     def forward(self,x):
#         x = self.hidden(x)
#         output = self.regression(x)
#         return output


# m1p1 = MLPmodel2()
# optimizer = SGD(m1p1.parameters(),lr=0.001)
# loss_func = nn.MSELoss()
# train_loss_all = []
# for epoch in range(30):
#     for step, (b_X,b_y) in enumerate(train_loader):
#         output = m1p1(b_X).flatten()
#         train_loss = loss_func(output,b_y)
#         optimizer.zero_grad()  #每个迭代步的梯度初始化为0
#         train_loss.backward()   #损失的后向传播   计算梯度
#         optimizer.step()      #使用梯度进行优化
#         train_loss_all.append(train_loss.item())

# plt.figure()
# plt.plot(train_loss_all,"r-")
# plt.show()
#torch.save(m1p1,"m1p1.pkl")
# pklload = torch.load("pkl/m1p1.pkl")
# print(pklload)

train_data = torchvision.datasets.MNIST(
    root = "./data/MNIST",
    train =True,
    transform =torchvision.transforms.ToTensor(),
    download = False
)

train_loader = Data.DataLoader(
                    dataset = train_data,
                    batch_size =128,
                    shuffle = True, 
                    num_workers =0,
                    )
print('len(train_loader)',len(train_loader))
test_data = torchvision.datasets.MNIST(
    root = "./data/MNIST",
    train =False,
    download = False
)

test_data_x = test_data.data.type(torch.FloatTensor)/255.0
print('test_data_x.dtype',test_data_x.dtype)
print("test_data_x  ",test_data_x.shape)
test_data_x  = torch.unsqueeze(test_data_x,dim=1)
print("test_data_x  ",test_data_x.shape)
test_data_y = test_data.targets


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self) .__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 1,
                    out_channels = 16,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                ),
                nn.ReLU(),
                nn.AvgPool2d(
                    kernel_size = 2,
                    stride = 2
                )
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(
                in_features = 32*7*7,
                out_features = 128
            ),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )
        self.out = nn.Linear(64,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        output = self.out(x)
        return output
MyConvnet = ConvNet()
print(MyConvnet)

#可视化卷积神经网络
# import hiddenlayer as hl
# hl_graph = hl.build_graph(MyConvnet,torch.zeros([1,1,28,28]))
# hl_graph.theme = hl.graph.THEMES["blue"].copy()
#hl_graph.save("graph/MyConvnet_hl",format="png")

#可视化网络
#from torchviz import make_dot
# x = torch.randn(1,1,28,28).requires_grad_(True)
# y = MyConvnet(x)
# MyConvnetvis = make_dot(y,params = dict(list(MyConvnet.named_parameters())+[('x',x)]))
# MyConvnetvis.directory = "graph/MyConvnet_vis"
# MyConvnetvis.view()

# from tensorboardX import SummaryWriter #运行指令 tensorboard --logdir=log
# SumWriter = SummaryWriter(log_dir = "log") 
# #定义优化器
# optimizer = torch.optim.Adam(MyConvnet.parameters(), lr = 0.0003)
# loss_func = nn.CrossEntropyLoss()  #损失函数
# train_loss = 0
# print_step = 100   #每100次迭代输出损失

# #迭代训练epoch轮
# for epoch in range(5):
#     #对数据加载器进行迭代计算
#     for step,(b_x,b_y) in enumerate(train_loader):
#         #计算每个batch损失
#         output = MyConvnet(b_x)
#         loss = loss_func(output,b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss = train_loss +loss
#         #计算迭代次数
#         niter = epoch * len(train_loader) + step + 1
#         #每经过print_step次迭代后输出
#         if niter % print_step ==0:
#             #为日志添加训练集损失函数
#             SumWriter.add_scalar("train loss" , train_loss.item()/niter , global_step = niter)
#             #计算测试集上的精度
#             output = MyConvnet(test_data_x)
#             _,pre_lab = torch.max(output,1)
#             acc = accuracy_score(test_data_y , pre_lab)
#             #为日志添加预测集上的精度
#             SumWriter.add_scalar("test acc" , acc.item() , niter)
#             #为日志添加训练数据的可视化图像，当前batch的图像
#             #将一个batch的数据进行预处理
#             b_x_im = vutils.make_grid(b_x , nrow = 12)
#             SumWriter.add_image('train image sample' , b_x_im , niter)
#             #使用直方图可视化网络参数分布
#             for name, param in MyConvnet.named_parameters():
#                 SumWriter.add_histogram(name, param.data.numpy(),niter)

##可视化网络
# import hiddenlayer as hl
# import time
# optimizer = torch.optim.Adam(MyConvnet.parameters(), lr = 0.0003) #定义优化器
# loss_func = nn.CrossEntropyLoss()  #损失函数
# history1 = hl.History() # 记录训练过程中的指标
# canvas1 = hl.Canvas() # 使用Canvas进行可视化
# print_step = 100   #每100次迭代输出损失

# #迭代训练epoch轮
# for epoch in range(5):
#     #对数据加载器进行迭代计算
#     for step,(b_x,b_y) in enumerate(train_loader):
#         #计算每个batch损失
#         output = MyConvnet(b_x)  #CNN在训练batch上的输出
#         loss = loss_func(output,b_y) #交叉熵损失函数    
#         optimizer.zero_grad() #每个迭代布的梯度初始化为0
#         loss.backward() #损失的后向传播
#         optimizer.step() #使用梯度进行优化

#         #每经过print_step次迭代后输出
#         if (step % print_step) ==0:

#             #计算测试集上的精度
#             output = MyConvnet(test_data_x)
#             _,pre_lab = torch.max(output,1)
#             acc = accuracy_score(test_data_y , pre_lab)
            
#             #计算每个epoch和step的模型的输出特征
#             history1.log((epoch, step), train_loss = loss, test_acc = acc, hidden_weight = MyConvnet.fc[2].weight)
#             # 可视化训练过程
#             with canvas1:
#                 canvas1.draw_plot(history1['train_loss'])
#                 canvas1.draw_plot(history1['test_acc'])
#                 #canvas1.draw_plot(history1['hidden_weight'])

#可视化
from visdom import Visdom   #python -m visdom.server
from sklearn.datasets import load_iris
iris_x,iris_y = load_iris(return_X_y = True)
vis = Visdom()
vis.scatter(iris_x[:,0:2],Y = 1+iris_y, win="2D", env = "main" )
vis.scatter(iris_x[:,0:3],Y = 1+iris_y, win= "3D", env = "main" ,opts = dict(markersize = 4, xlabel ="特征1",ylabel = "特征2"))