import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

train_data = FashionMNIST(
    root = "./data/FashionMNIST",
    train = True,
    transform = transforms.ToTensor(),
    download = False
)
train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = 64,
    shuffle = False,
    num_workers = 0,
)
# print("train_loader的batch数量: ",len(train_loader))
# for step, (b_x,b_y) in enumerate(train_loader):
#     if step > 0:
#         break
# print(step)
# print( b_x.size())
# print( b_y)
# batch_x = b_x.squeeze().numpy()
# batch_y = b_y.numpy()
class_label = train_data.classes
# class_label[0] = "T-shirt"
# plt.figure(figsize = (12,5))
# for ii in np.arange(len(batch_y)):
#     plt.subplot(4,16,ii+1)
#     plt.imshow(batch_x[ii,:,:],cmap = plt.cm.gray)
#     plt.title(class_label[batch_y[ii]],size = 9)
#     plt.axis("off")
#     plt.subplots_adjust(wspace = 0.05)
# plt.show()

test_data = FashionMNIST(
    root = "./data/FashionMNIST",
    train = False,
    download = False
)
test_data_x = test_data.data.type(torch.FloatTensor)/255.0
print(test_data_x.shape)
test_data_x = torch.unsqueeze(test_data_x,dim = 1)
test_data_y = test_data.targets 
print(test_data_x.shape)
print(test_data_y.shape)



class MyConvNet(nn.Module):
    def __init__(self):
        super( MyConvNet, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,             ##  输入的featrue map
                out_channels = 16,       ##  输出的featrue map
                kernel_size = 3,              ##  卷积核尺寸
                stride = 1,                          ##  卷积步长
                padding =1                       ##  填充
            ),                                      ##  卷积后：  （1*28*28） --> （16*14*14）
            nn.ReLU(),                              ##  激活函数
            nn.AvgPool2d(
                kernel_size = 2,                ##  平均池化层，使用 2*2
                stride = 2                             ##  池化步长为2
            ),                                      ##  池化后：  （16*28*28）--> （16*14*14）
        )
        #定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,0),      ##  卷积后： （16*14*14） --> （32*12*12）
            nn.ReLU(),
            nn.AvgPool2d(2,2)                 ##  池化后：  （32*12*12）--> （32*6*6）
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*6*6,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    
    def forward(self,x):
    #定义网络的前向传播路径
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)             ##  展平多维卷积图层
        output = self.classifier(x)
        return output

##  空洞卷积
class MyConvdilaNet(nn.Module):
    def __init__(self):
        super( MyConvdilaNet, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,             ##  输入的featrue map
                out_channels = 16,       ##  输出的featrue map
                kernel_size = 3,              ##  卷积核尺寸
                stride = 1,                          ##  卷积步长
                padding =1,                       ##  填充
                dilation = 2
            ),                                      ##  卷积后：  （1*28*28） --> （16*26*26）
            nn.ReLU(),                              ##  激活函数
            nn.AvgPool2d(
                kernel_size = 2,                ##  平均池化层，使用 2*2
                stride = 2                             ##  池化步长为2
            ),                                      ##  池化后：  （16*26*26）--> （16*13*13）
        )
        #定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,0,dilation=2),      ##  卷积后： （16*13*13） --> （32*9*9）
            nn.ReLU(),
            nn.AvgPool2d(2,2)                 ##  池化后：  （32*9*9）--> （32*4*4）
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*4*4,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    
    def forward(self,x):
    #定义网络的前向传播路径
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)             ##  展平多维卷积图层
        output = self.classifier(x)
        return output




##  输出网络结构
#myconvnet = MyConvNet()
##  空洞卷积
myconvnet = MyConvdilaNet()
def train_model(model, traindataloader, train_rate, criterion, optimizer, num_epochs = 25):
    """
    model:                       网络模型；
    traindataloader:    训练数据集，切分为训练和测试集；
    train_rate:                训练集barchsize百分比； 
    criterion:                   损失函数；
    optimizer:                 优化方法；  
    num_epochs:          训练轮数
    """
    ##  计算训练使用的batch数量
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    ##  复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)
        ##  每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        for step,(b_x,b_y) in enumerate(traindataloader):
            if step < train_batch_num:
                model.train()  ##  设置为训练模式
                output = model(b_x)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output,b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            else:
                model.eval()  ##  设置为评估模式
                output = model(b_x)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output,b_y)
                val_loss += loss.item( )* b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        ##  计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print('{}  train loss :  {:.4f}    train acc:  {:.4f}'.format(epoch,train_loss_all[-1], train_acc_all[-1]))
        print('{}  val loss :     {:.4f}     val acc:     {:.4f}'.format(epoch,val_loss_all[-1], val_acc_all[-1]))
        ##  拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f} m {:.0f} s".format(time_use//60,time_use%60))
    ##  使用最好的模型参数
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data = {"epoch":range(num_epochs),
                        "train_loss_all":train_loss_all,
                        "val_loss_all":val_loss_all,
                        "train_acc_all":train_acc_all,
                        "val_acc_all":val_acc_all}
        )
    return model,train_process 



optimizer = torch.optim.Adam(myconvnet.parameters(),lr = 0.0003)  ##  定义优化器
criterion = nn.CrossEntropyLoss()  ##  损失函数


#  训练
myconvnet, train_process = train_model(
    myconvnet,train_loader,0.8,
    criterion, optimizer, num_epochs = 25
)
#  过程可视化
plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
plt.plot(train_process.epoch,train_process.train_loss_all,"ro-",label = "Train loss")
plt.plot(train_process.epoch,train_process.val_loss_all,"bs-",label = "Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1,2,2)
plt.plot(train_process.epoch,train_process.train_acc_all,"ro-",label = "Train acc")
plt.plot(train_process.epoch,train_process.val_acc_all,"bs-",label = "Val acc")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
#  保存网络
#torch.save(myconvnet,"data/myconvdilanet.pkl")



myconvnet = torch.load("data/myconvnet.pkl")
#myconvnet = torch.load("data/myconvdilanet.pkl")
myconvnet.eval()
output = myconvnet(test_data_x)
pre_lab = torch.argmax(output,1)
acc = accuracy_score(test_data_y,pre_lab)
print(acc)
conf_mat = confusion_matrix(test_data_y,pre_lab)
df_cm = pd.DataFrame(conf_mat,index = class_label,
                                                columns = class_label)
heatmap = sns.heatmap(df_cm,annot=True,fmt = "d",cmap = "YlGnBu")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0,ha = 'right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 45,ha = 'right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
