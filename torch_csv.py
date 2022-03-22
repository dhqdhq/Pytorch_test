import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
import torch.utils.data as Data
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.15f}'.format
# df = pd.read_csv('./data/csv/Data_as_400wan_ik.csv')
# print(df )
print("************取消第一行作为表头*************")
data2 = pd.read_csv('./joint_data/joints.csv',header=None)
# print(data2)
# print("************为各个字段取名**************")
# f = open("./data/csv/Data_as_400wan_ik.csv", "rb")
# data = np.loadtxt(f, delimiter=",", skiprows=0)
# f.close()
# print(data)
# print("***********将某一字段设为索引***************")
# data3 = pd.read_csv('./data/csv/Data_as_400wan_ik.csv',
# 	names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','user_id','book_id','rating'],
# 	index_col = "user_id")
# print(data3)
# print("************用sep参数设置分隔符**************")
# data4 = pd.read_csv('./data/csv/Data_as_400wan_ik.csv',
# 	names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','user_id','book_id','rating'],
# 	sep=',')
# print(data4)
# print("************自动补全缺失数据为NaN**************")
# data5 = pd.read_csv('./data/csv/Data_as_400wan_ik.csv',names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','user_id','book_id','rating'],
# usecols=["a","b","c"],index_col=0)
# print(data5)
print("************data2.values.astype(float)**************")
data2 =data2.values.astype(float)
# print(data2)
#data = data.astype(float)
X = data2[:,:12] # X输入的数据点（向量值），前n- 列都是输入X： 最后一列是输出： Y
x= torch.from_numpy(X)
y = data2[:,30:36]
y = torch.from_numpy(y)

torch.set_printoptions(precision=20)


# dl = torch.utils.data.DataLoader(x , batch_size=10, shuffle=False, num_workers=0)
train_data = Data.TensorDataset(x,y)
train_loader = Data.DataLoader(
                    dataset = train_data,
                    batch_size =10,
                    shuffle = True,
                    num_workers =0,
                    )
class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

net = Net(n_feature=12, n_hidden=100, n_output=6) # 几个类别就几个 output
net =net.double()
net.cuda()
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = nn.L1Loss(reduction='mean')
train_loss_all = []
for step, (x,y) in enumerate(train_loader):
    x=x.cuda()
    y=y.cuda()
    out = net(x)     # 喂给 net 训练数据 x, 输出分析值
    loss = loss_func(out, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    train_loss_all.append(loss.item())
    print(train_loss_all[-1])
    plt.cla()
    plt.plot(train_loss_all,"ro-",label="Train loss")
    plt.legend()
    plt.grid()
    plt.xlabel("eopch")
    plt.ylabel("loss")
    plt.pause(0.1)
        # if step % 2 == 0:

    #     prediction = out.cpu()
    #     pred_y = prediction.data.numpy()
    #     target_y = y.data.cpu().numpy()






