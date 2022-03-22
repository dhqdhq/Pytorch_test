import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
#导入数据
housedata = fetch_california_housing()
#数据切分为训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(housedata.data,housedata.target,test_size=0.3,random_state=42)
#print("x_train:",x_train,"x_test:",x_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape) #(14448, 8) (6192, 8) (14448,) (6192,)
#数据标准化处理
scale = StandardScaler()
# 1.fit_transform功能：
# 			先计算均值、标准差，再标准化
# 		2.参数:
# 			X: 二维数组
x_train_s = scale.fit_transform(x_train)
x_test_s = scale.fit_transform(x_test)
#print("x_train:",x_train_s,"x_test:",x_test_s.shape) 
#将训练集数据处理为数据表，方便探索数据情况
#df3 = pd.DataFrame(np.random.randn(3, 3), index=list('ABC'), columns=list('ABC'))
#           A                    B                  C
# A  1.106838  0.309086  0.748472
# B  0.439228 -0.213154 -1.438647
# C  0.292450  0.841237 -0.069207
# housedatadf = pd.DataFrame(data=x_train_s,columns=housedata.feature_names)
# housedatadf["target"] = y_train
# #可视化数据的相关系数热力图
# datacor = np.corrcoef(housedatadf.values,rowvar=0)
# datacor = pd.DataFrame(data=datacor,columns=housedatadf.columns,index=housedatadf.columns)
# plt.figure(figsize=(8,6))
# ax = sns.heatmap(datacor,square=True,annot=True,fmt=".3f",linewidths=.5,cmap="YlGnBu",cbar_kws={"fraction":0.046,"pad":0.03})
# plt.show()
#将数据集转化为张量
train_xt = torch.from_numpy(x_train_s.astype(np.float32))
train_yt = torch.from_numpy(y_train.astype(np.float32))
test_xt = torch.from_numpy(x_test_s.astype(np.float32))
test_yt = torch.from_numpy(y_test.astype(np.float32))
#将训练数据处理为数据加载器
train_data = Data.TensorDataset(train_xt,train_yt)
test_data = Data.TensorDataset(test_xt,test_yt)
train_loader = Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)
#test_loader = Data.DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0)
#搭建全连接神经网络回归网络
class MLPPregression(nn.Module):
    def __init__(self):
        super(MLPPregression, self).__init__()
        #定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=8,out_features=100,bias=True)
        #定义第二个隐藏层
        self.hidden2 = nn.Linear(100,100)
        #定义第三个隐藏层
        self.hidden3 = nn.Linear(100,50)
        #回归预测层
        self.predict = nn.Linear(50,1)
    #定义网络的前向传播路径
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        #输出一个一维向量
        return output[:,0]
#输出网络结构
mlpreg = MLPPregression()
#print(mlpreg)
#定义优化器
optimizer = SGD(mlpreg.parameters(),lr=0.001)
loss_func = nn.MSELoss() #均方误差损失函数
train_loss_all = []
for epoch in range(30):
    train_loss = 0
    train_num = 0
    for step,(b_x,b_y) in enumerate(train_loader):#b_x：64 * 8
        #print(b_x.size()) 64 * 8
        #print(b_y.size()) 64
        output = mlpreg(b_x)
        loss = loss_func(output,b_y) #loss是均方误差，可认为是一个样本的loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * b_x.size(0) #b_x.size(0) = batch_size
        train_num += b_x.size(0)
    train_loss_all.append(train_loss / train_num) #一个batch_size的loss加和 / batch_size
#可视化损失函数的变化情况
plt.figure(figsize=(10,6))
plt.plot(train_loss_all,"ro-",label="Train loss")
plt.legend()
plt.grid()
plt.xlabel("eopch")
plt.ylabel("loss")
plt.show()
# #对测试集进行预测
pre_y = mlpreg(test_xt)
pre_y = pre_y.data.numpy()
mae = mean_absolute_error(y_test,pre_y)
print("在测试集上的绝对值误差为：",mae)
#可视化在测试集上真实值和预测值的差异
index = np.argsort(y_test)
plt.figure(figsize=(12,5))
plt.plot(np.arange(len(y_test)),y_test[index],"r",label="Original Y")
plt.scatter(np.arange(len(pre_y)),pre_y[index],s=3,c="b",label="Prediction")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("index")
plt.ylabel("Y")
plt.show()
