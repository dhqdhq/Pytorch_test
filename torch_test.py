import torch
import numpy as np
import torch.utils.data as Data
from sklearn.datasets import load_boston,load_iris
# boston_X,boston_y = load_boston(return_X_y=True)
# train_xt = torch.from_numpy(boston_X.astype(np.float32))
# train_yt = torch.from_numpy(boston_y.astype(np.float32))
# # print(train_yt)
# train_data = Data.TensorDataset(train_xt,train_yt)

# train_loader = Data.DataLoader(
#                     dataset = train_data,
#                     batch_size = 64,
#                     shuffle = True,
#                     num_workers =1,
#                     )
# for step, (b_x,b_y) in enumerate(train_loader):
#     if step>0:
#         break
# print('b_x.shape',b_x.shape)
# print('b_y.shape',b_y.shape)

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# train_data = FashionMNIST(
#     root = "./data/FashionMNIST",
#     train =True,
#     transform = transforms.ToTensor(),
#     download = False
# )

# train_loader = Data.DataLoader(
#                     dataset = train_data,
#                     batch_size = 64,
#                     shuffle = True, 
#                     num_workers =2,
#                     )
# print(len(train_loader))
# test_data = FashionMNIST(
#     root = "./data/FashionMNIST",
#     train =False,
#     download = False
# )
# print(test_data.data.dtype)
# test_data_x = test_data.data.type(torch.FloatTensor)/255
# print(test_data_x.dtype)
# print("test_data_x  ",test_data_x.shape)
# test_data_x  = torch.unsqueeze(test_data_x,dim=1)
# print("test_data_x  ",test_data_x.shape)
# test_data_y = test_data.targets

# train_data_transforms = transforms.Compose([    #组合多种变换
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],
#                                                    [0.229,0.224,0.225])
#                                                     ]) 
#     root = "./data/FashionMNIST",
#     train =True,
#     transform = transforms.ToTensor(),
#     download = False
# )
import torch.nn as nn
import torch.optim
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self) .__init__()
        self.hidden = nn.Sequential(nn.Linear(13,10),nn.ReLU(),)
        self.regression = nn.Linear(10,1)
    def forward(self,x):
        x = self.hidden(x)
        output = self.regression(x)
        return output
testnet = TestNet()
#print(testnet)
optimizer = torch.optim.Adam(
    [{"params":testnet.hidden.parameters(),"lr":0.0001},
     {"params":testnet.regression.parameters(),"lr":0.01}],
    lr=1e-2)
