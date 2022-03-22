import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import torch
from torchvision import transforms
import torchvision

model = torchvision.models.segmentation.fcn_resnet101(pretrained = True)
model.eval()

image = PIL.Image.open("data/FCN.png").convert('RGB')
image_transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406],
                                                   std = [0.229,0.224,0.225])
])
image_tensor = image_transf(image).unsqueeze(0)
output = model(image_tensor)["out"]
##  将输出转化为二维图像
outputarg = torch.argmax(output.squeeze(),dim = 0 ).numpy()
 
def decode_segmaps(image,label_colors, nc =10 ):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for cla in range (0 , nc):
        idx = image == cla
        r[idx] = label_colors[cla , 0]
        g[idx] = label_colors[cla , 1]
        b[idx] = label_colors[cla , 2]
    rgbimage = np.stack([r,g,b],axis = 2)
    return rgbimage

label_colors = np.array([(0,0,0),(128,0,0),(0,128,0),(0,0,128),(128,128,0),(128,128,128),
                                                    (0,128,128),(64,0,0),(128,0,128),(0,192,0)])
outputrgb = decode_segmaps(outputarg,label_colors)
plt.figure(figsize = (10,8))
plt.subplot(1,2,1)
plt.imshow(image)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(outputrgb)
#plt.axis("off")
#plt.subplots_adjust(wspace = 0.05)
plt.show()