import os

import argparse
import torch
from torch.autograd import Variable

from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np

from torch import nn
label_dict={
 1:'isolation',
 4:'punching',
 8:'strangle',
 0:'gossiping',
 7:'stabbing',
 6:'slapping',
 2:'laughing',
 3:'pullinghair',
 5:'quarrel'
}

loader = transforms.Compose([ transforms.ToTensor(),
                              transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU
def get_input_args():
    """
    """
    parser = argparse.ArgumentParser(description='Get NN arguments')
    # Define arguments
    parser.add_argument('imageFile', type=str, help='mandatory data directory')#compulsory


    return parser.parse_args()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 9),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)


# print(model)
model.load_state_dict(base.pt)


# print(get_input_args())
image = image_loader(get_input_args().imageFile)

with torch.no_grad():
    out = (model.forward(image))
    result = torch.exp(out).cpu().data.topk(1)  # MODIFIED
    # print(result)
classes = np.array(result[1][0], dtype=np.int)
result=label_dict[classes[0]]
print(result)

# print('result')


