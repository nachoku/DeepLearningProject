import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import predict

import argparse
import torch
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import json
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
from network_prep import load_model
from processimage import process_image
from torch import nn
label_dict={
 1:'isolation',
 4:'punching',
 8:'strangle',
 6:'gossiping',
 7:'stabbing',
 0:'slapping',
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


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

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

model = AlexNet().to(device)


SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'alexnet-bullying.pt')
print(model)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.classifier[6].out_features=9
total=0
correct=0
labels_true=[]
labels_pred=[]
directory='data/test'
for filename in os.listdir(directory):
    if (filename == ".*"):
        continue
    path = os.path.join(directory, filename)
    print(path)


    for imagefile in os.listdir(path):

        print(os.path.join(path, imagefile))
        total+=1
        try:
            # image = process_image(os.path.join(path, imagefile))
            image = image_loader(os.path.join(path, imagefile))
        except:
            continue
        # image = image.unsqueeze(0).float()
        # image = Variable(image)
        # if torch.cuda.is_available():
        #     model.cuda()
        #     image = image.cuda()
        #     print('GPU PROCESSING')
        # else:
        #     print('CPU PROCESSING')
        with torch.no_grad():
            out = (model.forward(image))
            result = torch.exp(out).cpu().data.topk(1)  # MODIFIED
            print(result)
        classes = np.array(result[1][0], dtype=np.int)
        result=label_dict[classes[0]]
        # print(result)
        if(result==filename):
            correct+=1
            print('Correct')
        labels_true.append(filename)
        labels_pred.append(result)


labels=os.listdir(directory)

matrix= confusion_matrix(labels_true, labels_pred, labels=labels, sample_weight=None)
matrix_labelled=pd.DataFrame(matrix, index=['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
print(matrix_labelled)
print('Accuracy: '+str(correct/total))
matrix_labelled.to_csv('confusion_matrix_.csv')
