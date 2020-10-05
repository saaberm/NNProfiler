#import profiler and model 
from profileCNN import NN_Profiler
#from functions import *

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
#import matplotlib.pyplot as plt
#from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle


#check for alexnet
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
inp_size = (1, 3, 224,224)

#uncomment for testing 3D CNN 
#this model is imported from https://github.com/HHTseng/video-classification

"""
# 3D CNN parameters
fc_hidden1, fc_hidden2 = 256, 256
dropout = 0.0        # dropout probability

# training parameters
k = 101            # number of target category
epochs = 15
batch_size = 30
learning_rate = 1e-4
log_interval = 10
img_x, img_y = 256, 342  # resize video 2d frame size

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1
selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# load UCF101 actions names
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}



model = CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y,
              drop_p=dropout, fc_hidden1=fc_hidden1,  fc_hidden2=fc_hidden2, num_classes=k).to(device)
inp_size=(1, 1, 28, 256, 342)
"""

#profiling example net
ops, paramz, neurons, synapses = NN_Profiler(model, inp_size)
print ("Total OPS", int(ops)/1e9, "GOPs")# {:.2e}".format(int(ops)))
print ("Total params", int(paramz)/1e6, "Meg")# {:.2e}".format(int(params)))
print ("Total neurons", int(neurons)/1e6, "Meg")# {:.2e}".format(int(params)))
print ("Total synapses", int(synapses)/1e6, "Meg")# {:.2e}".format(int(params)))
