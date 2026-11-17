import torch
import torch.nn as nn
import os
import urllib.request
import sys
import types
import streamlit as st

class DehazeNet(nn.Module):
    def __init__(self):
        super(DehazeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, 7, padding=3)
        self.conv5 = nn.Conv2d(64, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(7, stride=1, padding=3)
        self.brelu = nn.Hardtanh(0.0, 1.0)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2_1 = self.conv2(x1)
        x2_2 = self.conv3(x1)
        x2_3 = self.conv4(x1)
        x2 = torch.cat((x2_1, x2_2, x2_3, x1), 1) 
        x3 = self.relu(self.conv5(x2))
        x4 = self.pool(x3)
        transmission_map = self.brelu(self.conv6(x4))
        return transmission_map

fake_model = types.ModuleType('model')
fake_model.DehazeNet = DehazeNet
sys.modules['model'] = fake_model

@st.cache_resource
def load_ai_model(weights_path="dehazenet.pth"):
    model = DehazeNet()
    if not os.path.exists(weights_path):
        url = "https://github.com/thuBingo/DehazeNet_Pytorch/raw/master/defog4_noaug.pth"
        try:
            urllib.request.urlretrieve(url, weights_path)
        except Exception:
            return model 
    try:
        loaded_data = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        if hasattr(loaded_data, 'state_dict'):
            model.load_state_dict(loaded_data.state_dict())
        else:
            model.load_state_dict(loaded_data)
    except Exception:
        pass
    model.eval()
    return model