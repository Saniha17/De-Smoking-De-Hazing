import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 3, 5, 1, 2)
        self.conv4 = nn.Conv2d(6, 3, 7, 1, 3)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = self.relu(self.conv5(cat3))
        
        # J(x) = K(x)I(x) - K(x) + 1
        clean_image = k * x - k + 1
        return self.relu(clean_image)

def dehaze_image_ai(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([transforms.ToTensor()])
    hazy_tensor = transform(img_rgb).unsqueeze(0) 

    model = AODnet()
    model.eval() 

    with torch.no_grad():
        clean_tensor = model(hazy_tensor)

    clean_image = clean_tensor.squeeze().permute(1, 2, 0).numpy()
    clean_image = np.clip(clean_image, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Hazy Image')
    axes[0].axis('off')
    
    axes[1].imshow(clean_image)
    axes[1].set_title('AI Output (Untrained)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dehaze_image_ai('images/hazy_input.jpg')