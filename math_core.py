import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    return mean_a * I + mean_b

def get_dark_channel(image, patch_size=15):
    b, g, r = cv2.split(image)
    min_img = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_img, kernel)

def recover_image(img_array, engine, omega=0.95, model=None):
    img_float = img_array.astype(np.float64)
    dark_channel = get_dark_channel(img_float)
    
    h, w = img_float.shape[:2]
    image_size = h * w
    num_pixels = int(max(image_size * 0.001, 1))
    indices = np.argsort(dark_channel.reshape(image_size))[-num_pixels:]
    atmospheric_light = np.mean(img_float.reshape(image_size, 3)[indices], axis=0)
    
    if engine == "AI Deep Learning (DehazeNet)" and model is not None:
        transform = transforms.Compose([transforms.ToTensor()])
        hazy_tensor = transform(img_array).unsqueeze(0).float()
        with torch.no_grad():
            t_tensor = model(hazy_tensor)
        t_map = t_tensor.squeeze().numpy()
        if np.std(t_map) < 0.01: 
            norm_image = img_float / atmospheric_light
            raw_t = 1 - omega * get_dark_channel(norm_image, 15)
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
            transmission = guided_filter(gray_image, raw_t, r=40, eps=1e-3)
        else:
            transmission = cv2.resize(t_map, (w, h))
    else:
        norm_image = img_float / atmospheric_light
        raw_transmission = 1 - omega * get_dark_channel(norm_image, 15)
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        transmission = guided_filter(gray_image, raw_transmission, r=40, eps=1e-3)
        
    transmission = np.maximum(transmission, 0.1)
    transmission_map = np.empty(img_float.shape)
    for i in range(3):
        transmission_map[:, :, i] = transmission
        
    recovered_image = np.clip((img_float - atmospheric_light) / transmission_map + atmospheric_light, 0, 255).astype(np.uint8)
    gaussian = cv2.GaussianBlur(recovered_image, (0, 0), 2.0)
    final_img = cv2.addWeighted(recovered_image, 1.5, gaussian, -0.5, 0)
    
    return final_img