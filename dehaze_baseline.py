import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_dark_channel(image, patch_size=15):
    """Calculates the dark channel of an image."""
    b, g, r = cv2.split(image)
    min_img = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel):
    """Estimates the global atmospheric light."""
    h, w = image.shape[:2]
    image_size = h * w
    num_pixels = int(max(image_size * 0.001, 1))
    
    dark_channel_flat = dark_channel.reshape(image_size)
    image_flat = image.reshape(image_size, 3)
    
    indices = np.argsort(dark_channel_flat)[-num_pixels:]
    atmospheric_light = np.mean(image_flat[indices], axis=0)
    return atmospheric_light

def get_transmission(image, atmospheric_light, omega=0.95, patch_size=15):
    """Estimates the transmission map."""
    norm_image = image / atmospheric_light
    transmission = 1 - omega * get_dark_channel(norm_image, patch_size)
    return transmission

def recover_image(image, transmission, atmospheric_light, t0=0.1):
    """Recovers the clear image."""
    transmission = np.maximum(transmission, t0)
    transmission_map = np.empty(image.shape)
    for i in range(3):
        transmission_map[:, :, i] = transmission
        
    recovered_image = (image - atmospheric_light) / transmission_map + atmospheric_light
    return np.clip(recovered_image, 0, 255).astype(np.uint8)

def dehaze(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float64)
    
    # Apply DCP steps
    dark_channel = get_dark_channel(img_float)
    atmospheric_light = get_atmospheric_light(img_float, dark_channel)
    transmission = get_transmission(img_float, atmospheric_light)
    recovered_img = recover_image(img_float, transmission, atmospheric_light)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Hazy Image')
    axes[0].axis('off')
    
    axes[1].imshow(recovered_img)
    axes[1].set_title('De-hazed Output')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Replace with the path to your hazy image
    dehaze('images/hazy_input.jpg')