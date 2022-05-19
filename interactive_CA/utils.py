import requests
import torch
from torchvision.utils import save_image
import numpy as np
import PIL.Image, PIL.ImageDraw
import io

def adv_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def load_emoji(emoji_code, img_size):
    """Loads image of emoji with code 'emoji' from google's emojirepository"""
    emoji_code = hex(ord(emoji_code))[2:].lower()
    url = 'https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u%s.png' % emoji_code
    req = requests.get(url)
    img = PIL.Image.open(io.BytesIO(req.content))
    img.thumbnail((img_size, img_size), PIL.Image.ANTIALIAS)
    img = np.float64(img) / 255.0
    img[..., :3] *= img[..., 3:]

    return img

def to_alpha(x):
    """Return the alpha channel of an image."""
    return torch.clamp(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
    """Return the three first channels (RGB) with alpha deducted."""
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb

def to_rgba(x):
    """Return the four first channels (RGBA) of an image."""
    return x[..., :4]

def make_seed(size, n_channels):
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1
    return x

def to_rgb_ad(img_rgba):
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)

    