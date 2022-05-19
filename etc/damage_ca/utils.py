import requests
import torch
from torchvision.utils import save_image
import numpy as np
import PIL.Image, PIL.ImageDraw
import io
import math


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

# Adversarial attack
def adv_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def get_gaussian_kernel(kernel_size=3, sigma=1.0, channels=3, padding=1):
    """ Create a model for applying 2d convolutional gaussian blur filter"""
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, padding=padding, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

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

def save_model(ca, base_fn):
    """Save a PyTorch model to a specific path."""
    torch.save(ca.state_dict(), base_fn)

def visualize(xs, step_i, nrow=1):
    """Save a batch of multiple x's to file"""
    for i in range(len(xs)):
        xs[i] = to_rgb(xs[i]).permute(0, 3, 1, 2)
    save_image(torch.cat(xs, dim=0), './logg/pic/p%04d.png' % step_i, nrow=nrow, padding=0)

class Pool:
    """Class for storing and providing samples of different stages of growth."""
    def __init__(self, seed, size):
        self.size = size
        self.slots = np.repeat([seed], size, 0)
        self.seed = seed

    def commit(self, batch):
        """Replace existing slots with a batch."""
        indices = batch["indices"]
        for i, x in enumerate(batch["x"]):
            if (x[:, :, 3] > 0.1).any():  # Avoid committing dead image
                self.slots[indices[i]] = x.copy()

    def sample(self, c):
        """Retrieve a batch from the pool."""
        indices = np.random.choice(self.size, c, False)
        batch = {
            "indices": indices,
            "x": self.slots[indices]
        }
        return batch

    