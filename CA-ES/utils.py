import requests
import torch
from torchvision.utils import save_image
import numpy as np
import PIL.Image, PIL.ImageDraw
import io


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

    