# @title Image manipulation
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

def load_image(path, size=40):
    """Load an image.
    Parameters
    ----------
    path : pathlib.Path
        Path to where the image is located. Note that the image needs to be
        RGBA.
    size : int
        The image will be resized to a square wit ha side length of `size`.
    Returns
    -------
    torch.Tensor
        4D float image of shape `(1, 4, size, size)`. The RGB channels
        are premultiplied by the alpha channel.
    """
    img = Image.open(path)
    img = img.resize((size, size), Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]

    return torch.from_numpy(img).permute(2, 0, 1)[None, ...]


def to_rgb(img_rgba):
    """Convert RGBA image to RGB image.
    Parameters
    ----------
    img_rgba : torch.Tensor
        4D tensor of shape `(1, 4, size, size)` where the RGB channels
        were already multiplied by the alpha.
    Returns
    -------
    img_rgb : torch.Tensor
        4D tensor of shape `(1, 3, size, size)`.
    """
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)


def make_seed(size, n_channels):
    """Create a starting tensor for training.
    The only active pixels are going to be in the middle.
    Parameters
    ----------
    size : int
        The height and the width of the tensor.
    n_channels : int
        Overall number of channels. Note that it needs to be higher than 4
        since the first 4 channels represent RGBA.
    Returns
    -------
    torch.Tensor
        4D float tensor of shape `(1, n_chanels, size, size)`.
    """
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1
    return x


# display img with shape of (c, M, N) of RGBA tensor
def display(img):
  """Display an image using pyplot
    ----------
    img : torch.Tensor
        3D tensor of shape (c, M, N) where M and N is the height and width
    -------
    """
  img_ = to_rgb(img)[0].numpy()
  c, m, n = img_.shape
  plt.figure(figsize=(15,5),facecolor='w')
  plt.axis("off")
  arr = np.ones((m,n,c))
  for i in range(c):
    arr[..., i] = img_[i]
  plt.imshow(arr)
  plt.show()

def drawfunc(img):
  """Create a numpy array from an img
    ----------
    img : torch.Tensor
        3D tensor of shape (3, M, N) where M and N is the height and width
    Returns
    -------
    numpy.ndarray
        3D float ndarray of shape `(c, M, N)`.
    """
  img_ = to_rgb(img)[0].numpy()
  c, m, n = img_.shape
  arr = np.ones((m,n,c))
  for i in range(c):
    arr[..., i] = img_[i]
  return arr