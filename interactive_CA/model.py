import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CellularAutomataModel(nn.Module):
    def __init__(self, n_channels, hidden_channels, fire_rate):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.fire_rate = fire_rate

        self.fc0 = nn.Linear(self.n_channels * 3, self.hidden_channels, bias=False)
        self.fc1 = nn.Linear(self.hidden_channels, self.n_channels, bias=False)
        with torch.no_grad(): self.fc1.weight.zero_()

        identity = np.float64([0, 1, 0])
        identity = torch.from_numpy(np.outer(identity, identity))
        sobel_x = torch.from_numpy(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0)  # sobel filter
        sobel_y = sobel_x.T
        self.kernel = torch.cat([
            identity[None, None, ...], 
            sobel_x[None, None, ...], 
            sobel_y[None, None, ...]], 
            dim=0).repeat(self.n_channels, 1, 1, 1)

        for param in self.parameters(): param.requires_grad = False

        self.double()

    def perceive(self, x):
        """Percieve neighboors with two sobel filters and one single-entry filter"""
        y = F.conv2d(x.permute(0, 3, 1, 2), self.kernel, groups=16, padding=1)
        y = y.permute(0, 2, 3, 1)
        return y
    
    def loss(self, x, y):
        """mean squared error"""
        return torch.mean(torch.square(x[..., :4] - y), [-2, -3, -1])

    def forward(self, x, fire_rate=None, step_size=1.0):
        """Forward a cell grid through the network and return the cell grid with changes applied."""
        y = self.perceive(x)
        pre_life_mask = get_living_mask(x)
        dx1 = self.fc0(y)
        dx1 = F.relu(dx1)
        dx2 = self.fc1(dx1)
        dx = dx2 * step_size

        if fire_rate is None:
            fire_rate = self.fire_rate

        update_mask_rand = torch.rand(*x[:, :, :, :1].shape)
        update_mask = update_mask_rand <= fire_rate
        x += dx * update_mask.double()
        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask.bool() & post_life_mask.bool()
        res = x * life_mask.double()

        return res


def get_living_mask(x):
    """returns boolean vector of the same shape as x, except for the last dimension.
    The last dimension is a single value, true/false, that determines if alpha > 0.1"""
    alpha = x[:, :, :, 3:4]
    m = F.max_pool3d(alpha, kernel_size=3, stride=1, padding=1) > 0.1
    return m

class CAModel(nn.Module):
    def __init__(self, n_channels=16, hidden_channels=128, fire_rate=0.5, device=None):
        super().__init__()


        self.fire_rate = 0.5
        self.n_channels = n_channels
        self.device = device or torch.device("cpu")

        # Perceive step
        sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter_ / scalar
        sobel_filter_y = sobel_filter_.t() / scalar
        identity_filter = torch.tensor(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                dtype=torch.float32,
        )
        filters = torch.stack(
                [identity_filter, sobel_filter_x, sobel_filter_y]
        )  # (3, 3, 3)
        filters = filters.repeat((n_channels, 1, 1))  # (3 * n_channels, 3, 3)
        self.filters = filters[:, None, ...].to(
                self.device
        )  # (3 * n_channels, 1, 3, 3)

        # Update step
        self.update_module = nn.Sequential(
                nn.Conv2d(
                    3 * n_channels,
                    hidden_channels,
                    kernel_size=1,  # (1, 1)
                ),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_channels,
                    n_channels,
                    kernel_size=1,
                    bias=False,
                ),
        )

        with torch.no_grad():
            self.update_module[2].weight.zero_()

        self.to(self.device)

    def perceive(self, x):
        return nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)

    def update(self, x):
        return self.update_module(x)

    @staticmethod
    def stochastic_update(x, fire_rate):
        device = x.device

        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return x * mask  # broadcasted over all channels

    @staticmethod
    def get_living_mask(x):
        return (
            nn.functional.max_pool2d(
                x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1
            )
            > 0.1
        )

    def loss(self, x, y):
        return torch.mean(torch.square(x[:, :4, ...] - y), [-2, -3, -1])

    def forward(self, x):
        pre_life_mask = self.get_living_mask(x)

        y = self.perceive(x)
        dx = self.update(y)
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)

        x = x + dx

        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)

        return x * life_mask
