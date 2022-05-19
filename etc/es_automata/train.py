# @title Imports
import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchsummary import summary
from model import CAModel
from utils.imageutils import *
from utils.video import *
from ES import ES, setModelParams
from utils.hyperparams import *


# Logs
log_path = pathlib.Path(logdir)
log_path.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_path)

# Target image
target_img_ = load_image(img, size)
p = padding
target_img_ = nn.functional.pad(target_img_, (p, p, p, p), "constant", 0)
target_img = target_img_.to(device)
target_img = target_img.repeat(batch_size, 1, 1, 1)

writer.add_image("ground truth", to_rgb(target_img_)[0])

# Model and optimizer
model = CAModel(n_channels=n_channels, device=device)
ESoptim = ES(target_img=target_img, SIGMA=SIGMA, LR=LR, MIN_ERROR_WEIGHT=MIN_ERROR_WEIGHT, DECAY_RATE=DECAY_RATE,
             POPULATION_SIZE=POPULATION_SIZE, TOP_N=TOP_N, device=device, DECAY=DECAY)
# @title {vertical-output:true}
summary(model, (n_channels, size, size))
# Pool initialization
seed = make_seed(size, n_channels).to(device)
seed = nn.functional.pad(seed, (p, p, p, p), "constant", 0)
pool = seed.clone().repeat(pool_size, 1, 1, 1)

model = CAModel(n_channels=n_channels, device=device)
mother_vector = nn.utils.parameters_to_vector(model.parameters())

loss_log = []
loss_mean = []
model.train()
with torch.no_grad():
    for it in tqdm(range(n_batches)):
        # testing with pool size and batch size equals 1
        batch_ixs = np.random.choice(
            pool_size, batch_size, replace=False
        ).tolist()
        x = pool[batch_ixs]
        for i in range(np.random.randint(64, 96)):
            x = model(x)

        # take a step to optimize model parameters
        mother_vector = ESoptim.es_step(model=model, x=x, mother_vector=mother_vector)

        # calculate fitness of new mother vector and set mother vec
        setModelParams(model, mother_vector)
        loss, loss_batch = ESoptim.batch_loss(x)
        loss_log.append(loss.detach().cpu())

        # Make reward graph
        if it % 50 == 0 and it != 0:
            # output.clear()
            rm = np.mean(loss_log[50:])
            loss_mean.append((it, rm))
            plt.figure(figsize=(15, 5), dpi=80)
            plt.ylabel('Reward (log)')
            plt.xlabel('Training steps')
            plt.plot(loss_log, '-', alpha=0.7, linewidth=4)
            plt.plot(*zip(*loss_mean), 'r-')
            plt.yscale('log')
            plt.show()
            print(f"Iteration: {it}, Reward:{loss}, Last 50 reward mean:{rm}")
            #display(x[:, :4, ...].detach().cpu())

        # Pool stuff
        argmin_batch = loss_batch.argmin().item()  # find the batch with lowest reward
        argmin_pool = batch_ixs[argmin_batch]
        remaining_batch = [i for i in range(batch_size) if i != argmin_batch]  # remove arg min batch from batches
        remaining_pool = [i for i in batch_ixs if i != argmin_pool]  # remove arg min pool from pools
        pool[argmin_pool] = seed.clone()  # remove in pool the one with lowest score
        pool[remaining_pool] = x[remaining_batch].detach()  # set pool to remaining ones

        """
        # Evaluation video for tensorboard
        if it % eval_frequency == 0:
            x_eval = seed.clone()  # (1, n_channels, size, size)
  
            eval_video = torch.empty(1, eval_iterations, 3, *x_eval.shape[2:])
  
            for it_eval in range(eval_iterations):
                x_eval = model(x_eval)
                x_eval_out = to_rgb(x_eval[:, :4].detach().cpu())
                eval_video[0, it_eval] = x_eval_out
  
            writer.add_video("eval", eval_video, it, fps=60)
        """


make_animation(model=model, x=seed.clone(), n_iterations=200)