
import time

import numpy as np
import sys
import pygame
import torch
import torch.nn.functional as F
from PIL import Image
from torch import tensor as tt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from model import CAModel, CellularAutomataModel
from utils import load_emoji, to_rgb, adv_attack, make_seed, to_rgb_ad

class Interactive:
    def __init__(self, args):
        self.n_channels = args.n_channels
        self.hidden_size = args.hidden_size
        self.fire_rate = args.fire_rate
        self.logdir = args.logdir
        self.size = args.size
        self.es = args.es
        self.eps = args.eps
        self.emoji_size = args.emoji_size
        self.imgpath = '%s/one.png' % (self.logdir)
        self.isRepaired = False
        self.l_func = torch.nn.MSELoss()
        if self.es:
            self.target_img = load_emoji(args.img, self.emoji_size)
        else:
            self.target_img = torch.from_numpy(load_emoji(
                args.img, self.emoji_size)).permute(2, 0, 1)[None, ...]
        self.writer = SummaryWriter(self.logdir)
        # auto calculate padding
        p = (self.size-self.emoji_size)//2
        self.size = self.emoji_size+2*p

        if self.es:
            self.pad_target = F.pad(tt(self.target_img), (0, 0, p, p, p, p))
            self.net = CellularAutomataModel(
                n_channels=self.n_channels, fire_rate=self.fire_rate, hidden_channels=self.hidden_size)
            h, w = self.pad_target.shape[:2]
            self.seed = np.zeros([h, w, self.n_channels], np.float64)
            self.seed[h // 2, w // 2, 3:] = 1.0
            whidden = torch.concat((self.pad_target.detach(), torch.zeros((self.size,self.size,12))), axis=2)
            self.batch_target = np.repeat(whidden.clone().detach()[None, ...], 1, 0)
        else:
            self.net = CAModel(n_channels=args.n_channels,
                               hidden_channels=args.hidden_size)
            self.seed = make_seed(self.size, args.n_channels)
            self.pad_target = F.pad(
                self.target_img, (p, p, p, p), "constant", 0)
            self.pad_target = self.pad_target.repeat(1, 1, 1, 1)
            whidden = torch.concat((self.pad_target[0].detach(), torch.zeros((12,self.size,self.size))), axis=0)
            self.batch_target = np.repeat(whidden.clone().detach()[None, ...], 1, 0).float()

        

        if args.load_model_path != "":
            self.load_model(args.load_model_path)

    def load_model(self, path):
        """Load a PyTorch model from path."""
        self.net.load_state_dict(torch.load(path))
        if self.es:
            self.net.double()

    def seedclone(self):
        if self.es:
            return tt(np.repeat(self.seed[None, ...], 1, 0))
        else:
            return self.seed.clone()

    def game_update(self, surface, cur_img, sz):
        nxt = np.zeros((cur_img.shape[0], cur_img.shape[1]))

        for r, c, _ in np.ndindex(cur_img.shape):
            pygame.draw.rect(surface, cur_img[r, c], (c*sz, r*sz, sz, sz))

        return nxt

    def save_cell(self, x, path):
        if self.es:
            image = to_rgb(x).permute(0, 3, 1, 2)
        else:
            image = to_rgb_ad(x[:, :4].detach().cpu())
        save_image(image, path, nrow=1, padding=0)


    def interactive(self):
        """Do damage on model using pygame"""
        x_eval = self.seedclone()

        cellsize = 20

        pygame.init()
        surface = pygame.display.set_mode(
            (self.size * cellsize, self.size * cellsize))
        pygame.display.set_caption("Interactive CA-ES")

        damaged = 100
        counter = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # damage
                    if damaged == 100:
                        dmg_size = 20
                        mpos_x, mpos_y = event.pos
                        mpos_x, mpos_y = mpos_x // cellsize, mpos_y // cellsize
                        # mpos_y = (self.size // 2) + 1
                        # mpos_x = 0
                        dmg_size = self.size
                        if self.es:
                            x_eval[:, mpos_y:mpos_y + dmg_size,
                                   mpos_x:mpos_x + dmg_size, :] = 0
                        else:
                            x_eval[:, :, mpos_y:mpos_y + dmg_size,
                                   mpos_x:mpos_x + dmg_size] = 0
                        # damaged = 0 # number of steps to record loss after damage has occurred

                        # # For noise:
                        # l_func = torch.nn.MSELoss()
                        # e = x_eval.detach().cpu()
                        # e.requires_grad = True
                        # l = l_func(e, self.batch_target)
                        # self.net.zero_grad()
                        # l.backward()
                        # x_eval = adv_attack(x_eval, self.eps, e.grad.data)
                        pygame.display.set_caption("Saving loss...")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # reset when pressing r
                        x_eval = self.seedclone()

            x_eval = self.net(x_eval)
            self.save_cell(x_eval, self.imgpath)
            cur_path = f'{self.logdir}/{counter}.png'

            # if counter < 40:
            #     # For noise:
            #     e = x_eval.clone().detach().cpu()
            #     e.requires_grad = True
            #     l = self.l_func(e, self.batch_target)
            #     self.net.zero_grad()
            #     l.backward()
            #     x_eval = adv_attack(x_eval, self.eps, e.grad.data)
            # if counter in [40, 60, 100, 150, 200, 400]:
            #     self.save_cell(x_eval, cur_path)

            # Quadratic erasing at 51
            # if counter == 51:
            #     # record loss before dmg
            #     before_loss = self.net.loss(x_eval, self.pad_target)
            #     # For lower half:
            #     mpos_y = (self.size // 2) + 1
            #     mpos_x = 0
            #     dmg_size = self.size
            #     # damage then save image
            #     if self.es:
            #         x_eval[:, mpos_y:mpos_y + dmg_size,
            #                mpos_x:mpos_x + dmg_size, :] = 0
            #     else:
            #         x_eval[:, :, mpos_y:mpos_y + dmg_size,
            #                mpos_x:mpos_x + dmg_size] = 0
            #     self.save_cell(x_eval, cur_path)

            # damage x by x in middle of image:
            if counter == 51:
                dmg_size = 3
                mpos_y = (self.size // 2) - 1
                mpos_x = (self.size // 2) - 1
                # damage then save image
                if self.es:
                    x_eval[:, mpos_y:mpos_y + dmg_size,
                           mpos_x:mpos_x + dmg_size, :] = 0
                else:
                    x_eval[:, :, mpos_y:mpos_y + dmg_size,
                           mpos_x:mpos_x + dmg_size] = 0
                self.save_cell(x_eval, cur_path)


            loss = self.net.loss(x_eval, self.pad_target)
            self.writer.add_scalar("train/fit", loss, counter)

            # find the it that the image was repaired with tol=7 difference FITS together with quadratic erasing
            # if counter > 51 and loss/before_loss <= 7 and not self.isRepaired:
            #     self.save_cell(x_eval, cur_path)
            #     self.isRepaired = True

            # # For manual damage:
            # if damaged < 100:
            #     loss = self.net.loss(x_eval, self.pad_target)
            #     self.writer.add_scalar("train/fit", loss, damaged)

            #     if damaged == 99:
            #         pygame.display.set_caption("Interactive CA-ES")
            #     damaged += 1

            # Saving and loading each image as a quick hack to get rid of the batch dimension in tensor
            image = np.asarray(Image.open(self.imgpath))
            self.game_update(surface, image, cellsize)
            # time.sleep(0.005)  # update delay
            counter += 1
            pygame.display.update()
            if counter == 1000:
                print('Reached 400 iterations. Shutting down...')
                pygame.quit()
                sys.exit()


    def generate_graphic(self):
        model = self.net
        x_eval = self.seedclone()
        pics = []
        pics.append(to_rgb(x_eval).permute(0, 3, 1, 2))

        for eval in range(40):
            x_eval = model(x_eval)
            if eval in [4, 9, 20, 39]: # frames to save img of
                if self.es:
                    image = to_rgb(x_eval).permute(0, 3, 1, 2)
                else:
                    image = to_rgb_ad(x_eval[:, :4].detach().cpu())
                pics.append(image)

        save_image(torch.cat(pics, dim=0), '%s/graphic.png' % (self.logdir), nrow=len(pics), padding=0)
        
