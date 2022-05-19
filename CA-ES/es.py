import copy
import math
from tqdm import trange

import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.nn.functional as F
from torch import tensor as tt

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from model import CellularAutomataModel
from utils import load_emoji, save_model, to_rgb, Pool

import logging

class ES:
    def __init__(self, args):
        self.population_size = args.population_size
        self.n_iterations = args.n_iterations
        self.pool_size = args.pool_size
        self.batch_size = args.batch_size
        self.eval_freq = args.eval_freq
        self.n_channels = args.n_channels
        self.hidden_size = args.hidden_size

        self.fire_rate = args.fire_rate
        self.lr = args.lr
        self.sigma = args.sigma

        self.size = args.size + 2 * args.padding
        self.target_img = load_emoji(args.img, self.size)
        self.padding = args.padding

        self.logdir = args.logdir

        self.decay_state = 0

        p = self.padding
        self.pad_target = F.pad(tt(self.target_img), (0, 0, p, p, p, p))
        h, w = self.pad_target.shape[:2]
        self.seed = np.zeros([h, w, self.n_channels], np.float64)
        self.seed[h // 2, w // 2, 3:] = 1.0

        self.net = CellularAutomataModel(n_channels=self.n_channels, fire_rate=self.fire_rate, hidden_channels=self.hidden_size)
        self.net.share_memory()
        
        self.param_shape = [tuple(p.shape) for p in self.net.parameters()]

        self.pool = Pool(self.seed, self.pool_size)

        if args.load_model_path != "":
            self.load_model(args.load_model_path)
            self.lr = 0.00075 # test
            self.decay_state = 2
    
        t_rgb = to_rgb(self.pad_target).permute(2, 0, 1)

        if args.mode == "train":
            save_image(t_rgb, "%s/target_image.png" % self.logdir)
            self.writer = SummaryWriter(self.logdir)

    def load_model(self, path):
        """Load a PyTorch model from path."""
        self.net.load_state_dict(torch.load(path))
        self.net.double()

    def fitness_shape(self, x):
        """Sort x and and map x to linear values between -0.5 and 0.5"""
        shaped = np.zeros(len(x))
        shaped[x.argsort()] = np.arange(len(x), dtype=np.float64)
        shaped /= (len(x) - 1)
        shaped -= 0.5
        shaped = (shaped - shaped.mean()) / shaped.std()
        return shaped

    def update_parameters(self, fitnesses, epsilons):
        """Update parent network weights using evaluated mutants and fitness."""
        fitnesses = self.fitness_shape(fitnesses)

        for i, e in enumerate(epsilons):
            for j, w in enumerate(self.net.parameters()):
                w.data += self.lr * 1 / (self.population_size * self.sigma) * fitnesses[i] * e[j]

    def get_population(self):
        """Return an array with values sampled from N(0, sigma)"""
        epsilons = []

        for _ in range(int(self.population_size / 2)):
            e = []
            e2 = []
            for w in self.param_shape:
                j = np.random.randn(*w) * self.sigma
                e.append(j)
                e2.append(-j)
            epsilons.append(e)
            epsilons.append(e2)

        return np.array(epsilons, dtype=np.object)

    def step(self, model_try, x):
        """Perform a generation of CA using trained net."""
        torch.seed()
        iter_n = torch.randint(30, 40, (1,)).item()
        for _ in range(iter_n): x = model_try(x)

        loss = self.net.loss(x, self.pad_target)
        loss = torch.mean(loss)

        return x, loss.item()

    def fitness(self, epsilon, x0, pid, q=None):
        """start a generation of ES."""
        model_try = copy.deepcopy(self.net)
        if epsilon is not None:
            for i, w in enumerate(model_try.parameters()):
                w.data += torch.tensor(epsilon[i])

        x, loss = self.step(model_try, x0)
        fitness = -loss

        if not math.isfinite(fitness):
            raise ValueError('Encountered non-number value in loss. Fitness ' + str(fitness) + '. Loss: ' + str(loss))
        q.put((x, fitness, pid))
        return

    def decay_lr(self, fitness):
        # Fitness treshholds for adjusting learning rate
        # fit_t1 = -0.06 # testing for size ~20
        # fit_t2 = -0.03

        # fit_t1 = -0.05 # works well for size ~15
        # fit_t2 = -0.02

        fit_t1 = -0.03 # used for size 9
        fit_t2 = -0.01

        if not self.decay_state == 2:
            if fitness >= fit_t1 and self.decay_state == 0:
                reduce = 0.3
                self.lr *= reduce
                self.decay_state += 1
                logging.info("Fitness higher than than %.3f, lr set to %.5f (*%.2f)" % (fit_t1, self.lr, reduce))
            elif fitness >= fit_t2 and self.decay_state == 1:
                reduce = 0.5
                self.lr *= reduce
                self.decay_state += 1
                logging.info("Fitness higher than %.3f, lr set to %.5f (*%.2f)" % (fit_t2, self.lr, reduce))


    def evaluate_main(self, x0):
        """Return output and fitness from a generation using unperturbed weights/coeffs"""
        x_main, loss_main = self.step(self.net, x0.clone())
        fit_main = - loss_main
        return x_main, fit_main


    def train(self):
        """main training loop"""
        logging.info("Starting training")

        x0 = tt(np.repeat(self.seed[None, ...], self.batch_size, 0)) #seed
        _, _ = self.step(self.net, x0.clone())

        processes = []
        q = tmp.Manager().Queue()

        t = trange(self.n_iterations, desc='Mean reward:', leave=True)
        for iteration in t:
            if self.pool_size > 1:
                batch = self.pool.sample(self.batch_size)
                x0 = batch["x"]
                loss_rank = self.net.loss(tt(x0), self.pad_target).numpy().argsort()[::-1]
                x0 = x0[loss_rank]
                x0[:1] = self.seed
                x0 = tt(x0)
            
            epsilons = self.get_population()
            fitnesses = np.zeros(self.population_size, dtype=np.float64)
            xs = torch.zeros(self.population_size, *x0.shape, dtype=torch.float64)
            
            for i in range(self.population_size):
                p = tmp.Process(target=self.fitness, args=(epsilons[i], x0.clone(), i, q))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                x, fit, pid = q.get()
                fitnesses[pid] = fit
                xs[pid] = x
            processes = []

            if self.pool_size > 1:
                idx = np.argmax(fitnesses)
                batch["x"][:] = xs[idx]
                self.pool.commit(batch)

            fitnesses = np.array(fitnesses).astype(np.float64)
            self.update_parameters(fitnesses, epsilons)

            # Logging
            mean_fit = np.mean(fitnesses)
            self.writer.add_scalar("train/fit", mean_fit, iteration)
            
            if iteration % 10 == 0:
                t.set_description("Mean reward: %.4f    " % mean_fit, refresh=True)

            if (iteration) % self.eval_freq == 0:
                self.decay_lr(mean_fit)

                # Save picture of model
                x_eval = x0.clone()
                pics = []
                growth_loss = []
                for eval in range(201):
                    x_eval = self.net(x_eval)
                    if eval < 41 and eval % 5 == 0:
                        pics.append(to_rgb(x_eval).permute(0, 3, 1, 2))
                    if eval in [40, 200]: growth_loss.append(self.net.loss(x_eval, self.pad_target))

                self.writer.add_scalar("growth_loss/40", growth_loss[0], iteration)
                self.writer.add_scalar("growth_loss/200", growth_loss[1], iteration)
                save_image(torch.cat(pics, dim=0), '%s/pic/big%04d.png' % (self.logdir, iteration), nrow=1, padding=0)
                save_model(self.net, self.logdir + "/models/model_" + str(iteration))

