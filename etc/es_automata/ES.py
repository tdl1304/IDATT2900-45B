# @title ES Module { run: "auto", form-width: "20%" }
import numpy as np
import torch
from torch import nn


def setModelParams(model, solution):
    # set new params for model
    nn.utils.vector_to_parameters(solution, model.parameters())


class ES:
    def __init__(self, target_img, SIGMA, LR, MIN_ERROR_WEIGHT, DECAY_RATE, POPULATION_SIZE, TOP_N, device,
                 DECAY=False):
        self.target_img = target_img
        self.SIGMA = SIGMA
        self.LR = LR
        self.MIN_ERROR_WEIGHT = MIN_ERROR_WEIGHT
        self.DECAY_RATE = DECAY_RATE
        self.POPULATION_SIZE = POPULATION_SIZE
        self.TOP_N = TOP_N
        self.DECAY = DECAY
        self.eps = np.finfo(float).eps
        self.ERROR_WEIGHT = 1
        self.device = device

    # Fitness function
    def batch_loss(self, x):
        """Run loss on batch.
            Parameters
            ----------
            x : torch.Tensor
                Shape `(batch_size, n_channels, size+padding, size+padding)`.
            Returns
            -------
            float
                shape (1) Inverse of loss (1/loss), shape(1,batch_size) Inverse of loss (1/loss) for each in batch
        """
        loss_batch = ((self.target_img - x[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])  # Loss function
        loss = loss_batch.mean()
        return 1 / (loss + self.eps), 1 / loss_batch

    def batch_fitness_func(self, model, x, solution):  # apply solution then calculate fitness
        """ Apply solution then calculate fitness.
          Parameters
          ----------
          x : torch.Tensor
              Shape `(batch_size, n_channels, size+padding, size+padding)`.
          solution : numpy.ndarray
              Shape (8320)
          Returns
          -------
          torch.Tensor
              Inverse of loss on new solution (1/loss)
          """
        # solution is a vector of paramters like mother_parametrs
        setModelParams(model, solution)
        loss, _ = self.batch_loss(model(x))
        return loss + self.eps

    # in ES, our population is a slightly altered version of the mother parameters, so we implement a jitter function
    def jitter(self, mother_params, state_dict):
        """ Make a new parameter with specific noise.
          Parameters
          ----------
          mother_params : torch.Tensor
              Shape (8320)
          state_dict : torch.Tensor
              Shape (8320)
          Returns
          -------
          numpy.ndarray
              Shape(8320)
          """
        params_try = mother_params + self.ERROR_WEIGHT * self.SIGMA * state_dict
        return params_try

    # now, we calculate the fitness of entire population
    def batch_calculate_population_fitness(self, model, x, pop, mother_vector):
        """ Calculate population fitnesses.
          Parameters
          ----------
          :param model:
          x : torch.Tensor
              Shape `(batch_size, n_channels, size+padding, size+padding)`.
          pop : torch.Tensor
              Shape (POPULATION_SIZE, 8320)
          mother_vector : torch.Tensor
              Shape (8320)
          Returns
          -------
            (torch.Tensor, torch.Tensor)
              first of tuple with shape (POPULATION_SIZE, 1)
              and second of tuple with shape (POPULATION_SIZE, 8320)

          """
        fitness = torch.zeros(pop.shape[0], device=self.device)
        pop_weights = torch.zeros((pop.shape[0], pop.shape[1]), device=self.device)
        for i, params in enumerate(pop):
            p_try = self.jitter(mother_vector, params)
            pop_weights[i] = p_try
            fitness[i] = self.batch_fitness_func(model, x, p_try)
        return fitness, pop_weights

    def es_step(self, model, x, mother_vector):
        """ Calculate one-step new parameters based on Evolutionary Strategies method
          Parameters
          ----------
           :param model: torch model
          x : torch.Tensor
              Shape `(batch_size, n_channels, size+padding, size+padding)`.
          mother_vector : torch.Tensor
              Shape (8320)
          Returns
          -------
            torch.Tensor
              Shape (8320)

          """
        global ERROR_WEIGHT
        n_params = nn.utils.parameters_to_vector(model.parameters()).shape[0]
        # Create population in N(0, 1)
        pop = torch.rand(self.POPULATION_SIZE, n_params, device=self.device)
        # Get fitness for population and their population weights
        fitness, pop_weights = self.batch_calculate_population_fitness(model, x, pop, mother_vector)
        top_n_indices = torch.topk(fitness, self.TOP_N).indices
        F = fitness[top_n_indices]
        # Take top n fitness scores and weights
        F = (F - torch.mean(F)) / (torch.std(F) + self.eps)
        P = pop_weights[top_n_indices]
        # Update model parameters
        mother_vector += (self.LR / (self.TOP_N * self.SIGMA)) * torch.matmul(P.t(), F)
        # Decay error weight
        self.ERROR_WEIGHT = max(self.ERROR_WEIGHT * self.DECAY_RATE, self.MIN_ERROR_WEIGHT)
        if torch.nan in mother_vector:
            raise Exception('Values in mother_vector were torch.nan')
        return mother_vector
