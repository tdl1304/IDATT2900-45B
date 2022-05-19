import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logdir = 'logs'
img = 'rabbit.png'
padding = 1
size = 40
batch_size = 4  # default 8
n_batches = 1000  # default 5000
pool_size = 8 # default 1024
n_channels = 16
#eval_frequency = 50  # default 500
#eval_iterations = 30  # default 300

SIGMA = 0.01
LR = 0.001
MIN_ERROR_WEIGHT = 0.001
DECAY_RATE = 0.95
POPULATION_SIZE = 100
TOP_N = 50
DECAY = False