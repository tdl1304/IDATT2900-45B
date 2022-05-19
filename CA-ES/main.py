import torch
import argparse
import time
import os
import unicodedata
import logging

from es import ES

# rabbit 🐰
# carrot 🥕
# watermelon 🍉
# lizard 🦎

if __name__ == '__main__':
    emoji = '🥕'

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", metavar="train/graphic", help="Decides mode to run")

    parser.add_argument("-p", "--population_size", type=int, default=6, metavar=128, help="Population size")
    parser.add_argument("--n_iterations", type=int, default=2_000_000, help="Number of iterations to train for.")
    parser.add_argument("--pool_size", type=int, default=1, help="Size of the training pool, zero if training without pool")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Frequency for various saving/evaluating/logging",)
    parser.add_argument("--n_channels", type=int, default=16, help="Number of channels of the input tensor")
    parser.add_argument("--hidden_size", type=int, default=32, help="Number of hidden channels")

    parser.add_argument("--fire_rate", type=float, default=0.5, metavar=0.5, help="Cell fire rate")
    parser.add_argument("--lr", type=float, default=0.005, metavar=0.005, help="Learning rate") 
    parser.add_argument("--sigma", type=float, default=0.01, metavar=0.01, help="Sigma")

    parser.add_argument("-i", "--img", type=str, default=emoji, metavar="🐰", help="The emoji to train on")
    parser.add_argument("-s", "--size", type=int, default=15, help="Image size")
    parser.add_argument("--padding", type=int, default=0, help="Padding. The shape after padding is (h + 2 * p, w + 2 * p).")
    parser.add_argument("--logdir", type=str, default="logg", help="Logging folder for new model")
    parser.add_argument("-l", "--load_model_path", type=str, default="", help="Path to pre trained model")

    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        raise Exception("Logging directory '%s' not found in base folder" % args.logdir)

    args.logdir = "%s/%s-%s_%s" % (args.logdir, unicodedata.name(args.img), args.mode, time.strftime("%d-%m-%Y_%H-%M-%S"))
    os.mkdir(args.logdir)

    logging.basicConfig(filename='%s/logfile.log' % args.logdir, encoding='utf-8', level=logging.INFO)
    argprint = "\nArguments:\n"
    for arg, value in vars(args).items():
        argprint += ("%s: %r\n" % (arg, value))
    logging.info(argprint)

    es = ES(args)
    torch.set_num_threads(1) # disable pytorch's built in parallelization

    match args.mode:
        case "train": 
            os.mkdir(args.logdir + "/models")
            os.mkdir(args.logdir + "/pic")
            es.train()
        case "graphic": es.generate_graphic()
