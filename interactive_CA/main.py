import time
import logging
import argparse
import time
import os
import unicodedata
import logging

from interactive import Interactive

# rabbit 🐰
# carrot 🥕
# watermelon 🍉
# lizard 🦎

if __name__ == '__main__':
    # pick model [Index of model types][index of 9x9 or 15x15 rabbit or carrot]
    # model = models[1][2]  # change only this or size

    # Auto determined
    # load_model = model[0]
    # emoji = model[1]
    # emoji_size = model[2]  # size of training image usually 9 or 15
    # es = model[3]

    # canvas size
    # size = emoji_size

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img", type=str, default='',
                        metavar="🐰", help="The emoji to train on")
    parser.add_argument("-s", "--size", type=int,
                        default=9, help="Image size")
    parser.add_argument("--logdir", type=str, default="interactive_CA/logs",
                        help="Logging folder for new model")
    parser.add_argument("-l", "--load_model_path", type=str,
                        default='', help="Path to pre trained model")
    parser.add_argument("--n_channels", type=int, default=16,
                        help="Number of channels of the input tensor")
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="Number of hidden channels")
    parser.add_argument("--fire_rate", type=float, default=0.5,
                        metavar=0.5, help="Cell fire rate")
    parser.add_argument("-e", "--es", type=str, default='',
                        metavar=True, help="ES or adam")
    parser.add_argument("--eps", type=float, default=7e-3,
                        help="Epsilon scales the amount of damage done from adversarial attacks")

    args = parser.parse_args()
    args.emoji_size = args.size

    if not os.path.isdir(args.logdir):
        raise Exception(
            "Logging directory '%s' not found in base folder" % args.logdir)

    match args.es:
        case 'True': #heh
            method = 'ES'
            args.es = True
        case 'False': 
            method = 'ADAM'
            args.es = False
        

    args.logdir = "%s/%s-%s-%s_%s" % (args.logdir, args.emoji_size, method,
                                      unicodedata.name(args.img), time.strftime("%d-%m-%Y_%H-%M-%S"))
    os.mkdir(args.logdir)

    logging.basicConfig(filename='%s/logfile.log' %
                        args.logdir, encoding='utf-8', level=logging.INFO)
    argprint = "\nArguments:\n"
    for arg, value in vars(args).items():
        argprint += ("%s: %r\n" % (arg, value))
    logging.info(argprint)

    Interactive = Interactive(args)

    Interactive.interactive()
    # Interactive.generate_graphic_es()
