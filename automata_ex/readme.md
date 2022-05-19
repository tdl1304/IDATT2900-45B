Based on video 'Growing neural cellular automata in PyTorch': https://www.youtube.com/watch?v=21ACbWoF2Oo

Everything in this folder is from https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/automata
The project had no license

The log folder contains a trained example for tensorboard that ran for 10000 batches, that showed a decent result.
Run tensorboard with:
`tensorboard --logdir=logs`

Get parameters:
`python train.py --help`

Run args suggestion:
`python train.py -p 0 -n 10000 -s 9 -hch 32 -pool=true -th 3.33e-3`
Remove '-d cuda' to train using cpu

In pytorch edit configuration and add parameters:
`-d cuda -n 10000 -b 4`
