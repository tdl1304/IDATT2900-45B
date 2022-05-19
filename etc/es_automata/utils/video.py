# @title Animation { vertical-output: true, form-width: "40%" }
# comment out to not skip

import matplotlib.animation as animation
from matplotlib import rc
import matplotlib.pyplot as plt
from .imageutils import drawfunc


def make_animation(model, x, n_iterations, save=True):
    rc('animation', html='jshtml')
    model.training = False
    fig, ax = plt.subplots()

    ims = []
    for i in range(n_iterations):
        if i == 0:
            ims.append([ax.imshow(drawfunc(x[:, :4, ...].detach().cpu()))])
        x = model(x)
        im = drawfunc(x[:, :4, ...].detach().cpu())
        im = ax.imshow(im)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    if save:
        f = r"./animation.mp4"
        writervideo = animation.FFMpegWriter(fps=60)
        ani.save(f, writer=writervideo)