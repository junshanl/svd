import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

a = np.random.multivariate_normal([1,1], [[1,1],[1,2]], 30)


fig = plt.figure()
plt.scatter(a[:,0], a[:, 1])

c = np.cov(a.T)

u, _, _ = np.linalg.svd(c)

print u

def extended(ax, x, y, **args):

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_ext = np.linspace(xlim[0], xlim[1], 100)
    p = np.polyfit(x, y , deg=1)
    y_ext = np.poly1d(p)(x_ext)
    ax.plot(x_ext, y_ext, **args)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


ax = plt.gca()
ax.spines['bottom'].set_position(('data', 1))
ax.spines['left'].set_position(('data',1))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
extended(ax, [1, u[0,0]],[1, u[0, 1]] )

fig.savefig('2.png')
