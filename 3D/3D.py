import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from scipy.stats import multivariate_normal
import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage import segmentation
from skimage.filters import sobel
from skimage.color import label2rgb
from numpy import asarray
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def get_bus(full_filename):
  image = Image.open(full_filename).convert('L') # Make sure to convert to grayscale
  image_inv = ImageOps.invert(image)
  bus = asarray(image_inv)
  return bus

bus = get_bus("C:/Users/djhalama/Documents/Education/DS-785/BUS Project Home/Datasets/BUS_Dataset_B/original/000129.png")
print(bus.shape)


def plot1():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from scipy.stats import multivariate_normal
    X = np.linspace(-5,5,50)
    Y = np.linspace(-5,5,50)
    X, Y = np.meshgrid(X,Y)
    X_mean = 0; Y_mean = 0
    X_var = 5; Y_var = 8
    pos = np.empty(X.shape+(2,))
    pos[:,:,0]=X
    pos[:,:,1]=Y
    rv = multivariate_normal([X_mean, Y_mean],[[X_var, 0], [0, Y_var]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap="plasma")
    plt.show()

def plot2():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    f = lambda x,y: x**3 - 3*x*y**2
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,2,1,projection='3d')
    xvalues = np.linspace(-2,2,100)
    yvalues = np.linspace(-2,2,100)
    xgrid, ygrid = np.meshgrid(xvalues, yvalues)
    zvalues = f(xgrid, ygrid)
    surf = ax.plot_surface(xgrid, ygrid, zvalues,
    rstride=5, cstride=5,
    linewidth=0, cmap=cm.plasma)
    ax = fig.add_subplot(1,2,2)
    plt.contourf(xgrid, ygrid, zvalues, 30,
    cmap=cm.plasma)
    fig.colorbar(surf, aspect=18)
    plt.tight_layout()
    plt.show()

def plot3():
    #https://machinelearningknowledge.ai/matplotlib-surface-plot-tutorial-for-beginners/
    dim = bus.shape
    x = np.arange(0,dim[0],1)
    y = np.arange(0,dim[1],1)
    X, Y = np.meshgrid(x,y)
    zs = np.array(bus[np.ravel(X), np.ravel(Y)])
    Z = zs.reshape(X.shape)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,2,1, projection='3d')
    # ax.plot_surface(X, Y, Z)
    surf = ax.plot_surface(X, Y, Z,
    rstride=5, cstride=5,
    linewidth=0, cmap=cm.plasma)
    ax = fig.add_subplot(1,2,2)
    plt.contourf(X, Y, Z, 30,cmap=cm.plasma)
    fig.colorbar(surf, aspect=18)
    # plt.tight_layout()
    plt.show()
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

def fun(x, y):
    return x**2 + y

def plot4():
    from mpl_toolkits.mplot3d import Axes3D
    import random
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-9.5, 15.0, 0.15)
    X, Y = np.meshgrid(x, y)
    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

# plot1()
# plot2()
plot3()