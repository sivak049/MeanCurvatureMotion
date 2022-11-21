import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math as m
import stdexamples
import utils

plt.ion()

mesh_size = 100
h = 1/(mesh_size-1)
h2 = h**2
dt = 0.5*h2
mode = "improv" # adam_8; adam_12; adam_16; improv
boundary = "Dirichlet" # "Neumann" or "Dirichlet" boundary condition
plot_style = "surface" # "contour" or "surface"

X_comp,Y_comp = utils.init_grid(mesh_size, mode)
u_comp = 10 * X_comp**2 + 4*Y_comp**2 -1

stdexample_category = "default" # "ellipse" for ellipse; 
            # "circle" for Circle (with accuracy measuring)
            # "fattening" for fattening example
            # "default" for default

if stdexample_category in ["ellipse", "circle", "fattening"]:
    u_comp = stdexamples.init_u(stdexample_category,X_comp,Y_comp, u_comp)

fig = plt.figure()
ax = utils.init_ax(plot_style, fig)

for i in range(2490):
    j = utils.get_j(mode)
    u_comp[j:-j,j:-j] = utils.mcmPde(u_comp, mode, boundary, dt, h2)    
    if i % 25 == 0:
        cs = utils.plots(X_comp, Y_comp, u_comp, stdexample_category, plot_style, ax, mode, fig)
