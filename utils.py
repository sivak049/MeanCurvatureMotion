import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math as m

def init_grid(n, s):
    h = 1/(n-1)
    if s == "adam_8" or s == "improv":
        return np.mgrid[-1-h:1+3*h/2:h,-1-h:1+3*h/2:h]
    elif s == "adam_12":
        return np.mgrid[-1-2*h:1+5*h/2:h,-1-2*h:1+5*h/2:h]
    elif s == "adam 16":
        return np.mgrid[-1-3*h:1+7*h/2:h,-1-3*h:1+7*h/2:h] 
    else: return np.mgrid[-1-h:1+3*h/2:h,-1-h:1+3*h/2:h] 


def init_u_plot(u_comp, mode):
    if mode == "adam_8" or mode == "improv":
        return u_comp[-1:1,-1:1]
    elif mode == "adam_12":
        return u_comp[-2:2,-2:2]
    elif mode == "adam 16":
        return np.mgrid[-3:3,-3:3] 
    else: return u_comp[-1:1,-1:1]

def init_ax(plot_style, fig):
    if plot_style == "surface":
        return fig.gca(projection = '3d')
    else: return fig.gca()

def neighbors(u, mode, boundary):
    if mode == "adam_8" or mode == "improv":
        neighbors = np.stack((u[:-2,:-2],u[:-2,1:-1],
        u[:-2,2:],u[1:-1,:-2],u[1:-1,2:],u[2:,:-2],
        u[2:,1:-1],u[2:,2:]),axis=2)
        if boundary == "Neumann":
            neighbors = np.pad(neighbors[1:-1,1:-1], 1, mode='reflect')[:,:,1:-1]
        return neighbors    
    elif mode == "adam_12":
        neighbors = np.stack((u[:-4,1:-3], u[:-4,2:-2],
        u[:-4,3:-1],u[1:-3,:-4],u[1:-3,4:],u[2:-2,:-4],
        u[2:-2,4:],u[3:-1,:-4],u[3:-1,4:],u[4:,1:-3],
        u[4:,2:-2],u[4:,3:-1]),axis=2)
        if boundary == "Neumann":
            neighbors = np.pad(neighbors[2:-2,2:-2], 2, mode='reflect')[:,:,2:-2]
        return neighbors    
    elif mode == "adam_16":
        neighbors = np.stack((u[:-6,2:-4],u[:-6,3:-3],
        u[:-6,4:-2],u[6:,4:-2],u[6:,2:-4],u[6:,3:-3],u[2:-4,:-6],
        u[3:-3,:-6],u[4:-2,:-6],u[2:-4,6:],u[3:-3,6:],u[4:-2,6:],
        u[1:-5,5:-1],u[5:-1,5:-1],u[1:-5,1:-5],u[5:-1,1:-5]),
        axis=2)
        if boundary == "Neumann":
            neighbors = np.pad(neighbors[3:-3,3:-3], 3, mode='reflect')[:,:,3:-3]
        return neighbors    
    else:
        neighbors = np.stack((u[:-2,:-2],u[:-2,1:-1],
        u[:-2,2:],u[1:-1,:-2],u[1:-1,2:],u[2:,:-2],
        u[2:,1:-1],u[2:,2:]),axis=2)
        if boundary == "Neumann":
            neighbors = np.pad(neighbors[1:-1,1:-1], 1, mode='reflect')[:,:,1:-1]
        return neighbors

def find_multiplier(u_neighbours, mode):
    if mode == "improv":
        p = m.pow(2,1/2)
        distances = np.array([p,1,p,1,1,p,1,p])
        u_neighbours_sorted = np.argsort(u_neighbours,axis=2)
        dist = (distances[u_neighbours_sorted[:,:,3]] + distances[u_neighbours_sorted[:,:,4]])/2
        multiplier = dist**2
        return multiplier
    return np.ones(shape = u_neighbours[:,:,0].shape)

def circle_error(cs, i, dt):
    coordinates = np.array(cs.allsegs[0][0])
    coordinates_x = np.split(coordinates,2,axis=1)[0]
    coordinates_y = np.split(coordinates,2,axis=1)[1]
    radius = np.power((np.power(coordinates_x,2) + np.power(coordinates_y,2)),1/2)
    mean_radius = np.mean(radius)
    std_radius = np.std(radius)
    true_radius = m.pow(2,1/2) * m.pow(1/8 - i * dt, 1/2)
    if i in range(0,2401,25):
        error = (abs(mean_radius-true_radius)/true_radius) * 100
        print(i,mean_radius,true_radius, error,std_radius)

def get_j(mode):
    if mode == "adam_8" or mode == "improv":
        return 1
    elif mode == "adam_12":
        return 2
    elif mode == "adam_16":
        return 3
    else:
        return 1

def mcmPde(u_comp, mode, boundary, dt, h2):
    u_neighbors = neighbors(u_comp, mode, boundary)
    multiplier = find_multiplier(u_neighbors, mode)
    u_median = np.median(u_neighbors,axis=2)
    j = get_j(mode)
    return u_comp[j:-j,j:-j] + 2 * (dt/h2) * (u_median[:,:] - u_comp[j:-j,j:-j])/multiplier

def plots(X_comp, Y_comp, u_comp, stdexample_category, plot_style, ax, mode, fig):
    j = get_j(mode)
    if plot_style == "surface":
        cs = ax.plot_surface(X_comp[j:-j,j:-j], Y_comp[j:-j,j:-j], u_comp[j:-j,j:-j], edgecolors='black')
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        cs.remove()
        return cs
    else:
        if stdexample_category != "fattening":
            cs = ax.contour(X_comp[j:-j,j:-j],Y_comp[j:-j,j:-j],u_comp[j:-j,j:-j], [0])
            ax.axis('equal')
        else:
            cs = ax.contour(X_comp[j:-j,j:-j],Y_comp[j:-j,j:-j],u_comp[j:-j,j:-j], [-0.02,0.02])
            ax.axis('equal')
        fig.canvas.draw()
        fig.canvas.flush_events()
        for coll in cs.collections:
            plt.gca().collections.remove(coll) 
        return cs
    

        











              

