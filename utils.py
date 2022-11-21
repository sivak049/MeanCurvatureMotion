"""
Utilities
==========

This module implements several useful functions that are 
used throughout main.
"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math as m

def init_grid(n, s):
    """Grid/Mesh initializer
    ===================

    Initializes the grid/mesh to suit the requirements of 
    the chosen stencil.

    Parameters
    ----------
    n : int
        length of the side of the square grid/mesh
    s : string
        The chosen stencil mode. Available stencil modes
        are Adam_8, Adam_12, Adam_16, and improv.

    Returns
    -------
    X_comp, Y_comp : (n,n) numpy "meshgrid"
    """
    h = 1/(n-1)
    if s == "adam_8" or s == "improv":
        return np.mgrid[-1-h:1+3*h/2:h,-1-h:1+3*h/2:h]
    elif s == "adam_12":
        return np.mgrid[-1-2*h:1+5*h/2:h,-1-2*h:1+5*h/2:h]
    elif s == "adam_16":
        return np.mgrid[-1-3*h:1+7*h/2:h,-1-3*h:1+7*h/2:h] 
    else: return np.mgrid[-1-h:1+3*h/2:h,-1-h:1+3*h/2:h] 


def init_u_plot(u_comp, mode):
    """Helper for Plot
    ==================

    Chooses how much of the meshgrid to use for plotting
    since there are additional rows and colums computed only
    as neighbor data.

    Parameters
    ----------
    u_comp : (n,n) numpy meshgrid
             initialized mesh
    mode : string
           Specifies the stencil being used; helps decide the number 
           of rows/colums to plot during the plotting stage.

    Returns
    -------
    subset of the original meshgrid to be the domain of the plot.
    


    """
    if mode == "adam_8" or mode == "improv":
        return u_comp[-1:1,-1:1]
    elif mode == "adam_12":
        return u_comp[-2:2,-2:2]
    elif mode == "adam 16":
        return np.mgrid[-3:3,-3:3] 
    else: return u_comp[-1:1,-1:1]

def init_ax(plot_style, fig):
    """Plot mode Setup  
    ==========

    Chooses between contour and surface plot, and finishes 
    some required setup.

    Parameters
    ----------
    plot_style : string
                contour or surface
    fig : Initialized MatPlotLib plotter object

    Returns
    -------
    MatPlotLib plotter object with completed setup

    """
    if plot_style == "surface":
        return fig.gca(projection = '3d')
    else: return fig.gca()

def neighbors(u, mode, boundary):
    """List of Neighbors
    ====================

    Hardcodes the list of neighboring points for 
    each point in the meshgrid based on the chosen stencil
    and boundary condition.

    Parameters
    ----------
    u : (n,n) numpy meshgrid
    mode: string
        specifies stencil: adam_8, adam_12, adam_16, and improv
    boundary: string
        specifies Dirichlet or Neumann boundary conditions.

    Returns
    -------
    Multidimensional array of neighboring points for the meshgrid.    
                   
    """
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
    """PDE multiplier Computation
    =========================

    Computes the additional multiplier in the case of the improved
    stencil.

    Parameters
    ----------
    u_neighbours : Multi-dimesional numpy array
                   Stores the list of neighbors.
    mode: string
          specifies stencil mode

    Return
    ------
    A numpy array of the computed multipliers 
    """
    if mode == "improv":
        p = m.pow(2,1/2)
        distances = np.array([p,1,p,1,1,p,1,p])
        u_neighbours_sorted = np.argsort(u_neighbours,axis=2)
        dist = (distances[u_neighbours_sorted[:,:,3]] + distances[u_neighbours_sorted[:,:,4]])/2
        multiplier = dist**2
        return multiplier
    return np.ones(shape = u_neighbours[:,:,0].shape)

def circle_error(cs, i, dt):
    """Error computation
    ================

    Computes the absolute error and deviation in the
    case of a circular initial level set

    Parameters
    ---------
    cs : Contour/surface plotter object
    i : int
        current iteration step
    dt : float
        current timestep per iteration

    Returns
    -------
    NULL. Function prints the error merics during the later iterations of the 
    PDE's 
    """
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
    """Helper
    =========

    A helper function that allows for handling all stencil 
    modes with just a single expression

    Parameters
    ----------
    mode: string
        Specifies the chosen stencil

    Returns
    -------
    int 1, 2 or 3 depending on stencil mode
    """
    if mode == "adam_8" or mode == "improv":
        return 1
    elif mode == "adam_12":
        return 2
    elif mode == "adam_16":
        return 3
    else:
        return 1

def mcmPde(u_comp, mode, boundary, dt, h2):
    """PDE Computation
    ==================

    Computes the PDE for the current iteration.

    Parameters
    ----------
    u_comp : (n,n) numpy meshgrid
    mode: string, specifies stencil mode
    dt: float, specifies timestep per iteration
    h2: float, d^2(x)

    Returns
    -------
    numpy array storing the current state of the 
    partial differential equation
    """
    u_neighbors = neighbors(u_comp, mode, boundary)
    multiplier = find_multiplier(u_neighbors, mode)
    u_median = np.median(u_neighbors,axis=2)
    j = get_j(mode)
    return u_comp[j:-j,j:-j] + 2 * (dt/h2) * (u_median[:,:] - u_comp[j:-j,j:-j])/multiplier

def plots(X_comp, Y_comp, u_comp, stdexample_category, plot_style, ax, mode, fig):
    """Main Plotter
    ===============

    Updates the plot every few iterations.

    Parameters
    ----------

    X_comp,Y_comp : (n,n) original numpy meshgrid
    u_comp : The function we are computing 
            mean curvature motion for
    stdexample_category : Specifies if the chosen 
            function is part of the module of standard example
    plt_style: string, specifies plot style, contour or surface
    ax: MatPlotLib object for plotting on the initialized axes
    mode: string, stencil mode
    fig: MatPlotLib object for Plotting

    Returns
    -------
    New plot for the current iteration 
    """
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
       

