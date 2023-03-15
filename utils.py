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
from scipy.spatial import KDTree
import math as m
import stdexamples

def init_grid(n, m, s, x_start, x_end, y_start, y_end):
    """Grid/Mesh initializer
    ===================

    Initializes the grid/mesh to suit the requirements of 
    the chosen stencil.

    Parameters
    ----------
    n : int
        length of the side of the square grid/mesh
    m : string
        The chosen stencil mode. Available stencil modes
        are adam and improv.
    s:  int
        Stencil size: 1, 2 or 3 for 8, 12, and 16 point stencils res.        

    Returns
    -------
    X_comp, Y_comp : (n,n) numpy "meshgrid"
    """
    h = (min(x_end-x_start, y_end - y_start))/((n-1)*2)
    if (m,s) == ("adam",1) or (m,s) == ("improv",1):
        return np.mgrid[x_start-h:x_end+3*h/2:h,y_start-h:y_end+3*h/2:h]
    elif (m,s) == ("adam",2) or (m,s) == ("improv",2):
        return np.mgrid[x_start-2*h:x_end+5*h/2:h,y_start-2*h:y_end+5*h/2:h]
    elif (m,s) == ("adam",3) or (m,s) == ("improv",3):
        return np.mgrid[x_start-3*h:x_end+7*h/2:h,y_start-3*h:y_end+7*h/2:h] 
    else: return np.mgrid[x_start-h:x_end+3*h/2:h,y_start-h:y_end+3*h/2:h] 


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

def neighbors(u, m, s, boundary):
    """List of Neighbors
    ====================

    Hardcodes the list of neighboring points for 
    each point in the meshgrid based on the chosen stencil
    and boundary condition.

    Parameters
    ----------
    u : (n,n) numpy meshgrid
    m: string
        specifies stencil: adam and improv
    boundary: string
        specifies Dirichlet or Neumann boundary conditions.

    Returns
    -------
    Multidimensional array of neighboring points for the meshgrid.    
                   
    """
    if (m,s) == ("adam",1) or (m,s) == ("improv",1):
        if boundary == "periodic":
            u = u[:-1,:]
        neighbors = np.stack((u[:-2,:-2],u[:-2,1:-1],
        u[:-2,2:],u[1:-1,:-2],u[1:-1,2:],u[2:,:-2],
        u[2:,1:-1],u[2:,2:]),axis=2)
        if boundary == "Neumann":
            neighbors = np.pad(neighbors[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='reflect')
        if boundary == "periodic":
            neighbors = np.pad(neighbors[1:-1,1:-1], ((1, 2),(0,0),(0,0)), mode='wrap')
            neighbors = np.pad(neighbors, ((0, 0),(1,1),(0,0)), mode='reflect')
        return neighbors    
    elif (m,s) == ("adam",2) or (m,s) == ("improv",2):
        if boundary == "periodic":
            u = u[:-1,:]
        neighbors = np.stack((u[:-4,1:-3], u[:-4,2:-2],
        u[:-4,3:-1],u[1:-3,:-4],u[1:-3,4:],u[2:-2,:-4],
        u[2:-2,4:],u[3:-1,:-4],u[3:-1,4:],u[4:,1:-3],
        u[4:,2:-2],u[4:,3:-1]),axis=2)
        if boundary == "Neumann":
            neighbors = np.pad(neighbors[2:-2,2:-2], ((2, 2),(2,2),(0,0)), mode='reflect')
        if boundary == "periodic":
            neighbors = np.pad(neighbors[2:-2,2:-2], ((2, 3),(0,0),(0,0)), mode='wrap')
            neighbors = np.pad(neighbors, ((0, 0),(2,2),(0,0)), mode='reflect')
        return neighbors    
    elif (m,s) == ("adam",3) or (m,s) == ("improv",3):
        if boundary == "periodic":
            u = u[:-1,:]
        neighbors = np.stack((u[:-6,2:-4],u[:-6,3:-3],
        u[:-6,4:-2],u[6:,4:-2],u[6:,2:-4],u[6:,3:-3],u[2:-4,:-6],
        u[3:-3,:-6],u[4:-2,:-6],u[2:-4,6:],u[3:-3,6:],u[4:-2,6:],
        u[1:-5,5:-1],u[5:-1,5:-1],u[1:-5,1:-5],u[5:-1,1:-5]),
        axis=2)
        if boundary == "Neumann":
            neighbors = np.pad(neighbors[3:-3,3:-3], ((3, 3),(3,3),(0,0)), mode='reflect')
        if boundary == "periodic":
            neighbors = np.pad(neighbors[3:-3,3:-3], ((3, 4),(0,0),(0,0)), mode='wrap')
            neighbors = np.pad(neighbors, ((0, 0),(3,3),(0,0)), mode='reflect')
        return neighbors    
    else:
        neighbors = np.stack((u[:-2,:-2],u[:-2,1:-1],
        u[:-2,2:],u[1:-1,:-2],u[1:-1,2:],u[2:,:-2],
        u[2:,1:-1],u[2:,2:]),axis=2)
        if boundary == "Neumann":
            neighbors = np.pad(neighbors[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='reflect')
        return neighbors

def find_multiplier(u_neighbours, mode, s):
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
        if s == 1:
            p = m.pow(2,1/2)
            distances = np.array([p,1,p,1,1,p,1,p])
            u_neighbours_sorted = np.argsort(u_neighbours,axis=2)
            dist = (distances[u_neighbours_sorted[:,:,3]] + distances[u_neighbours_sorted[:,:,4]])/2
            multiplier = dist**2
            return multiplier
        elif s == 2:
            p = m.pow(5,1/2)
            distances = np.array([p,2,p,p,p,2,2,p,p,p,2,p])
            u_neighbours_sorted = np.argsort(u_neighbours,axis=2)
            dist = (distances[u_neighbours_sorted[:,:,5]] + distances[u_neighbours_sorted[:,:,6]])/2
            multiplier = dist**2
            return multiplier
        elif s == 3:
            p = 2*m.pow(2,1/2)
            q = m.pow(10,1/2)
            distances = np.array([q,3,q,q,q,3,q,3,q,q,3,q,p,p,p,p])
            #distances = np.array([3,3,3,3,3,3,2,2,2,2,2,2,2.5,2.5,2.5,2.5])
            u_neighbours_sorted = np.argsort(u_neighbours,axis=2)
            dist = (distances[u_neighbours_sorted[:,:,7]] + distances[u_neighbours_sorted[:,:,8]])/2
            multiplier = dist**2
            return multiplier
    else:
        if s == 1:
            p = m.pow(2,1/2)
            distances = np.array([p,1,p,1,1,p,1,p])
            factor = np.mean(distances)
            #factor = 1 
        elif s == 2:
            p = m.pow(5,1/2)
            distances = np.array([p,2,p,p,p,2,2,p,p,p,2,p])
            factor = np.mean(distances)
            #factor = 2
        elif s == 3:
            p = 2*m.pow(2,1/2)
            q = m.pow(10,1/2)
            distances = np.array([q,3,q,q,q,3,q,3,q,q,3,q,p,p,p,p]) 
            factor = np.mean(distances)
            #factor = 3
    return factor**2 * np.ones(shape = u_neighbours[:,:,0].shape)


def mcmPde(u_comp, mode, s, boundary, dt, h2):
    """PDE Computation
    ==================

    Computes the PDE for the current iteration.

    Parameters
    ----------
    u_comp : (n,n) numpy meshgrid
    mode: string, specifies stencil mode
    s: int, stencil size: 1,2, or 3 (for 8, 12, 16 point stencil res.)
    boundary : "Dirichlet","Neumann", or "Periodic"
    dt: float, specifies timestep per iteration
    h2: float, d^2(x)

    Returns
    -------
    numpy array storing the current state of the 
    partial differential equation
    """
    u_neighbors = neighbors(u_comp, mode, s, boundary)
    multiplier = find_multiplier(u_neighbors, mode, s)
    u_median = np.median(u_neighbors,axis=2)
    return u_comp[s:-s,s:-s] + 2 * (dt/h2) * (u_median[:,:] - u_comp[s:-s,s:-s])/multiplier

def plots(X_comp, Y_comp, u_comp, stdexample_category, plot_style, ax, s, fig):
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
    s: int, stencil size: 1,2, or 3 (for 8, 12, 16 point stencil res.)
    fig: MatPlotLib object for Plotting

    Returns
    -------
    New plot for the current iteration 
    """
    if plot_style == "surface":
        cs = ax.plot_surface(X_comp[s:-s,s:-s], Y_comp[s:-s,s:-s], u_comp[s:-s,s:-s], edgecolors='black')
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        cs.remove()
        return cs
    else:
        if stdexample_category != "fattening":
            cs = ax.contour(X_comp[s:-s,s:-s],Y_comp[s:-s,s:-s],u_comp[s:-s,s:-s], [0])
            ax.axis('equal')
        else:
            cs = ax.contour(X_comp[s:-s,s:-s],Y_comp[s:-s,s:-s],u_comp[s:-s,s:-s], [-0.02,0.02])
            ax.axis('equal')
        fig.canvas.draw()
        fig.canvas.flush_events()
        #for coll in ax.collections:
        #    plt.gca().collections.remove(coll) 
        return cs

def plot_explicit(X, Y, u, stdexample_category, ax, i, dt, s):
    """Main Plotter
    ===============

    Updates the plot every few iterations.

    Parameters
    ----------

    X,Y : (n,n) original numpy meshgrid
    u : The function we are computing 
            mean curvature motion for
    stdexample_category : Specifies if the chosen 
            function is part of the module of standard example
    ax: MatPlotLib object for plotting on the initialized axes
    i: int, current timestep
    dt: float, timestep size
    s: int, stencil size: 1,2, or 3 (for 8, 12, 16 point stencil res.)

    Returns
    -------
    New plot for the current iteration 
    """
    cs2 = ax.contour(X[s:-s,s:-s],Y[s:-s,s:-s],u[s:-s,s:-s], [0], linestyles='dashed')
    return cs2

def axis_clear(ax):
    for coll in ax.collections:
        plt.gca().collections.remove(coll)

def compute_errors(stdexample_category, cs, i, dt):
    if stdexample_category == "circle":
        circle_error(cs,i, dt)
        return -1
    else:
        coordinates = np.array(cs.allsegs[0][0])
        coordinates_x = np.split(coordinates, 2, axis=1)[0]
        coordinates_y = np.split(coordinates, 2, axis=1)[1]
        if stdexample_category == "explicit_1":
            deviation = (1/6) * np.log(np.maximum(-np.cos(6 * coordinates_x), 0.0000000000000001)) - coordinates_y - 6*i*dt
        elif stdexample_category == "explicit_2":
            deviation = coordinates_x**2 + coordinates_y**2 - 2*(0.4 - i*dt)
        elif stdexample_category == "explicit_3":
            deviation = coordinates_y**2 - 0.25*(np.arctan(np.sqrt(np.maximum(m.exp(-4*i*dt)*np.exp(2*coordinates_x)-1,0))))**2
        elif stdexample_category == "explicit_4":
            deviation = np.cosh(coordinates_y) - 5*(m.exp(-i*dt))*np.cos(coordinates_x)
        mean_squared_deviation = np.median(np.power(deviation, 2))
        #if i in range(0, 2601, 100):
        print(i, mean_squared_deviation)

def compute_errors_kdtree(stdexample_category, cs, cs2, i, dt):
    if stdexample_category == "circle":
        return circle_error(cs,i, dt)
    else:
        coordinates = np.array(cs.allsegs[0][0])
        true_values = np.array(cs2.allsegs[0][0])
        tree = KDTree(true_values)
        errors = tree.query(coordinates)[0]
        std_error = np.mean(errors)
        print(i, std_error)
        return [std_error]
    

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
    if i in range(0,2401,100):
        error = (abs(mean_radius-true_radius)/true_radius) * 100
        #print(i,mean_radius,true_radius, error,std_radius)
        print(i, error)
        return [error]       
