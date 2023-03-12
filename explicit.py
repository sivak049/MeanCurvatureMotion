import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math as m
import stdexamples
import utils
import csv
import pandas

plt.ion()

def explicit_analysis(mode, stencil_size, stdexample_category, mesh_size):
    boundary = stdexamples.boundaries(stdexample_category)
    x_start, x_end, y_start, y_end = stdexamples.var_intervals(stdexample_category)
    mesh_size = 100
    #h = 1/(mesh_size-1)
    h = (min(x_end-x_start, y_end - y_start))/((mesh_size-1)*2)
    h2 = h**2
    dt = 0.5*h2
    mode = mode # adam; improv
    stencil_size = stencil_size # 1: 8 points, 2: 12 points, 3: 16 points
    boundary = boundary # "Neumann" or "Dirichlet" boundary condition
    plot_style = "contour" # "contour" or "surface"

    X_comp,Y_comp = utils.init_grid(mesh_size, mode, stencil_size, x_start, x_end, y_start, y_end)
    u_comp = 10 * X_comp**2 + 4*Y_comp**2 -1
    u = u_comp

    stdexample_category = stdexample_category # "ellipse" for ellipse; 
            # "circle" for Circle (with accuracy measuring)
            # "fattening" for fattening example
            # "explicit_1" to "explicit_6" for error computations with functions we know explicit solutions for
            # "default" for default

    if stdexample_category in ["ellipse", "circle", "fattening","explicit_1","explicit_2","explicit_3","explicit_4","explicit_5"]:
        u_comp = stdexamples.init_u(stdexample_category,X_comp,Y_comp, u_comp)
    fig = plt.figure()
    ax = utils.init_ax(plot_style, fig)
    #fig2 = plt.figure()
    #ax2 = fig2.gca()
    print(mode, stencil_size, boundary, stdexample_category)
    if stdexample_category == "circle":
        print("i","error")
        #print("i","mean_radius","true_radius", "error", "std_radius")
    name = mode+"_"+str(stencil_size)+"_"+boundary+"_"+stdexample_category+"_"+str(mesh_size)
    error = [name]    
    for i in range(2401):
        s = stencil_size
        u_comp[s:-s,s:-s] = utils.mcmPde(u_comp, mode, stencil_size, boundary, dt, h2)  
        u = stdexamples.explicit_sol(stdexample_category, X_comp, Y_comp, i, dt)
        if i % 100 == 0:
            #if i in [0, 300, 750, 1050, 1450, 1850, 2350]:
            #cs2 = ax.contour(X_comp[s:-s,s:-s],Y_comp[s:-s,s:-s],u[s:-s,s:-s], [0], linestyles='dashed')
            if stdexample_category in ["explicit_1","explicit_2","explicit_3","explicit_4","explicit_5"]:
                cs2 = utils.plot_explicit(X_comp,Y_comp, u, stdexample_category, ax, i, dt, stencil_size)
            cs = utils.plots(X_comp, Y_comp, u_comp, stdexample_category, plot_style, ax, stencil_size, fig)
            if stdexample_category == "circle":
                cs2 = cs
            if stdexample_category in ["circle","explicit_1","explicit_2","explicit_3","explicit_4","explicit_5"]:
                #utils.compute_errors(stdexample_category, cs, i, dt)
                error += utils.compute_errors_kdtree(stdexample_category, cs, cs2, i, dt)
            utils.axis_clear(ax)
                #ax2.axis('equal')
                #fig2.canvas.draw()
                #fig2.canvas.flush_events()
    plt.close(fig)
    return error

error_file = open('explicit_error.csv',newline='',mode='a')
writer = csv.writer(error_file, dialect='excel', delimiter=',', quotechar='|',quoting=csv.QUOTE_MINIMAL)
list_of_time_steps = []
for i in range(25):
    list_of_time_steps += [100*i]
#writer.writerow(['name']+list_of_time_steps)
for i in ["explicit_2","explicit_4","explicit_5","circle"]:
    for j in ["adam", "improv"]:
        for k in [1,2,3]:
            writer.writerow(explicit_analysis(j,k,i,100))







