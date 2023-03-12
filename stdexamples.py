import numpy as np
import math as m

def init_u(n, X_comp, Y_comp, u_comp):
    if (n == "ellipse"):
        return 10 * X_comp**2 + 4*Y_comp**2 -1
    elif (n == "circle"):
        return 4 * X_comp**2 + 4*Y_comp**2 -1
    elif (n == "fattening"):
        return abs(X_comp) - abs(Y_comp)
    elif (n == "explicit_1"):
        return (1/6) * np.log(np.maximum(-np.cos(6 * X_comp), 0.0000000000000001)) - Y_comp
    elif (n == "explicit_2"):
        return X_comp**2 + Y_comp**2 - 2*0.4
    elif (n == "explicit_3"):
        return Y_comp**2 - 0.25*(np.arctan(np.sqrt(np.maximum(np.exp(2*X_comp)-1,0.00000000000000001))))**2
    elif (n == "explicit_4"):
        return (np.cosh(Y_comp) - 5*np.cos(X_comp))
    elif (n == "explicit_5"):
        return np.sin(X_comp)/np.sqrt(np.cos(X_comp)**2 + 1) - Y_comp   
    else:
        return u_comp

def explicit_sol(n, X_comp, Y_comp, i, dt):
    if (n == "explicit_1"):
        return (1/6) * np.log(np.maximum(-np.cos(6 * X_comp), 0.0000000000000001)) - Y_comp - 6*i*dt
    elif (n == "explicit_2"):
        return X_comp**2 + Y_comp**2 - 2*(0.4 - i*dt)
    elif (n == "explicit_3"):
        return Y_comp**2 - 0.25*(np.arctan(np.sqrt(np.maximum(m.exp(-4*i*dt)*np.exp(2*X_comp)-1,0.00000000000000001))))**2
    elif (n == "explicit_4"):
        return np.cosh(Y_comp) - 5*(m.exp(-i*dt))*np.cos(X_comp)
    elif (n == "explicit_5"):
        return np.sin(X_comp)/np.sqrt(np.cos(X_comp)**2 + np.exp(2 * i * dt)) - Y_comp
    elif (n == "circle"):
        return -1
    return -1

def var_intervals(stdexample_category):
    if stdexample_category == "explicit_2":
        return (-1.5,1.5,-1.5,1.5)
    elif stdexample_category == "explicit_4":
        return (-3,3,-4,4)
    elif stdexample_category == "explicit_5":
        return (-m.pi, m.pi, -3, 3)
    elif stdexample_category == "circle":
        return (-1,1,-1,1)
    else:
        return (-3,3,-3,3)

def boundaries(stdexample_category):
    if stdexample_category == "explicit_2":
        return "Dirichlet"
    elif stdexample_category == "explicit_4":
        return "Dirichlet"
    elif stdexample_category == "explicit_5":
        return "periodic"
    elif stdexample_category == "circle":
        return "Dirichlet"
    else:
        return "Dirichlet"

