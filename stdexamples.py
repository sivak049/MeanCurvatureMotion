def init_u(n, X_comp, Y_comp, u_comp):
    if (n == "ellipse"):
        return 10 * X_comp**2 + 4*Y_comp**2 -1
    elif (n == "circle"):
        return 4 * X_comp**2 + 4*Y_comp**2 -1
    elif (n == "fattening"):
        return abs(X_comp) - abs(Y_comp)
    else:
        return u_comp

