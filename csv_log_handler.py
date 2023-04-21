import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})"""
styles = ['^-.r','o-g','d-.b','s-r','pm-.','xc-']
problem_list = ["Dirichlet_explicit_2","Dirichlet_explicit_4","periodic_explicit_5","Dirichlet_circle"]

D = pd.read_csv('explicit_error_log.csv',header = 0,index_col=0)
index = np.arange(0,2500,100)
ind = list(map(str,index))
D['mean_error'] = D[ind].mean(axis=1)

lbl = ['oberman','improv']
for j in range(len(problem_list)):
    plt.figure()
    ax = plt.subplot(111)
    for i in range(2):
        x = D['dx'].iloc[[6*j + 3*i,6*j + 3*i + 1, 6*j + 3*i + 2]]
        y = D['mean_error'].iloc[[6*j + 3*i,6*j + 3*i + 1, 6*j + 3*i + 2]]
        #d = D[6*j +i : 6*j + i + 1].values.reshape(26,)
        ax.loglog(x,y,styles[i],label=lbl[i])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    plt.xlabel(r'timestep',fontsize=18)
    plt.ylabel('Error',fontsize=18)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #          fancybox=True, shadow=True, ncol=2)
    plt.legend(loc='upper left',fontsize=16)
    name = problem_list[j] + "_log.pdf"
    plt.savefig(name)

plt.show()

