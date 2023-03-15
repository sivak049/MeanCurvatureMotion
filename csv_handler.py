import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})"""
styles = ['^-.r','o-g','d-.b','s-r','pm-.','xc-']
problem_list = ["Dirichlet_explicit_2_100","Dirichlet_explicit_4_100","periodic_explicit_5_100","Dirichlet_circle_100"]

D = pd.read_csv('explicit_error.csv',header = 0,index_col=0)
index = np.arange(0,2500,100)


for j in range(len(problem_list)):
    plt.figure()
    ax = plt.subplot(111)
    for i in range(6):
        d = D[6*j +i : 6*j + i + 1].values.reshape(25,)
        ax.semilogy(index,d,styles[i],label=D.index[6*j + i], markevery = 4)
        #ax.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    plt.xlabel(r'timestep',fontsize=18)
    plt.ylabel('Error',fontsize=18)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)
    #plt.legend(loc='upper right',fontsize=16)
    name = problem_list[j] + ".pdf"
    plt.savefig(name)

plt.show()



