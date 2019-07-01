import math
import sys
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

# Para ejecutarse se deben tener dos archivos en la carpeta data en la raiz del directorio
# con el siguiente formato x_clientes y x_impostores, el script se ejecuta asi:
# python curvaROC.py x 

def FPFN(score_imposto, score_cliente, x):
    fn = []
    fp = []
    for i in x:
        t = score_imposto > i
        fp.append(collections.Counter(t)[True]/len(t))

        t1 = score_cliente < i
        fn.append(1- (collections.Counter(t1)[True]/len(t1)) )

    return np.array(fp), np.array(fn)

def FP_FN(fp, fn, x):
    min = np.abs(fp-(1-fn))
    arg_min = np.argmin(min)
    umbral = x[arg_min]
    fp_min = fp[arg_min]
    fn_min = fn[arg_min]

    print("umbral FN = FP", umbral)

    return fp_min, fn_min

def FN_x(fp, fn , X, x):
    min = np.abs(fn-(1-x))
    arg_min = np.argmin(min)
    umbral = X[arg_min]
    fp_x = fp[arg_min]
    fn_x = fn[arg_min]

    print("umbral FN = x", umbral)
    print("FP(FN = x) = ", fp_x)

    return fp_x, fn_x

def FP_x(fp, fn , X, x):
    min = np.abs(fp-x)
    arg_min = np.argmin(min)
    umbral = X[arg_min]
    fp_x = fp[arg_min]
    fn_x = fn[arg_min]

    print("umbral FP = x", umbral)
    print("FN(FP = x) = ", 1-fn_x)

    return fp_x, fn_x

def plot_ROC(fpa, fna, xA, x, name):
    fp_min, fn_min = FP_FN(fpa, fna, xA)
    fp_x, fn_x = FN_x(fpa, fna, xA, x)
    fp_x2, fn_x2 = FP_x(fpa, fna, xA, x)


    plt.plot(fpa, fna, 'b')
    plt.plot(fp_min, fn_min, 'r^')
    plt.plot(fp_x, fn_x, 'g^')
    plt.plot(fp_x2, fn_x2, 'y^')
    plt.plot([0, 1], [0, 1], 'k--')
    red_patch = mpatches.Patch(color='red', label='FP=FN')
    blue_patch = mpatches.Patch(color='blue', label='Curva ROC')
    green_patch = mpatches.Patch(color='green', label='FN = x')
    yellow_patch = mpatches.Patch(color='yellow', label='FP = x')
    plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch])
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel('FP')
    plt.ylabel('1-FN')
    plt.title(name)
    plt.show()

def D_Prime(p, n):
    d = (p.mean()-n.mean())/math.sqrt(p.var()+n.var())
    print("Deprime = ", d)

def AROC(sC, sI):
    H=0
    for c in sC:
        H += collections.Counter(sI<c)[True] + 0.5*collections.Counter(c==sI)[True]
    
    return H/(len(sC)*len(sI))


# ArgPaser
parser = argparse.ArgumentParser(description='Curva ROC y metricas')
parser.add_argument('-x', '--x', type =float, default=0.5,
                   help='Valor de x para calcular FP(FN=x), FN(FP=x)')
parser.add_argument('-d', '--data',  type =str, default='scoresA',
                   help='nombre de datos, los datos deben tener el formato'+'\n'+ 
                       'nombre_clientes y nombre_impostores')

args = parser.parse_args()

name = args.data
x = args.x

score_cliente = np.loadtxt("data/"+name+"_clientes")[:,1]
score_imposto  = np.loadtxt("data/"+name+"_impostores")[:,1]
xA = np.unique(np.append(score_cliente, score_imposto)) # x min es umbrales

fpa, fna1 = FPFN(score_imposto, score_cliente, xA)
D_Prime(score_cliente, score_imposto)
print('AROC = ',AROC(score_cliente, score_imposto))
plot_ROC(fpa, fna1, xA, x, name)
