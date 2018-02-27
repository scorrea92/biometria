import sys
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print('Argument List:', str(sys.argv))

try:
    # Para Python2
    x = raw_input("Ingrese X, por defecto (=0.5): ") or 0.5
except :
    # Para Python3
    x = input("Ingrese X, por defecto (=0.5): ") or 0.5

x = float(x)

name = sys.argv[1]
score_cliente_A = np.loadtxt("practicas/"+name+"_clientes")[:,1]
score_imposto_A = np.loadtxt("practicas/"+name+"_impostores")[:,1]
xA = np.unique(np.append(score_cliente_A, score_imposto_A))

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

def D_Prime(fp, fn):
    fn = 1-fn
    d = (fp.mean()-fn.mean())/np.sqrt(fp.std()+fn.std())
    print("Deprime = ", d)

fpa, fna = FPFN(score_imposto_A, score_cliente_A, xA)
D_Prime(fpa, fna)
plot_ROC(fpa, fna, xA, x, name)
