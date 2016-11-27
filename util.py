import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plot_states(path, values, corrpath=None):
    """
    Plots the results of a algorithm. 

    :param path: List or array of state ids
    :param values: detailed values returned by the algorithm (forward distribution, ...)
    """
    plt.figure(figsize=(12,4))
    plt.imshow(values.T, interpolation='none', cmap=plt.cm.Greys)
    
    if not (corrpath is None):
        for i in range(len(corrpath)-1):
            plt.arrow(i+0.1, corrpath[i], 0.9, corrpath[i+1] - corrpath[i], length_includes_head=True, head_width=0.25, fc='g', ec='g')
    
    for i in range(len(path)-1):
        plt.arrow(i+0.1, path[i], 0.9, path[i+1] - path[i], length_includes_head=True, head_width=0.25, fc='r', ec='r')


def saveToFile(name, obj):
    fwy = open(name, 'wb')
    pickle.dump(obj,fwy)
    fwy.close()
    
def loadFromFile(name):
    fwy = open(name, 'rb')
    a = pickle.load(fwy)
    fwy.close()
    return a;

