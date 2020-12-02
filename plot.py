import matplotlib.pyplot as plot
from matplotlib import cm
import numpy as np
import os

RESULTS_DIR = "eval/trec_output/"

X = []
Y = []
Z = []


def get_precision(file):
    file_path = f"{RESULTS_DIR}/{file}"
    results_file = open(file_path, "r")
    lines = results_file.readlines()
    for line in lines:
        split = line.split()
        category = split[0]
        domain = split[1]
        if domain == "all" and category == "map":
            precision = float(split[2])
            return precision


def main():
    global X, Y, Z
    for file in os.listdir(RESULTS_DIR):
        file_parts = file.split("_")
        frange = float(file_parts[1])
        threshold = float(file_parts[3].split(".txt")[0])
        precision = get_precision(file)
        if precision is None:
            precision = 0.0

        X.append(threshold)
        Y.append(frange)
        Z.append(precision)

    max_precision = 0.0

    for i in range(len(Z)):
        if Z[i] > max_precision:
            max_precision = Z[i]

    for i in range(len(Z)):
        if Z[i] == max_precision:
            print(f"Max Precision: {max_precision}, Occurs at threshold: {X[i]}, Range: {Y[i]}")

    # Format data for matplotlib
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # Create graph in matplotlib
    figure = plot.figure()
    ax = figure.gca(projection='3d')
    surface = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    figure.colorbar(surface, shrink=0.5, aspect=5)
    plot.show()


if __name__ == "__main__":
    main()
