import matplotlib.pyplot as plot
import numpy as np
import os

RESULTS_DIR = "eval/trec_sup_output/"

X = []
Y = []


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
    global X, Y
    for file in os.listdir(RESULTS_DIR):
        file_parts = file.split("_")
        weight = float(file_parts[1].split(".txt")[0])
        precision = get_precision(file)
        if precision is None:
            precision = 0.0

        X.append(weight)
        Y.append(precision)

    max_precision = 0.0

    for i in range(len(Y)):
        if Y[i] > max_precision:
            max_precision = Y[i]

    for i in range(len(Y)):
        if Y[i] == max_precision:
            print(f"Max Precision: {max_precision}, Occurs at weight: {X[i]}")

    # Format data for matplotlib
    X = np.array(X)
    Y = np.array(Y)

    # Create graph in matplotlib
    figure, ax = plot.subplots()
    ax.set(ylim=[0.35, 0.39])
    ax.bar(X, Y, alpha=0.4, color='b')
    plot.xticks(np.arange(1, 21, 1))
    plot.xlabel("Weight")
    plot.ylabel("Mean Average Precision")
    plot.show()


if __name__ == "__main__":
    main()
