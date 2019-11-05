# K means algorithm
# By Nicolas Veiga

import numpy as np
import matplotlib.pyplot as plt

# First normal distribution
first_distribution = np.random.normal(0, 1, (1000, 2))
# Second normal distribution
second_distribution = np.random.normal(4, 1, (1000, 2))
# Stacking this two distribution together
data = np.vstack([first_distribution, second_distribution])

plt.scatter(data[:, 0], data[:, 1]);
plt.grid();
plt.title("Data from two normal distributions");
plt.xlabel("x axis");
plt.ylabel("y axis");
plt.show()

c = np.zeros([2,2])
c[:,0] = np.random.uniform(np.min(data[:,0]), np.max(data[:,0]), 2)
c[:,1] = np.random.uniform(np.min(data[:,1]), np.max(data[:,1]), 2)

plt.scatter(data[:, 0], data[:, 1], alpha=0.3);
plt.scatter(c[:, 0], c[:, 1], color="red");
plt.grid();
plt.title("Data and random cluster centers");
plt.xlabel("x axis");
plt.ylabel("y axis");
plt.show()

difference_between_new_and_old_cluster_center = 1.0
while(difference_between_new_and_old_cluster_center != 0.0):
    # The distance matrix will have the shape [number of samples, number of clusters]
    dist = np.zeros((len(data[:, 0]), len(c[:, 0])))

    for i in range(0, len(c[:,0])):
        dist[:, i] = ((data[:, 0] - c[i, 0]) ** 2 + (data[:, 1] - c[i, 1]) ** 2) ** (1/2)

    # assign each point to its respective cluster based on the smallest distance to it.
    # {1: [[x1, y1], [x2, y2], ...]}
    clusters = {}
    pixels_classification = np.zeros(len(data[:, 0]))
    for i in range(len(c[:, 0])):
        clusters[i] = []

    for i in range(len(data[:, 0])):
        clusters[np.argmin(dist[i, :])].append([data[i, 0], data[i, 1]])
        pixels_classification[i] = np.argmin(dist[i, :])

    # Computing the mean for each cluster
    new_c = np.zeros([2, 2])
    for key in clusters.keys():
        new_c[key, :] = np.mean(clusters[key], 0)

    # Computing loss function
    objective_function = 0.0
    for i in range(len(data[:, 0])):
        objective_function = objective_function + np.min(dist[i, :])
    print("Objective function: " + str(objective_function))
    difference_between_new_and_old_cluster_center = np.sum((c - new_c) ** 2)

    c = new_c

    plt.scatter(data[np.where(pixels_classification==0)][:, 0], data[np.where(pixels_classification==0)][:, 1], alpha=0.3);
    plt.scatter(data[np.where(pixels_classification==1)][:, 0], data[np.where(pixels_classification==1)][:, 1], alpha=0.3);
    plt.scatter(new_c[:, 0], new_c[:, 1], color="red");
    plt.grid();
    plt.title("Data and random cluster centers");
    plt.xlabel("x axis");
    plt.ylabel("y axis");
    plt.show()