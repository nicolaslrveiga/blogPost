#% Your goal of this assignment is implementing your own K-medoids.
#% Please refer to the instructions carefully, and we encourage you to
#% consult with other resources about this algorithm on the web.
#%
#% Input:
#%     pixels: data set. Each row contains one data point. For image
#%     dataset, it contains 3 columns, each column corresponding to Red,
#%     Green, and Blue component.
#%
#%     K: the number of desired clusters. Too high value of K may result in
#%     empty cluster error. Then, you need to reduce it.
#%
#% Output:
#%     class: the class assignment of each data point in pixels. The
#%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
#%     of class should be either 1, 2, 3, 4, or 5. The output should be a
#%     column vector with size(pixels, 1) elements.
#%
#%     centroid: the location of K centroids in your result. With images,
#%     each centroid corresponds to the representative color of each
#%     cluster. The output should be a matrix with K rows and
#%     3 columns. The range of values should be [0, 255].
#%     
#%
#% You may run the following line, then you can see what should be done.
#% For submission, you need to code your own implementation without using
#% the kmeans matlab function directly. That is, you need to comment it out.

#from sklearn.cluster import KMeans

#def my_kmedoids(image_data, K):
#    kmeans = KMeans(n_clusters=K).fit(image_data)
#    label = kmeans.labels_
#    centroid = kmeans.cluster_centers_
#    return label, centroid
import random
import time
import numpy as np
	
def my_kmedoids(pixels, k = 3):
    data_x = pixels[:, 0]
    data_y = pixels[:, 1]
    data_z = pixels[:, 2]
    timer_step_1 = 0
    timer_step_2_3 = 0
    timer_step_4_5 = 0

    start = time.time()
    # Code to avoid selecting repeated initial clusters
    #First it creates a dict storing all different values as keys
    non_repeat = {}
    for pixel in pixels:
        if (pixel[0], pixel[1], pixel[2]) in non_repeat:
            non_repeat[(pixel[0], pixel[1], pixel[2])] += 1
        else:
            non_repeat[(pixel[0], pixel[1], pixel[2])] = 1

    if (k > len(non_repeat.keys())):
        print("k is too large, reducing it to the max number of cluster possible: ", len(non_repeat.keys()))
        k = len(non_repeat.keys())

    #Second we randomly select k number of points from the dict to be the centers
    initial_center_indexes = []
    index = int(random.uniform(0, len(non_repeat.keys())))
    for i in range(0, k):
        while(index in initial_center_indexes):
            index = int(random.uniform(0, len(non_repeat.keys())))
        initial_center_indexes.append(index)

    centers_x = []
    centers_y = []
    centers_z = []
    for i in range(0, k):
        center = list(non_repeat.keys())[initial_center_indexes[i]]
        centers_x.append(center[0])
        centers_y.append(center[1])
        centers_z.append(center[2])
    end = time.time()
    timer_step_1 = end - start

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_z = np.array(data_z)
    centers_x = np.array(centers_x)
    centers_y = np.array(centers_y)
    centers_z = np.array(centers_z)
    pixels_classification = np.zeros(pixels.shape[0])

    warning_counter = 0
    center_difference = 1
    while (center_difference != 0):
        start = time.time()
        dist = np.zeros((len(data_x), len(centers_x)))
        for i in range(0, len(centers_x)):
            dist[:, i] = (np.absolute(data_x - centers_x[i]) + np.absolute(data_y - centers_y[i]) + np.absolute(data_z - centers_z[i]))

        # assign each point to its respective cluster based on the smallest distance to it.
        # {1: [[x, y, z]]}
        clusters = {}
        for i in range(len(centers_x)):
            clusters[i] = []

        for i in range(len(data_x)):
            clusters[np.argmin(dist[i, :])].append([data_x[i], data_y[i], data_z[i]])
            pixels_classification[i] = np.argmin(dist[i, :])
        end = time.time()
        timer_step_2_3 += end - start

        start = time.time()
        # dist now will only store the sum of the distances, we do not need to store each distance
        new_centers_x = np.zeros(k)
        new_centers_y = np.zeros(k)
        new_centers_z = np.zeros(k)
        for i in range(len(clusters)):
            data = np.array(clusters[i])

            # If cluster is not empty
            if (len(data) != 0):
                #Only compute the distance between the values that are unic, we do not need to compute the same value two times.
                non_repeat = {}
                for pixel in data:
                    if (pixel[0], pixel[1], pixel[2]) in non_repeat:
                        non_repeat[(pixel[0], pixel[1], pixel[2])] += 1
                    else:
                        non_repeat[(pixel[0], pixel[1], pixel[2])] = 1
                data = np.array(list(non_repeat.keys()))
                dist = np.zeros((len(data)))

                #Compute the sum of distances from each point to all the points in the cluster and store
                for index, point in enumerate(data):
                    dist[index] = np.sum(np.absolute(data[:, 0] - point[0]) + np.absolute(data[:, 1] - point[1]) + np.absolute(data[:, 2] - point[2]))

                new_center = data[np.argmin(dist)]
                new_centers_x[i] = new_center[0]
                new_centers_y[i] = new_center[1]
                new_centers_z[i] = new_center[2]
            else:
                new_centers_x[i] = centers_x[i]
                new_centers_y[i] = centers_y[i]
                new_centers_z[i] = centers_z[i]
                warning_counter += 1
        end = time.time()
        timer_step_4_5 += end - start

        center_difference_x = centers_x - new_centers_x
        center_difference_y = centers_y - new_centers_y
        center_difference_z = centers_z - new_centers_z

        centers_x = new_centers_x
        centers_y = new_centers_y
        centers_z = new_centers_z

        center_difference = np.sum(np.absolute(center_difference_x)) + np.sum(np.absolute(center_difference_y)) + np.sum(np.absolute(center_difference_z))

    output = []
    for i in range(0, len(centers_x)):
        output.append([centers_x[i], centers_y[i], centers_z[i]])

    if warning_counter:
        print("Warning, cluster did not change ", warning_counter, " times!!!")

    #print("time step 1: ", timer_step_1)
    #print("time step 2 and 3: ", timer_step_2_3)
    #print("time step 4 and 5: ", timer_step_4_5)
    return pixels_classification.astype(np.int), np.array(output).astype(np.int)