#% Your goal of this assignment is implementing your own K-means.
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

#def my_kmeans(image_data, K):
#    kmeans = KMeans(n_clusters=K).fit(image_data)
#    label = kmeans.labels_
#    centroid = kmeans.cluster_centers_
#    return label, centroid

import random
import time
import numpy as np

# Kmeans
def my_kmeans(pixels, k):
    data_x = pixels[:, 0]
    data_y = pixels[:, 1]
    data_z = pixels[:, 2]

    timer_step_1 = 0
    timer_step_2_3 = 0
    timer_step_4_5 = 0
    
    number_of_centers = k
    
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
        number_of_centers = len(non_repeat.keys())
    
    #Second we randomly select k number of points from the dict to be the centers
    #initial_center_indexes = []
    #index = int(random.uniform(0, len(non_repeat.keys())))
    #for i in range(0, k):
    #    while(index in initial_center_indexes):
    #        index = int(random.uniform(0, len(non_repeat.keys())))
    #    initial_center_indexes.append(index)    
    
    #Random initialize the centers, we will have 3 centers for now
    centers = []
    centers_x = []
    centers_y = []
    centers_z = []
    for i in range(0, number_of_centers):
        center_x = int(random.uniform(0, 255))
        center_y = int(random.uniform(0, 255))
        center_z = int(random.uniform(0, 255))
        center = (center_x, center_y, center_z)

        while(center in centers):
            center_x = int(random.uniform(0, 255))
            center_y = int(random.uniform(0, 255))
            center_z = int(random.uniform(0, 255))
            center = (center_x, center_y, center_z)

        centers.append(center)
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

    # Compute distance from center using euclidean distance
    center_difference = 1.0
    empty_clusters = 0
    while (center_difference != 0.0):
        
        start = time.time()
        dist = np.zeros((len(data_x), len(centers_x)))
        for i in range(0, len(centers_x)):
            dist[:, i] = ((data_x - centers_x[i]) ** 2 + (data_y - centers_y[i]) ** 2 + (data_z - centers_z[i]) ** 2) ** (1/2)
        
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
        # Compute new center, the new center is the average of the points in each cluster
        new_c_x = []
        new_c_y = []
        new_c_z = []
        for key in clusters.keys():
            data = clusters[key]
            if (len(data) != 0):
                mean_data = np.mean(data, 0).astype(np.int)
                new_c_x.append(mean_data[0])
                new_c_y.append(mean_data[1])
                new_c_z.append(mean_data[2])
            else:
                new_c_x.append(centers_x[key])
                new_c_y.append(centers_y[key])
                new_c_z.append(centers_z[key])
                empty_clusters += 1
        end = time.time()
        timer_step_4_5 += end - start

        # Compute objective function, the goal is to obtain the smallest possible objetive function,
        # that is the sum of the squared difference between sample and its center
        obj_func = 0
        for key in clusters.keys():
            data = clusters[key]
            if (len(data) != 0):
                obj_func = obj_func + (np.sum(np.sum(((np.array(data) - [new_c_x[key], new_c_y[key], new_c_z[key]]) ** 2), 0)))
        #print("Object function: ", obj_func)

        new_c_x = np.array(new_c_x)
        new_c_y = np.array(new_c_y)
        new_c_z = np.array(new_c_z)

        center_difference_x = centers_x - new_c_x
        center_difference_y = centers_y - new_c_y
        center_difference_z = centers_z - new_c_z

        #print(center_difference_x)
        #print(center_difference_y)
        #print(center_difference_z)

        centers_x = new_c_x
        centers_y = new_c_y
        centers_z = new_c_z

        center_difference = np.sum(center_difference_x ** 2)

    output = []
    for i in range(0, len(centers_x)):
        output.append([centers_x[i], centers_y[i], centers_z[i]])
    
    if empty_clusters:
        print("Warning, number of empty clusters: ", empty_clusters)
    
    #print("time step 1: ", timer_step_1)
    #print("time step 2 and 3: ", timer_step_2_3)
    #print("time step 4 and 5: ", timer_step_4_5)
    
    return pixels_classification.astype(np.int), np.array(output)