#% This is a simple example to help you test your implementation using
#% an image. Please feel free to use the attached images, or your own
#% images.
#%
#% An example of running this script is
#%   homework1('beach.bmp', 4);
#%
#% You are not supposed to edit this file. Your job is implementing k-means
#% and k-medoids in the other files, mykmeans.m and mykmedoids.m. Have fun!


import matplotlib.pyplot as plt
import numpy as np

from my_sol_kmeans import my_kmeans
from my_sol_kmedoids import my_kmedoids

def score(image_name, K):
    image = plt.imread(image_name)
    image = image * 255
    image = image.astype('int32') 
    rows = image.shape[0]
    cols = image.shape[1]
    pixels = np.zeros((rows*cols,3))

    for i in range(rows):
        for j in range(cols):
            pixels[j*rows+i,0] = image[i,j,0]
            pixels[j*rows+i,1] = image[i,j,1]
            pixels[j*rows+i,2] = image[i,j,2]
    
    class1, centroid1 = my_kmeans(pixels, K)
    class2, centroid2 = my_kmedoids(pixels, K)
    
    converted_image1 = np.zeros((rows, cols, 3))
    converted_image2 = np.zeros((rows, cols, 3))
    
    for i in range(rows):
        for j in range(cols):
            converted_image1[i,j,:] = centroid1[class1[j*rows+i],:]
            converted_image2[i,j,:] = centroid2[class2[j*rows+i],:]
            
    converted_image1 /= 255
    converted_image2 /= 255
    
    plt.subplot(1,3,1)
    plt.title('Origin')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.title('K-means')
    plt.imshow(converted_image1)
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.title('K-medoids')
    plt.imshow(converted_image2)
    plt.axis('off')
    
    plt.show()
    return None
    
score('image.png',2)