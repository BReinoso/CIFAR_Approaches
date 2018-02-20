import pickle
import numpy as np
from Constants import *
from Augmentation import *
from ExtractData import *
from Model1 import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

# Main execution
if __name__ == "__main__":
    # Extracting images and labels from the files
    X1, Y1 = get_data(BATCH1)
    X2, Y2 = get_data(BATCH2)
    X3, Y3 = get_data(BATCH3)
    X4, Y4 = get_data(BATCH4)
    X5, Y5 = get_data(BATCH5)
    X_test, Y_test = get_data(TEST_BATCH)
    #Collecting data into arrays
    batches_array = []
    batches_array.append(X1)
    batches_array.append(X2)
    batches_array.append(X3)
    batches_array.append(X4)
    batches_array.append(X5)
    labels_array = []
    labels_array.append(Y1)
    labels_array.append(Y2)
    labels_array.append(Y3)
    labels_array.append(Y4)
    labels_array.append(Y5)
    #X1= X1.reshape(3,IMAGE_SIZE,IMAGE_SIZE,BATCH_SIZE).transpose([3,1,2,0])
    #print(X1.shape)
    #temp = np.zeros([1,32,32,3])
    #temp[0]=X1[50,:,:,:]
    #img,_ = rotate_images(temp, Y1[0])
    #img = image_convertion(img)
    #plt.imshow(img[0])
    #plt.savefig(CIFART_IMAGES+"FloatImage.png")
    # Preparing the data to be treatable and performing data augmentation
    for i in range(len(batches_array)):
        batches_array[i]= batches_array[i].reshape(3,IMAGE_SIZE,IMAGE_SIZE,BATCH_SIZE).transpose([3,1,2,0])
        labels_array[i]= labels_array[i].transpose([1,0])
        rotated_images, rotated_labels = rotate_images(batches_array[i],labels_array[i])
        flipped_images,flipped_labels = flip_images(batches_array[i],labels_array[i])
        batches_array[i] = np.concatenate((batches_array[i],rotated_images), axis=0)
        labels_array[i] = np.concatenate((labels_array[i],rotated_labels), axis=0)
        batches_array[i] = np.concatenate((batches_array[i], rotated_images),axis=0)
        labels_array[i] = np.concatenate((labels_array[i], rotated_labels),axis=0)
        batches_array[i] = image_convertion(batches_array[i])
     #Extracting test set
    X_test = X_test.reshape(3,IMAGE_SIZE,IMAGE_SIZE,BATCH_SIZE).transpose([3,1,2,0])
    Y_test = Y_test.transpose([1,0])
    _, _, parameters = model(batches_array, labels_array, X_test, Y_test)