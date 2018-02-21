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
    X1= X1.reshape(3,IMAGE_SIZE,IMAGE_SIZE,BATCH_SIZE).transpose([3,1,2,0])
    Y1 = Y1.transpose([1,0])
    X2, Y2 = get_data(BATCH2)
    X2= X2.reshape(3,IMAGE_SIZE,IMAGE_SIZE,BATCH_SIZE).transpose([3,1,2,0])
    Y2 = Y2.transpose([1,0])
    X3, Y3 = get_data(BATCH3)
    X3= X3.reshape(3,IMAGE_SIZE,IMAGE_SIZE,BATCH_SIZE).transpose([3,1,2,0])
    Y3 = Y3.transpose([1,0])
    X4, Y4 = get_data(BATCH4)
    X4= X4.reshape(3,IMAGE_SIZE,IMAGE_SIZE,BATCH_SIZE).transpose([3,1,2,0])
    Y4 = Y4.transpose([1,0])
    X5, Y5 = get_data(BATCH5)
    X5= X5.reshape(3,IMAGE_SIZE,IMAGE_SIZE,BATCH_SIZE).transpose([3,1,2,0])
    Y5 = Y5.transpose([1,0])
    X_test, Y_test = get_data(TEST_BATCH)
    #Collecting data into one array
    batches_array = np.array(X1)
    batches_array = np.concatenate((batches_array, X2))
    batches_array = np.concatenate((batches_array, X3))
    batches_array = np.concatenate((batches_array, X4))
    batches_array = np.concatenate((batches_array, X5))
    labels_array = np.array(Y1)
    labels_array = np.concatenate((labels_array, Y2))
    labels_array = np.concatenate((labels_array, Y3))
    labels_array = np.concatenate((labels_array, Y4))
    labels_array = np.concatenate((labels_array, Y5))
    rotated_images, rotated_labels = rotate_images(batches_array,labels_array)
    flipped_images, flipped_labels = flip_images(batches_array,labels_array)
    batches_array = np.concatenate((batches_array, rotated_images))
    batches_array = np.concatenate((batches_array, flipped_images))
    labels_array = np.concatenate((labels_array, rotated_labels))
    labels_array = np.concatenate((labels_array, flipped_labels))
    #Same shuffling in both arrays
    temp = np.arange(batches_array.shape[0])
    np.random.shuffle(temp)
    batches_array = batches_array[temp]
    labels_array = labels_array[temp]


    X_test = X_test.reshape(3,IMAGE_SIZE,IMAGE_SIZE,BATCH_SIZE).transpose([3,1,2,0])
    X_test = image_convertion(X_test)
    Y_test = Y_test.transpose([1,0])
    # dev_X = X1[:100,:,:,:]
    # dev_Y = Y1[:100,:]
    # rotated_images, rotated_labels = rotate_images(dev_X, dev_Y)
    # flipped_images, flipped_labels = flip_images(dev_X, dev_Y)
    # dev_Y = np.concatenate((dev_Y,rotated_labels), axis=0)
    # dev_Y = np.concatenate((dev_Y,flipped_labels), axis=0)
    # dev_X = np.concatenate((dev_X,rotated_images), axis=0)
    # dev_X = np.concatenate((dev_X,flipped_images), axis=0)
    # dev_X = image_convertion(dev_X)
    _, _, parameters = model(batches_array, labels_array, X_test, Y_test)
    #_, _, parameters = model(dev_X, dev_Y, X_test, Y_test)