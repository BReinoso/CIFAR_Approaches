#Functions extracted from: https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
#Functions has been lightly modified to adapt it to my needs
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
from Constants import IMAGE_SIZE, BATCH_SIZE


#Funtionc to rotate images and return the corresponding labels
def rotate_images(X_imgs, Y_labels):
    X_rotate = []
    Y_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k=k)
    i = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict={X: img, k: i + 1})
                X_rotate.append(rotated_img)
                Y_rotate.append(Y_labels[i])
            i = i+1
    X_rotate = np.array(X_rotate, dtype=np.uint8)
    Y_rotate = np.array(Y_rotate)
    return X_rotate, Y_rotate

#Function to flip images and return the corresponding labels
def flip_images(X_imgs, Y_labels):
    X_flip = []
    Y_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    i = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
            for i in range(3):
                Y_flip.append(Y_labels[0])
            i = i+1
    X_flip = np.array(X_flip, dtype = np.uint8)
    Y_flip = np.array(Y_flip)
    return X_flip, Y_flip

#Function to change from uint8 and float32
def image_convertion(imgs):
    images_converted = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    converted_img =tf.image.convert_image_dtype(X, dtype = tf.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in imgs:
            images_converted.append(sess.run(converted_img, feed_dict={X: img}))
    images_converted = np.array(images_converted, dtype= np.float32)
    return images_converted