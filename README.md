# CIFAR_Approaches
Personal work on CIFAR dataset testing some original and article-based ideas.


#Related Artciles:

    1.1- SNAPSHOT ENSEMBLES: https://arxiv.org/pdf/1704.00109.pdf
    1.2- Spatially-sparse convolutional neural networks: https://arxiv.org/pdf/1409.6070.pdf
    1.3- Multi-column Deep Neural Networks for Image Classification: http://people.idsia.ch/~juergen/cvpr2012.pdf
    1.4- Network In Network: https://arxiv.org/pdf/1312.4400.pdf
    1.5- VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION: https://arxiv.org/pdf/1409.1556.pdf
    
#Project Environment:

    2.1- Toolkit CUDA Development 9.0
    2.2- Cudnn 7.0
    2.3- Tensorflow GPU
    2.4- Anaconda Environment
    2.5- GPU GTX 1080 Ti
	2.6- PyCharm Pro with student license

#Input Images Structure

	3.1 - Images data: The extracted data has a shape of (3072,10000) where 10000 are the number of images of the batch (a total of 5 batches in different documents) and 3072 is the value of the different pixels for each image. 
		The first step to prepare the data is reshape it to (3, 32, 32, 10000) to collect each image in with its channels.
		The second step is the transpose function to have the shape (10000, 32, 32, 3) to be able to access to the images as X[0] for the first image.
		
	3.2- Label data: Label data is given with (10, 10000) shape. The only step is the transpose of these values to have shape as (10000, 10)

#Python Files

	4.1- Augmentation.py: Implementation of two different methods of augmentation. The first one is the rotation of the original image with a 90 grade angle three times.
		The second one is the flip function to obtain there mirrored images.
		The total amount of images obtained from the application of these two process is 6 per image.
	
	4.2- Cifar20.py: The main file where the execution is deployed. It is called CIFAR10 because is to start tests with this dataset and we should be able to call every Model example.
	4.3- Constants.py: A global python file to store all configuration constants.
	4.5- ExtractData.py: Python fail to define the functions to extract the raw data from CIFAR dataset
	4.6- ModelN.py: Python files where N is substituted by  a number representing the Model defined. Here are some approaches of models to try with the dataset.
	
#Models

	4.1- Very simple mode based in a programming assignment from Coursera Convolutional Neural Networks (https://www.coursera.org/learn/convolutional-neural-networks/home/info):
		Input Layer: Image of size 32X32X3
		Layer 1: Convolutional layer with 8 filters 4X4 and stride=1 and padding= SAME. Activation function= RELU
		Layer 2: Maxpool Layer with widow of 8X8, stride of 8 and padding= SAME
		Layer 3: Convolutional layer with 16 filters 2X2 and stride=1 and padding= SAME. Activation function= RELU
		Layer 4: Maxpool Layer with widow of 4X4, stride of 4 and padding= SAME
		Ouput Layer: Fully connected layer with 10 neurons and a softmax activation.
	4.2- Extension model of the first one. 
		Layer 1: Convolutional layer with 128 filters 4X4 and stride=1 and padding= SAME. Activation function= RELU
		Layer 2: Maxpool Layer with widow of 8X8, stride of 8 and padding= SAME
		Layer 3: Convolutional layer with 256 filters 2X2 and stride=1 and padding= SAME. Activation function= RELU
		Layer 4: Maxpool Layer with widow of 4X4, stride of 2 and padding= SAME
		Layer 5: Flatten Convolutional layer with 200 filters 2X2 and stride=1 and padding= SAME. Activation function= RELU. This layer convert the map features into a vector.
		Layer 6: Flatten step to delete useless dimensions after the Flatten Convolutional Layer
		Layer 7: Fully connected layer with 200 neurons and RELU activation function. And dropout (Probability has to be set manually)
		Layer 8: Fully connected layer with 100 neurons and RELU activation function. And dropout (Probability has to be set manually)
		Layer 9: Fully connected layer with 50 neurons and RELU activation function. And dropout (Probability has to be set manually)
		Ouput Layer: Fully connected layer with 10 neurons and a softmax activation.