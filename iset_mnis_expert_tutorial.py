# -*- coding: utf-8 -*-
# http://www.tensorflow.org/tutorials/mnist/beginners/index.md

# Import mnist database and adapt it the tf way
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import os
from PIL import Image
workDir = '/Users/nupic/code/tfnotebook'
os.chdir(workDir)

import iset2tf  # Created to read images instead of the original db format
import numpy
from PIL import Image
import os
join = os.path.join

# Read first the original data in png format to check that the reading and
# converting system works fine and yields exactly the same results as above.
# train_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/training'
# test_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/testing'


# From now on we can change the images to the iset generated ones, we will have
# to make changes to the parameters in the script to optmize results
# EYE: To use the 72x88 images with the b&w cone voltage information
# train_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/training'
# test_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/testing'

# SENSOR: To use the images with the b&w sensor voltage information for scien
fov = [0.8];
sceneLights = [10, 50, 100];
pixelSizes = [1.1e-06, 1.25e-06, 1.4e-06];

f = fov[0]
s = sceneLights[0]
p = pixelSizes[0]

train_imagePath = str('/Users/nupic/code/WLletterClass/data/isetMnist' + str(f) +
                      '/train/Light_' + str(s) + '_pSize_' + str(p))
test_imagePath = str('/Users/nupic/code/WLletterClass/data/isetMnist' + str(f) +
                      '/test/Light_' + str(s) + '_pSize_' + str(p))

# Added several options when reading the images, mostly:
# Binarize: We can input binary images to the classifier now
# Randomize: Google inputs images randomized so it is another possibility now.
#            It is possible to input sequences as well, did it for nupic, not yet
#            tested in tensorflow
isetmnist = iset2tf.read_iset_data_sets(train_imagePath, test_imagePath,
                                       one_hot=True, tf_or_nupic='tf',
                                       binarize=False, randomize = True)



# Info about our data, how many categories and image size? If not square we will
# have to indicate per every different image size.
numCats = isetmnist.train.labels[0].shape[0]
imSize = isetmnist.train.images[0,:].shape[0]
im = Image.open(join(train_imagePath, '0', '000000.png'))
if not imSize == im.size[0]*im.size[1]:
    print 'ERROR: image size in numpy and .png is not the same'
else:
    imH = im.size[1]
    imW = im.size[0]



# Now first make images square just in case
# I have an error 
# InvalidArgumentError: Input has 1267200 values, which isn't divisible by 3136
# First of all I am going to convert the images to squares, just in case

# train_sqImageList = iset2tf.createImageList(train_imagePath, '.png', 'tf')
# train_outdir = '/Users/nupic/code/tfnotebook/sqMnist/training'
# test_sqImageList = iset2tf.createImageList(test_imagePath, '.png', 'tf')
# test_outdir = '/Users/nupic/code/tfnotebook/sqMnist/testing'
#
# for nnc in train_sqImageList:
#     iset2tf.squareImage(nnc[0], train_outdir, nnc[2], saveFile=True)
# for nnc in test_sqImageList:
#     iset2tf.squareImage(nnc[0], test_outdir, nnc[2], saveFile=True)
#
# # Read the files again
# isetmnist = iset2tf.read_iset_data_sets(train_outdir, test_outdir,
#                                        one_hot=True, tf_or_nupic='tf')
# # ImgSq.save(newFilename, 'PNG')
# imSize = isetmnist.train.images[0].shape[0]
# imH = 88
# imW = 88
# numCats = isetmnist.train.labels[0].shape[0]
#
# # Just for comparison purposes, here we have the images from the original mnist
# origmnist = iset2tf.read_data_sets('myMNIST/', one_hot=True)

# Start TensorFlow InteractiveSession
# Tensorflow relies on a highly efficient C++ backend to do its computation. 
# The connection to this backend is called a session. The common usage for 
# TensorFlow programs is to first create a graph and then launch it in a session.
# Here we instead use the convenience InteractiveSession class, 
# which makes TensorFlow more flexible about how you structure your code. 
# It allows you to interleave operations which build a computation graph 
# with ones that run the graph. This is particularly convenient when working 
# in interactive contexts like iPython. If you are not using an InteractiveSession, 
# then you should build the entire computation graph before starting a session 
# and launching the graph.
import tensorflow as tf
sess = tf.InteractiveSession()

# Computation Graph
# To do efficient numerical computing in Python, we typically use libraries like 
# NumPy that do expensive operations such as matrix multiplication outside Python,
#  using highly efficient code implemented in another language. Unfortunately, 
#  there can still be a lot of overhead from switching back to Python every 
#  operation. This overhead is especially bad if you want to run computations 
#  on GPUs or in a distributed manner, where there can be a high cost to 
#  transferring data.

# TensorFlow also does its heavy lifting outside Python, but it takes things a 
# step further to avoid this overhead. Instead of running a single expensive 
# operation independently from Python, TensorFlow lets us describe a graph of 
# interacting operations that run entirely outside Python. This approach is 
# similar to that used in Theano or Torch.

# The role of the Python code is therefore to build this external computation 
# graph, and to dictate which parts of the computation graph should be run. 
# See the Computation Graph section of Basic Usage for more detail.

# BUILD A SOFTMAX REGRESSION MODEL
# In this section we will build a softmax regression model with a single linear 
# layer. In the next section, we will extend this to the case of softmax 
# regression with a multilayer convolutional network.

# -- Placeholders: 
# We start building the computation graph by creating nodes for 
#    the input images and target output classes.
x = tf.placeholder("float", shape=[None, imSize])
y_ = tf.placeholder("float", shape=[None, numCats])

# Here x and y_ aren't specific values. Rather, they are each a placeholder -- a 
# value that we'll input when we ask TensorFlow to run a computation.
# The input images x will consist of a 2d tensor of floating point numbers. 
# Here we assign it a shape of [None, 784], where 784 is the dimensionality of a 
# single flattened MNIST image (28x28), and None indicates that the first 
# dimension, corresponding to the batch size, can be of any size. 
# For isetMNIST we are using 72x88 right now. 
# The target output classes y_ will also consist of a 2d tensor, 
# where each row is a one-hot 10-dimensional vector indicating which digit 
# class the corresponding MNIST image belongs to.
# The shape argument to placeholder is optional, but it allows TensorFlow to 
# automatically catch bugs stemming from inconsistent tensor shapes.
# 
# -- Variables
# We now define the weights W and biases b for our model. We could imagine 
# treating these like additional inputs, but TensorFlow has an even better 
# way to handle them: Variable. A Variable is a value that lives in TensorFlow's 
# computation graph. It can be used and even modified by the computation. 
# In machine learning applications, one generally has the model paramaters be 
# Variables.
W = tf.Variable(tf.zeros([imSize,numCats]))
b = tf.Variable(tf.zeros([numCats]))

# We pass the initial value for each parameter in the call to tf.Variable. 
# In this case, we initialize both W and b as tensors full of zeros. 
# W is a 784x10 matrix (because we have 784 input features and 10 outputs) 
# and b is a 10-dimensional vector (because we have 10 classes).
# For isetMNIST will be 6336 * 10
# Before Variables can be used within a session, they must be initialized 
# using that session. This step takes the initial values (in this case tensors 
#     full of zeros) that have already been specified, and assigns them to each 
# Variable. This can be done for all Variables at once.
sess.run(tf.initialize_all_variables())

# Predicted Class and Cost Function

# We can now implement our regression model. It only takes one line! 
# We multiply the vectorized input images x by the weight matrix W, 
# add the bias b, and compute the softmax probabilities that are assigned to 
# each class.

y = tf.nn.softmax(tf.matmul(x,W) + b)
# The cost function to be minimized during training can be specified just 
# as easily. Our cost function will be the cross-entropy between the target 
# and the model's prediction.

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Note that tf.reduce_sum sums across all images in the minibatch, 
# as well as all classes. We are computing the cross entropy for the entire 
# minibatch.

# TRAIN THE MODEL
# Now that we have defined our model and training cost function, 
# it is straightforward to train using TensorFlow. Because TensorFlow knows 
# the entire computation graph, it can use automatic differentiation to find the 
# gradients of the cost with respect to each of the variables. 
# TensorFlow has a variety of builtin optimization algorithms. 
# For this example, we will use steepest gradient descent, with a step length 
# of 0.01, to descend the cross entropy.

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# What TensorFlow actually did in that single line was to add new operations to 
# the computation graph. These operations included ones to compute gradients, 
# compute parameter update steps, and apply update steps to the parameters.

# The returned operation train_step, when run, will apply the gradient descent 
# updates to the parameters. Training the model can therefore be accomplished 
# by repeatedly running train_step.

for i in range(10000):
  batch = isetmnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Each training iteration we load 50 training examples. We then run the 
# train_step operation, using feed_dict to replace the placeholder 
# tensors x and y_ with the training examples. Note that you can replace any 
# tensor in your computation graph using feed_dict -- it's not restricted to 
# just placeholders.

# EVALUATE THE MODEL
# How well did our model do?
# First we'll figure out where we predicted the correct label. tf.argmax is an 
# extremely useful function which gives you the index of the highest entry in a 
# tensor along some axis. For example, tf.argmax(y,1) is the label our model 
# thinks is most likely for each input, while tf.argmax(y_,1) is the true label. 
# We can use tf.equal to check if our prediction matches the truth.

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# That gives us a list of booleans. To determine what fraction are correct, 
# we cast to floating point numbers and then take the mean. 
# For example, [True, False, True, True] would become [1,0,1,1] which would 
# become 0.75.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Finally, we can evaluate our accuracy on the test data. This should be about 
# 91% correct.

print accuracy.eval(feed_dict={x: isetmnist.test.images, y_: isetmnist.test.labels})















# BUILD A MULTILAYER CONVOLUTIONAL NETWORK
# Getting 91% accuracy on MNIST is bad. It's almost embarrassingly bad. 
# In this section, we'll fix that, jumping from a very simple model to something 
# moderatly sophisticated: a small convolutional neural network. 
# This will get us to around 99.2% accuracy -- not state of the art, but 
# respectable.

# -- Weight Initialization
# To create this model, we're going to need to create a lot of weights and biases.
# One should generally initialize weights with a small amount of noise for 
# symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons, 
# it is also good practice to initialize them with a slightly positive initial 
# bias to avoid "dead neurons." Instead of doing this repeatedly while we build 
# the model, let's create two handy functions to do it for us.

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# -- Convolution and Pooling
# TensorFlow also gives us a lot of flexibility in convolution and pooling 
# operations. How do we handle the boundaries? What is our stride size? 
# In this example, we're always going to choose the vanilla version. 
# Our convolutions uses a stride of one and are zero padded so that the output 
# is the same size as the input. Our pooling is plain old max pooling over 2x2 
# blocks. To keep our code cleaner, let's also abstract those operations into 
# functions.

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# -- First Convolutional Layer
# We can now implement our first layer. It will consist of convolution, 
# followed by max pooling. The convolutional will compute 32 features for 
# each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. 
# The first two dimensions are the patch size, the next is the number of input 
# channels, and the last is the number of output channels. We will also have a 
# bias vector with a component for each output channel.

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 4d tensor, with the second and 
# third dimensions corresponding to image width and height, and the final 
# dimension corresponding to the number of color channels.

x_image = tf.reshape(x, [-1,imW,imH,1])

# We then convolve x_image with the weight tensor, add the bias, apply the ReLU 
# function, and finally max pool.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# Second Convolutional Layer
# In order to build a deep network, we stack several layers of this type. 
# The second layer will have 64 features for each 5x5 patch.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# -- Densely Connected Layer
# Now that the image size has been reduced to 7x7, we add a fully-connected layer
#  with 1024 neurons to allow processing on the entire image. We reshape the 
#  tensor from the pooling layer into a batch of vectors, multiply by a weight 
#  matrix, add a bias, and apply a ReLU.

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# ---- Dropout
# To reduce overfitting, we will apply dropout before the readout layer. 
# We create a placeholder for the probability that a neuron's output is kept 
# during dropout. This allows us to turn dropout on during training, and turn 
# it off during testing. TensorFlow's tf.nn.dropout op automatically handles 
# scaling neuron outputs in addition to masking them, so dropout just works 
# without any additional scaling.

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# -- Readout Layer
# Finally, we add a softmax layer, just like for the one layer softmax 
# regression above.
W_fc2 = weight_variable([1024, numCats])
b_fc2 = bias_variable([numCats])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



# -- Train and Evaluate the Model
# How well does this model do? To train and evaluate it we will use code that 
# is nearly identical to that for the simple one layer SoftMax network above. 
# The differences are that: we will replace the steepest gradient descent 
# optimizer with the more sophisticated ADAM optimizer; we will include the 
# additional parameter keep_prob in feed_dict to control the dropout rate; 
# and we will add logging to every 100th iteration in the training process.

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = isetmnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: isetmnist.test.images, y_: iseetmnist.test.labels, keep_prob: 1.0})


# The final test set accuracy after running this code should be approximately 
# 99.2%.
# We have learned how to quickly and easily build, train, and evaluate a 
# fairly sophisticated deep learning model using TensorFlow.








