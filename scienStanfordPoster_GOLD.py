# -*- coding: utf-8 -*-
# http://www.tensorflow.org/tutorials/mnist/pros/index.md


import os, sys
from PIL import Image
import iset2tf  # Created to read images instead of the original db format
import numpy
from sys import argv
import datetime
join = os.path.join


script, light, distance, loop = argv
# light = "15"
# distance = "140"
# loop = "500"




# Example in how to launch it from command line
# nohup python scienStanfordPoster.py 15 140 10 >> 1514010.txt &


# Check that we are in the home directory to make all paths relative
workdir = os.getcwd()
if not workdir.split('/')[-1] == 'WLletterClass':
    print 'ERROR: change the work dir to the home folder of the project'
    sys.exit(0)

# Read first the original data in png format to check that the reading and
# converting system works fine and yields exactly the same results as above.
# This is going to read Yann Lecun's data
# import input_data
# isetmnist = input_data.read_data_sets("data/", one_hot=True)
#
# This is going to read the same data but that it was in png format, so it will check
# that my script is working for other type of data as well.
# train_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/training'
# test_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/testing'
# It yields the same results, so we can continue with iset generated data.

# From now on we can change the images to the iset generated ones, we will have
# to make changes to the parameters in the script to optmize results

# EYE: To use the 72x88 images with the b&w cone voltage information
# train_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/training'
# test_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/testing'




# SENSOR: To use the images with the b&w sensor voltage information for scien
# fov = [0.8, 1, 1.2];
# sceneLights = [5, 15, 45];
# pixelSizes = [1.1e-06, 1.25e-06, 1.4e-06];
# dist = [35, 70, 140]
# f = fov[0]
# s = sceneLights[1]
# p = pixelSizes[0]
# d = dist[2]
s = light
d = distance

train_imagePath = str('data/Light_' + str(s) + '_DistFt_' + str(d) + '/train')
test_imagePath = str('data/Light_' + str(s) + '_DistFt_' + str(d) + '/test')


# BIG MNIST: To use the 64x64 images, upsampled bilinearly from the 28x28 ones
train_imagePath = 'data/origMnistSmallUpsampled/train'
test_imagePath = 'data/origMnistSmallUpsampled/test'


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
# For the poster I am using 64x64
numCats = isetmnist.train.labels[0].shape[0]
imSize = isetmnist.train.images[0,:].shape[0]
imW = int(numpy.sqrt(imSize))
imH = imW

# im = Image.open(join(train_imagePath, '0', '000000.png'))
# if not imSize == im.size[0]*im.size[1]:
#     print 'ERROR: image size in numpy and .png is not the same'
#     sys.exit(1)
# else:
#     imH = im.size[1]
#     imW = im.size[0]


# ****************************************
# ****************************************
# BUILD A MULTILAYER CONVOLUTIONAL NETWORK
# ****************************************
# ****************************************
# Getting 91% accuracy on MNIST is bad. It's almost embarrassingly bad.
# In this section, we'll fix that, jumping from a very simple model to something 
# moderatly sophisticated: a small convolutional neural network. 
# This will get us to around 99.2% accuracy -- not state of the art, but 
# respectable.

import tensorflow as tf
# sess = tf.InteractiveSession()  # This is for interactive

x = tf.placeholder("float", shape=[None, imSize])
y_ = tf.placeholder("float", shape=[None, numCats])


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

# Number of steps = 2  >> /2 and /2  so 7 comes from 28 / 4

W_fc1 = weight_variable([(imW/4) * (imH/4) * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, (imW/4) * (imH/4) * 64])
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
cost = cross_entropy
# cross_entropy = tf.Print(cross_entropy, [cross_entropy], "CrossE")

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # it was 1e-4
# train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
# sess.run(tf.initialize_all_variables())  #for interactiveSession
init = tf.initialize_all_variables()





# Create a summary to monitor cost function
tf.scalar_summary("loss function", cost)
# Merge all summaries to a single operator
merged_summary_op = tf.merge_all_summaries()
# Create the log folder
logDir = join(workdir, 'logs', str(light+'_'+distance+'_'+loop))
# logDir = join(workdir, 'logs', 'mnistUpsampledGOLD10000')
if not os.path.isdir(logDir):
    os.mkdir(logDir)


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Set logs writer into logDir
    summary_writer = tf.train.SummaryWriter(logDir, graph_def=sess.graph_def)

    for i in range(int(loop)):
      batch = isetmnist.train.next_batch(50)
      # TRAIN
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

      # PRINT AND LOG
      if i%10 == 0:
        feed_dict = {x:batch[0], y_: batch[1], keep_prob: 1.0}
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print "step %d, training accuracy %g"%(i, train_accuracy),
        print ', Light(', light, '), Dist(', distance, '),', str(datetime.datetime.now()).split('.')[0]
        # LOGS
        summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, i)


    # Print the test accuraccy within the session
    print "test accuracy %g"%accuracy.eval(feed_dict={
        x: isetmnist.test.images, y_: isetmnist.test.labels, keep_prob: 1.0})

# The final test set accuracy after running this code should be approximately 
# 99.2%. (this is for the 60,000 + 10,000  set run with loop = 20,000

