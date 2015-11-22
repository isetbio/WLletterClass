# Adapted from:
# http://www.tensorflow.org/tutorials/mnist/beginners/index.md

# Check the script reading the original Lecun data, in order to have sthg to compare to
# import input_data
# isetmnist = input_data.read_data_sets("/Users/nupic/code/tfnotebook/MNIST_data", one_hot=True)

import iset2tf  # Created to read images instead of the original db format
import numpy

# Read first the original data in png format to check that the reading and
# converting system works fine and yields exactly the same results as above.
# train_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/training'
# test_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/testing'


# From now on we can change the images to the iset generated ones, we will have
# to make changes to the parameters in the script to optmize results
train_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/training'
test_imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist/testing'

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
imSize = isetmnist.train.images[0].shape[0]
im = Image.open(join(train_imagePath, '0', '000000.png'))
if not imSize == im:
    print 'ERROR: image size in numpy and .ong is not the same'
else:
    imH = im.size[1]
    imW = im.size[0]


import tensorflow as tf

# This is the formula we are going to use
# y=softmax(Wx+b)

# Initialize variables
x = tf.placeholder("float", [None, imSize])
W = tf.Variable(tf.zeros([imSize,numCats]))
b = tf.Variable(tf.zeros([numCats]))

# Now we implement the model
# matmul means Matrix Multiplication.
# See the theory behind it in the URL
y = tf.nn.softmax(tf.matmul(x,W) + b)



# TRAINING
# To implement cross-entropy we need to first add a new placeholder to input 
# the correct answers:
y_ = tf.placeholder("float", [None,numCats])

# Then we can implement the cross-entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = tf.Print(cross_entropy, [cross_entropy], "CrossE")

# In this case, we ask TensorFlow to minimize cross_entropy using the gradient 
# descent algorithm with a learning rate of 0.01. Gradient descent is a simple 
# procedure, where TensorFlow simply shifts each variable a little bit in the 
# direction that reduces the cost. But TensorFlow also provides many other 
# optimization algorithms: using one is as simple as tweaking one line.
# For iset transformed data I had to make the learning steps smaller.
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# initialize the variables we created:
init = tf.initialize_all_variables()

# We can now launch the model in a Session, and run the operation that 
# initializes the variables:
sess = tf.Session()
sess.run(init)


# Let's train -- we'll run the training step 1000 times!
for i in range(1000):
    batch_xs, batch_ys = isetmnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# EVALUATING OUR MODEL
# How well does our model do?
# Well, first let's figure out where we predicted the correct label. 
# tf.argmax is an extremely useful function which gives you the index of the 
# highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the 
# label our model thinks is most likely for each input, while tf.argmax(y_,1) is 
# the correct label. We can use tf.equal to check if our prediction matches the 
# truth.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# That gives us a list of booleans. To determine what fraction are correct,
# we cast to floating point numbers and then take the mean. For example,
# [True, False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Finally, we ask for our accuracy on our test data.
print sess.run(accuracy, feed_dict={x: isetmnist.test.images,
                                    y_: isetmnist.test.labels})
