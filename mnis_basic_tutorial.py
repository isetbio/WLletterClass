# http://www.tensorflow.org/tutorials/mnist/beginners/index.md

# Import mnist database and adapt it the tf way
# import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




# Start with the project
import tensorflow as tf

# This is the formula we are going to use
# y=softmax(Wx+b)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Now we implement the model
# matmul means Matrix Multiplication.
# Loo the theory behind it in the URL
y = tf.nn.softmax(tf.matmul(x,W) + b)



# TRAINING
# To implement cross-entropy we need to first add a new placeholder to input 
# the correct answers:
y_ = tf.placeholder("float", [None,10])

# Then we can implement the cross-entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# In this case, we ask TensorFlow to minimize cross_entropy using the gradient 
# descent algorithm with a learning rate of 0.01. Gradient descent is a simple 
# procedure, where TensorFlow simply shifts each variable a little bit in the 
# direction that reduces the cost. But TensorFlow also provides many other 
# optimization algorithms: using one is as simple as tweaking one line.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize the variables we created:
init = tf.initialize_all_variables()

# We can now launch the model in a Session, and run the operation that 
# initializes the variables:
sess = tf.Session()
sess.run(init)


# Let's train -- we'll run the training step 1000 times!
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
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
# That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Finally, we ask for our accuracy on our test data.
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
