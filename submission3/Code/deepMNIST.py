import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Input Data - MNIST 
mnist = input_data.read_data_sets("Data", one_hot = True)

#Placeholders for images -> input and logits -> output
x = tf.placeholder('float', [None, 784])
y_ = tf.placeholder('float')


#Initialize Weights
def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1,name=name)
  return tf.Variable(initial)

#Initialize Bias
def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape,name=name)
  return tf.Variable(initial)

#Convolution  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Max Pooling   
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

sess  = tf.InteractiveSession()

#1st Covolution Layer		
with tf.name_scope("Conv_1"):				
	W_conv1 = weight_variable([5, 5, 1, 32],name="W_conv1")
	b_conv1 = bias_variable([32],name="b_conv1")
	x_image = tf.reshape(x, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

#2nd Covolution Layer
with tf.name_scope("Conv_2"):
	W_conv2 = weight_variable([5, 5, 32, 64],name="W_conv2")
	b_conv2 = bias_variable([64],name="b_conv2")
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

#Fully Connected Layer
with tf.name_scope("Fully_Connected_Layer"):
	W_fc1 = weight_variable([7 * 7 * 64, 1024],name="W_fc1")
	b_fc1 = bias_variable([1024],name="b_fc1")
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
with tf.name_scope("Dropout"):
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Final Output Layer
with tf.name_scope("Output_Layer"):
	W_fc2 = weight_variable([1024, 10],name="W_fc2")
	b_fc2 = bias_variable([10],name="b_fc2")
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Cost
with tf.name_scope("Cost"):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	tf.summary.scalar("Cost",cross_entropy)

#Accuracy
with tf.name_scope("Accuracy"):
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar("Accuracy",accuracy)

#Initialize all variables	
sess.run(tf.global_variables_initializer())

#Write tensorflow graph to local machine
writer = tf.summary.FileWriter("C:/Users/cmp/Deep/Semester 6/AOBD/Projects/Project 2/Codes/2.DNN/Tensor_Flow_Deep_MNIST_CNN/nn_logs_graphs/",sess.graph)
merged = tf.summary.merge_all()

#Training and Evalution
for i in range(1000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		summary,acc = sess.run([merged,accuracy],feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		writer.add_summary(summary,i)
		print("Step %d, Training accuracy %g"%(i, acc))
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		
	
print("Test Accuracy -> %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))