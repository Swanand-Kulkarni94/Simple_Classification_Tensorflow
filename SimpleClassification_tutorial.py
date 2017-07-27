#Program 6, Simple Classification program using Tensorflow

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

tf.set_random_seed(1)
np.random.seed(1)

#Generate dumy data
n_data = np.ones((100, 2))
x0 = np.random.normal(2*n_data, 1) #Class0 x shape = (100, 2)
y0 = np.zeros(100)				   #Class0 y shape = (100, 1)
x1 = np.random.normal(-2*n_data, 1) #Class1 x shape = (100, 2)
y1 = np.ones(100)
x = np.vstack((x0, x1)) #shape (200, 2) + some noise
y = np.hstack((y0, y1)) #shape (200, )

#Plot data on the graph
plt.scatter(x[:, 0], x[:, 1], c = y, s = 100, lw = 0, cmap = 'RdYlGn')
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.int32, y.shape)

#Neural Network layers
layer1 = tf.layers.dense(tf_x, 10, tf.nn.relu) #Hidden Layers
output = tf.layers.dense(layer1, 2) #Output Layers

loss = tf.losses.sparse_softmax_cross_entropy(labels = tf_y, logits = output)
accuracy = tf.metrics.accuracy(labels = tf.squeeze(tf_y), predictions = tf.argmax(output, axis = 1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.05)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op) #Initialize var in graph

plt.ion()
for step in range(100):
	#Train and obtain ouput
	_, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})
	if step % 2 == 0:
		#Plot and display learning process
		plt.cla()
		plt.scatter(x[:, 0], x[:, 1], c = pred.argmax(1), s = 100, lw = 0, cmap = 'RdYlGn')
		plt.text(1.5, -4, 'Accuracy = % 0.2f' % acc, fontdict = {'size': 20, 'color': 'red'})
		plt.pause(0.1)

plt.ioff()
plt.show()