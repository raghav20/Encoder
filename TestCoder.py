from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pdb
df=pd.read_csv('/home/ubuntu/test/round2.csv')
pdb.set_trace()
#df=df.drop(df.columns[0], axis=1)
#df1, df2 = df[:25000, :], df[25000:, :] if len(df) > 25000 else df, None
df1=df.head(25000)
df2=df.tail(len(df)-25000)
#trY=df1['ACTION'].as_matrix()
#teY=df2['ACTION'].as_matrix()
#df1=df1.drop(df.columns[9], axis=1)
#df2=df2.drop(df.columns[9], axis=1)
trX=df1.as_matrix()
teX=df2.as_matrix()



# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 20 # 1st layer num features
n_hidden_2 = 5 # 2nd layer num features
n_input = trX.shape[1] # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()



# Launch the graph
# Using InteractiveSession (more convenient while using Notebooks)
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(trX.shape[0]/batch_size)
for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs= trX[batch_size*i:batch_size*(i+1)]
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))

print("Optimization Finished!")

# Applying encode and decode over test set
encode_decode = sess.run(
    y_pred, feed_dict={X: teX})

