import tensorflow as tf
import pdb
class AutoEncoder:

    def __init__(self,input,hidden,learning_rate=0.01,training_epochs=50,
                 batch_size = 100, display_step = 10):
        print('hello,world\n')
        self.X = input
        self.hidden = hidden
        self.weights = []
        self.biases = []
        self.inputfeature = input.shape[1]
        self.learning_rate = learning_rate
        self.trainning_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step
    def initialPara(self):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.inputfeature,self.hidden])),
            'decoder_h1': tf.Variable(tf.random_normal([self.hidden,self.inputfeature]))
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.hidden])),
            'decoder_b1': tf.Variable(tf.random_normal([self.inputfeature]))
        }
        self.weights = weights
        self.biases = biases
    def encoder(self,X):
        layer = tf.nn.sigmoid(
            tf.add(
                tf.matmul(X, self.weights['encoder_h1']),self.biases['encoder_b1']
            )
        )
        return layer
    def decoder(self,X):
        layer = tf.nn.sigmoid(
            tf.add(
                tf.matmul(X, self.weights['decoder_h1']),self.biases['decoder_b1']
            )
        )
        return layer

    def train(self):
	#self.X = tf.placeholder(tf.float32,X.shape[1])
        pdb.set_trace()
	X = self.X
        batch_size = self.batch_size

        self.initialPara()

        encoder_op = self.encoder(X)
        decoder_op = self.decoder(encoder_op)

        y_pred = decoder_op
        y_true = X

        # define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(
            tf.pow(y_true-y_pred,2)
        )
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)

        init = tf.initialize_all_variables()

        # launch the graph
        with tf.Session() as sess:
            sess.run(init)
            #total_batch = int(X.shape[0])/float(batch_size)
            total_batch = int(X.size)/float(batch_size)
		# training cycle
            for epoch in range(self.trainning_epochs):
                # loop over all batches
                for i in range(total_batch):
                    batch_xs = X[i*batch_size:(i+1)*batch_size]
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                #display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d'%(epoch+1),
                          "cost=","{:.9f}".foramt(c))

            print("optimization finished!!")

        self.encoderOp = encoder_op
        self.decoderOp = decoder_op
