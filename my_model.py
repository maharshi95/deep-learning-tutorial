import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as pp

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

print(mnist.train)

n_features = 784

n_classes = 10

batch_size = 100

n_layers = 4

n_hidden_layers = n_layers - 1

n_nodes = (n_features, 50, 50, 50, n_classes)

activation = (tf.nn.relu, tf.nn.relu, tf.nn.relu)

x = tf.placeholder(tf.float32, [n_features, None])
y = tf.placeholder(tf.float32)

def nn_model(data):
    layers = [
        {
            'W': tf.Variable(tf.random_normal([n_nodes[i], n_nodes[i-1]])),
            'b': tf.Variable(tf.zeros([n_nodes[i], 1]))
        }
        for i in range(1, len(n_nodes))
    ]

    a = data
    for i in range(n_hidden_layers):
         z = tf.matmul(layers[i]['W'], a) + layers[i]['b']
         a = activation[i](z)

    output = tf.matmul(layers[-1]['W'], a) + layers[-1]['b']

    return output

def train_nn(x):
    prediction = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        losses = []

        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x.T, y: epoch_y.T})
                epoch_loss += c
            losses.append(epoch_loss)
            print('Epoch', epoch, 'completed out of', hm_epochs, ' loss:',  epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy, losses

acc, losses = train_nn(x)
x = np.arange(len(losses))
pp.plot(x, losses)
pp.show()

