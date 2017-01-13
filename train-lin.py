import pickle
import pandas as pd
import math
import os
import numpy as np
from itertools import product
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.python.ops.variables import Variable
from tensorflow.contrib.layers import flatten

import helpers as h

#---------------------------------------------
# Training parameters
#---------------------------------------------
training_file = '/home/ubuntu/data/train.p'
testing_file = '/home/ubuntu/data/test.p'
n_hidden_layer = 400
features_count = 32 * 32
batch_size=100
epochs = 2000
learning_rate = 0.001
early_stopping_rounds = 10
TRAIN_DIR = 'logs/'

#---------------------------------------------
# Load data
#---------------------------------------------
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_test = len(X_test)
image_shape = X_train.shape[1]
n_classes = len(set(y_train))

#---------------------------------------------
# Preprocessing of images 
#---------------------------------------------

X_train_gray = np.empty( [n_train, image_shape, image_shape], dtype = np.int32)
for i, img in enumerate(X_train):
    X_train_gray[i,:,:] = cv2.equalizeHist(h.grayscale(img), (0, 254) )
X_test_gray = np.empty( [n_test, image_shape, image_shape], dtype = np.int32)
for i, img in enumerate(X_test):
    X_test_gray[i,:,:] = cv2.equalizeHist(h.grayscale(img), (0, 254) )
   
X_train_gray, X_valid_gray, y_train, y_valid = train_test_split( 
    X_train_gray,
    y_train,
    test_size=0.10,
    random_state=1973)

encoder = LabelBinarizer()
encoder.fit(y_train)
train_labels = encoder.transform(y_train)
valid_labels = encoder.transform(y_valid)
test_labels = encoder.transform(y_test)

# Change to float32 for Tensorflow
train_labels = train_labels.astype(np.float32)
valid_labels = valid_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

X_train_gray_flat = h.flatten_all_gray(X_train_gray)
X_valid_gray_flat = h.flatten_all_gray(X_valid_gray)
X_test_gray_flat = h.flatten_all_gray(X_test_gray)

X_train_gray_flat = h.normalize_grayscale(X_train_gray_flat)
X_valid_gray_flat = h.normalize_grayscale(X_valid_gray_flat)
X_test_gray_flat = h.normalize_grayscale(X_test_gray_flat)

#---------------------------------------------
# Construct Tensorflow Graph
#---------------------------------------------

# Important in Notebooks!
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, features_count], name='X')
y = tf.placeholder(tf.float32, [None, n_classes], name='y') 
# Need these when restoring model checkpoint:
tf.add_to_collection("X",X)
tf.add_to_collection("y",y)

train_feed_dict = { X: X_train_gray_flat, y: train_labels }
valid_feed_dict = { X: X_valid_gray_flat, y: valid_labels }
test_feed_dict  = { X: X_test_gray_flat,  y: test_labels  }

#---------------------------------------------
# Fully Connected Neural Net 
#---------------------------------------------

with tf.name_scope('hidden_layer1'):
    weights_h1 = tf.Variable(tf.random_normal(shape=[features_count, n_hidden_layer], stddev=0.05, dtype=tf.float32), name='weights')
    biases_h1 = tf.Variable(tf.random_normal(shape=[n_hidden_layer], stddev=0.05, dtype=tf.float32), name='biases')
    hidden_layer1 = tf.nn.relu(tf.matmul(X, weights_h1) + biases_h1)

with tf.name_scope('output'):
    weights_o = tf.Variable(tf.random_normal(shape=[n_hidden_layer, n_classes], stddev=0.05, dtype=tf.float32), name='weights')
    biases_o = tf.Variable(tf.random_normal(shape=[n_classes], stddev=0.05, dtype=tf.float32), name='biases')
    output = tf.matmul(hidden_layer1, weights_o) + biases_o
    tf.add_to_collection('output',output) ## Need this when restoring model checkpoint
    output_probability = tf.nn.softmax(output)
    predicted_class = tf.argmax(output_probability, dimension=1)

is_correct_prediction = tf.equal(tf.argmax(output_probability, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
# For tensorboard
tf.scalar_summary("accuracy", accuracy)

cross_entropy = -tf.reduce_sum(y * tf.log(output_probability), reduction_indices=1, name='cross_entropy')
loss = tf.reduce_mean(cross_entropy, name='loss')
# For tensorboard
tf.scalar_summary("loss", loss)
optimizer = tf.train.AdamOptimizer(learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(TRAIN_DIR)
saver = tf.train.Saver()
init = tf.initialize_all_variables()

#---------------------------------------------
# Train model - execute graph
#---------------------------------------------

batches = int(math.ceil(len(X_train_gray_flat)/batch_size))

# Early stopping flags
epoch_i = 0
continue_training = True
consider_stopping = False
epochs_since_better = 0
best_accuracy = 0.0
        
# Clear out tensorboard logs before training
[ os.remove(TRAIN_DIR + '/' + f) for f in os.listdir(TRAIN_DIR) ]

checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')

with tf.Session() as sess:
    sess.run(init)
    
    writer.add_graph(sess.graph)
        
    while epoch_i < epochs and continue_training:
        print('Epoch {} of {}'.format(epoch_i, epochs-1))
        batch_i = 0
        while batch_i < batches and continue_training:
        
            batch_start = batch_i*batch_size
            batch_features = X_train_gray_flat[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            _, l = sess.run(
                [train_op, loss],
                feed_dict={X: batch_features, y: batch_labels})

            if not batch_i % 50:
                summary, acc = sess.run([merged, accuracy], feed_dict=valid_feed_dict)    
                writer.add_summary(summary, epoch_i)
            
            batch_i += 1
        
        # Check accuracy against Validation data
        validation_accuracy = sess.run(accuracy, feed_dict=valid_feed_dict)
        y_pred = sess.run(predicted_class, feed_dict=valid_feed_dict)

        # Early stopping?
        if validation_accuracy < best_accuracy:    
            if consider_stopping:
                epochs_since_better += 1
            else:
                consider_stopping = True
                epochs_since_better = 1
        else:
            print('Improved accuracy of {} at epoch {}'.format(validation_accuracy,epoch_i))
            best_accuracy = validation_accuracy
            saver.save(sess, checkpoint_file, global_step = epoch_i)
            consider_stopping = False

        if epochs_since_better > early_stopping_rounds:
            print('Stopping no improvement for {} epochs'.format(early_stopping_rounds))
            continue_training = False

        epoch_i += 1
        
