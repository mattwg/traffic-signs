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
features_count = 32 * 32
batch_size=100
epochs = 2000
learning_rate = 0.0001
early_stopping_rounds = 10
dropout_probability = 0.2
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

X_train_balanced = np.empty([2500*43, 32, 32, 3], dtype=np.uint8)
y_train_balanced = np.empty([2500*43])
indices = np.arange(0,len(X_train))
start_idx = 0
for c in range(0,n_classes):
    bidx = (y_train == c)
    class_indices = indices[bidx]
    n_imgs = len(class_indices)
    n_new_imgs = 2500 - n_imgs
    end_idx = start_idx + n_imgs
    #print(c, n_imgs, n_new_imgs, start_idx, end_idx, end_idx + n_new_imgs)
    # copy old images over
    X_train_balanced[start_idx:end_idx,:,:,:] = X_train[class_indices,:,:,:]
    y_train_balanced[start_idx:end_idx] = y_train[class_indices]
    # sample remaining images
    new_idx = np.random.choice(class_indices,n_new_imgs)
    for i, j in enumerate(new_idx):
        X_train_balanced[end_idx+i,:,:,:] = h.random_rotate(X_train[j,:,:,:])
        y_train_balanced[end_idx+i] = y_train[j
                                                   ]
    start_idx = start_idx + 2500
   
X_train = X_train_balanced
y_train = y_train_balanced
n_train = len(X_train_balanced)

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
# Convolutional Neural Net - LeNet
#---------------------------------------------

def LeNet(x):
    x = tf.reshape(x, (-1, 32, 32, 1))
    # Convolution layer 1. The output shape should be 28x28x24.
    x=h.conv_layer(input=x, num_input_channels=1, filter_size=5, num_filters=24, stride=1, padding='VALID')
    # Activation 1. Your choice of activation function.
    x=tf.nn.relu(x)
    # Pooling layer 1. The output shape should be 14x14x24.
    x=tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Convolution layer 2. The output shape should be 10x10x24.
    x=h.conv_layer(input=x, num_input_channels=24, filter_size=5, num_filters=48, stride=1, padding='VALID')
    # Activation 2. Your choice of activation function.
    x=tf.nn.relu(x)
    # Pooling layer 2. The output shape should be 5x5x16.
    x=tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
    x=tf.contrib.layers.flatten(x)
    # Fully connected layer 1. This should have 200 outputs. 
    x=h.fully_connected_layer(x, 1200, 200)
    # Activation 3. Your choice of activation function.
    x=tf.nn.relu(x)
    # Fully connected layer 2. This should have 43 outputs. With dropout.
    #x=tf.nn.dropout(x, dropout_prob)
    x=h.fully_connected_layer(x, 200, 43)
    # Return the result of the last fully connected layer.
    return x


#---------------------------------------------
# Construct Tensorflow Graph
#---------------------------------------------

# Important in Notebooks!
tf.reset_default_graph()

dropout_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, features_count], name='X')
y = tf.placeholder(tf.float32, [None, n_classes], name='y') 
# Need these when restoring model checkpoint:
tf.add_to_collection("X",X)
tf.add_to_collection("y",y)

train_feed_dict = { X: X_train_gray_flat, y: train_labels, dropout_prob: dropout_probability}
valid_feed_dict = { X: X_valid_gray_flat, y: valid_labels, dropout_prob: 0.0}
test_feed_dict  = { X: X_test_gray_flat,  y: test_labels, dropout_prob: 0.0}

with tf.name_scope('cnn'):
    output = LeNet(X)
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
                feed_dict={X: batch_features, y: batch_labels, dropout_prob: dropout_probability})

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
        
