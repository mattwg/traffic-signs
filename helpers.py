import math
import pickle
import os
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from itertools import product
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.charts import Line, show

import tensorflow as tf
from tensorflow.contrib.layers import flatten

#Print out single image
def plot_image(img, width, height):
    fig = plt.figure(1, (width, height))
    plt.imshow(img)
    plt.axis('off')
    return plt

# Print out images in rows of length n_col
def plot_images(images, n_cols, grayscale = False):
    n_rows = int(np.ceil(len(images) / n_cols))
    fig = plt.figure(figsize=(n_cols, n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
    for i, xy in enumerate(product(range(0,n_rows), range(0,n_cols))):
        ax = plt.subplot(gs[xy[0], xy[1]])
        if grayscale:
            ax.imshow(images[i], cmap='gray')
        else:
            ax.imshow(images[i])
            
        ax.set_xticks([])
        ax.set_yticks([])

# Return a sample of image indexes based on the class
def sample_indices(images, image_class, n_sample):
    n_classes = len(set(image_class))
    indices = np.arange(0,len(images))
    result = np.empty(0,dtype=int)
    for i in range(0,n_classes):
        bidx = (image_class == i)
        class_indices = indices[bidx]
        n_images = len(class_indices)
        sample_indices = np.random.choice(class_indices,n_sample)
        result = np.concatenate((result, sample_indices))
    return result

def normalize_grayscale(image_data, rng = (0.1, 0.9)):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = rng[0]
    b = rng[1]
    x_min = np.min(image_data)
    x_max = np.max(image_data)
    return a + (((image_data - x_min)*(b - a))/(x_max - x_min))

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def flatten_all_gray(images):
    images_flat = []
    for i, img in enumerate(images):
        images_flat.append(np.array(images[i,:,:], dtype=np.float32).flatten())
    return np.array(images_flat)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def conv_layer(input,                  # previous layer.
                   num_input_channels, # depth of previous layer
                   filter_size,        # filter width/height
                   num_filters,        # number of filters
                   stride,             # stride size
                   padding):           # padding 'VALID' or 'SAME'
    
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, stride, stride, 1],
                         padding=padding)
    layer += biases
    return layer

def fully_connected_layer(input,              # previous layer.
                   num_inputs,                # number of inputs
                   num_outputs):              # number of outputs
    
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    return layer

def train_model(model_name, init, loss, features, labels, 
                X_train, y_train, train_feed_dict, valid_feed_dict, 
                accuracy, predicted_class,
                epochs, batch_size, learning_rate, early_stopping_rounds, opt="GD"):

    if (opt == "GD"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
    # For saving the tensorflow variables
    saver = tf.train.Saver()

    # The accuracy measured against the validation set
    validation_accuracy = 0.0

    # Measurements use for graphing loss and accuracy
    log_batch_step = 50
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []

    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(X_train)/batch_size))

        # Early stopping flags
        epoch_i = 0
        continue_training = True
        consider_stopping = False
        epochs_since_better = 0
        best_accuracy = 0.0

        with tqdm(range(epochs), unit='epochs') as pbar: 
            while epoch_i < epochs and continue_training:
                batch_i = 0
                while batch_i < batch_count and continue_training:

                    # Get a batch of training features and labels
                    batch_start = batch_i*batch_size
                    batch_features = X_train[batch_start:batch_start + batch_size]
                    batch_labels = y_train[batch_start:batch_start + batch_size]

                    # Run optimizer and get loss
                    _, l = session.run(
                        [optimizer, loss],
                        feed_dict={features: batch_features, labels: batch_labels})

                    # Log every 50 batches
                    if not batch_i % log_batch_step:
                        # Calculate Training and Validation accuracy
                        training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                        # Log batches
                        previous_batch = batches[-1] if batches else 0
                        batches.append(log_batch_step + previous_batch)
                        loss_batch.append(l)
                        train_acc_batch.append(training_accuracy)
                        valid_acc_batch.append(validation_accuracy)

                    batch_i += 1

                # Check accuracy against Validation data
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)
                # Get predicted class
                y_pred = session.run(predicted_class, feed_dict=valid_feed_dict)

                # Early stopping?
                if validation_accuracy < best_accuracy:    
                    if consider_stopping:
                        epochs_since_better += 1
                    else:
                        consider_stopping = True
                        epochs_since_better = 1
                else:
                    best_accuracy = validation_accuracy
                    saver.save(session, model_name + ".ckpt")
                    consider_stopping = False

                if epochs_since_better > early_stopping_rounds:
                        print('Stopping no improvement for {} epochs'.format(early_stopping_rounds))
                        continue_training = False

                epoch_i += 1
                pbar.update(1)

        # Restore best model
        saver.restore(session, model_name + ".ckpt")
        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)
        # Get predicted class
        y_pred = session.run(predicted_class, feed_dict=valid_feed_dict)

        result = { "model" : model_name+".ckpt",
                   "validation_accuracy" : validation_accuracy,
                   "y_pred" : y_pred,
                   "loss_batch" : loss_batch,
                   "train_acc_batch" : train_acc_batch,
                   "valid_acc_batch" : valid_acc_batch,
                   "epochs" : epochs,
                   "batch_size" : batch_size,
                   "learning_rate" : learning_rate
                 }
        
    return result 

def serialize_training_data(pickle_file, result):
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                result,
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

def deserialize_training_data(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            result = pickle.load(f)
    except Exception as e:
        print('Unable to load data from',pickle_file, ":", e)
    return result    
    
def training_plots(result):
    p = Line({'train loss':result['loss_batch']}, xlabel='Batch', ylabel='Train loss',width=900, height=300)
    show(p)
    p = Line({'train':result['train_acc_batch'], 'valid': result['valid_acc_batch']}, 
         xlabel='Batch', ylabel='Accuracy', width=900, height=300)
    show(p)
    
def plot_contingency_matrix(cm, n_classes, names):
    cmax=np.max(cm)
    cmx = np.ndarray(shape=(n_classes**2))
    cmy = np.ndarray(shape=(n_classes**2))
    cmz = np.ndarray(shape=(n_classes**2))
    alpha = np.ndarray(shape=(n_classes**2))
    xn = np.ndarray(shape=(n_classes**2),dtype=np.chararray)
    yn = np.ndarray(shape=(n_classes**2),dtype=np.chararray)
    cmx.shape
    i = 0
    for x in range(0,n_classes):
        for y in range (0,n_classes):
            cmx[i] = x
            cmy[i] = y
            cmz[i] = cm[x,y]
            if cmz[i] == 0:
                alpha[i] = 0
            else:
                alpha[i] = 0.04 + 0.96 * (cmz[i]/cmax)
            xn[i] = names[x]
            yn[i] = names[y]
            i=i+1   

    p = figure(title='Confusion Matrix', x_axis_location="above", 
               tools="pan,wheel_zoom,box_zoom,reset,hover,save",  
               x_range=names,
               y_range=list(reversed(names)))
    source = ColumnDataSource(data=dict(
            cmx=cmx,
            cmy=cmy,
            cmz=cmz,
            alpha=alpha,
            xn=xn, yn=yn))
    p.rect('xn', 'yn', width=0.9, height=0.9, source=source,
            color="#0000FF", alpha = 'alpha', line_width = 2) 
            #width_units="screen", height_units="screen")
    p.select_one(HoverTool).tooltips = [
            ('actual, predicted', '@xn, @yn'),
            ('count', '@cmz')]
    p.axis.major_label_text_font_size = "6pt"
    p.axis.major_label_standoff = 1
    p.xaxis.major_label_orientation = np.pi/3
    p.plot_width = 900
    p.plot_height = 900
    return(p)
    
def plot_false_class(pred_class, image_class):
    idx = ( pred_class != image_class )
    plt.hist([image_class[idx]],bins=max(np.unique(image_class)))

def plot_true_class(pred_class, image_class):
    idx = ( pred_class == image_class )
    plt.hist([image_class[idx]],bins=max(np.unique(image_class)))

# sample a correctly classified from each class
# sample n incorrect from each class
# display in a column
def sample_misclassified(pred_class, image_class, n_sample_true = 2, n_sample_false = 8):
    n_classes = max(image_class)
    indices = np.arange(0,len(image_class))
    result = np.zeros(shape=[n_classes,n_sample_true + n_sample_false],dtype=int)
    for i in range(0,n_classes):
        bidx_t = np.array([ p==a and a==i for p, a in zip(pred_class, image_class) ])
        bidx_f = np.array([ p!=a and p==i for p, a in zip(pred_class, image_class) ])
        idx = indices[bidx_t]
        if len(idx) > 0 :
            sample_indices = np.random.choice(idx,n_sample_true)
            result[i,0:n_sample_true] = sample_indices
            idx = indices[bidx_f]
            if len(idx) > 0:
                n = min(len(idx), n_sample_false)
                sample_indices = np.random.choice(idx,n)
                result[i, n_sample_true:n_sample_true+n] = sample_indices
    
    return result

# Print out image in grid
def plot_image_grid(images, indices, grayscale = False):
    n_rows = indices.shape[0]
    n_cols = indices.shape[1]
    fig = plt.figure(figsize=(n_cols, n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
    for i, xy in enumerate(product(range(0,n_rows), range(0,n_cols))):
        ax = plt.subplot(gs[xy[0], xy[1]])
        idx = indices[xy[0], xy[1]]
        if idx > 0:
            if grayscale:
                ax.imshow(images[idx], cmap='gray')
            else:
                ax.imshow(images[idx])  
        ax.set_xticks([])
        ax.set_yticks([])
        
        
#rotate around random point from center of image
def random_rotate(img):
    x = np.random.uniform(12)+10
    y = np.random.uniform(12)+10
    ang = np.random.uniform(90)-45
    M = cv2.getRotationMatrix2D((x,y),ang,1)
    return cv2.warpAffine(img,M,(32,32))