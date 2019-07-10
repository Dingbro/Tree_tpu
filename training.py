import os
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers as KL
from random import uniform
from matplotlib import pyplot as plt
import random
from keras import backend as K
from keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.python.keras import Model

from efficientnet import EfficientNetB3
from efficientnet.initializers import dense_kernel_initializer
print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

#@title custom loss function



""" Weighted binary crossentropy between an output tensor and a target tensor.
# Arguments
    pos_weight: A coefficient to use on the positive examples.
# Returns
    A loss function supposed to be used in model.compile().
"""
def weighted_binary_crossentropy(pos_weight=1):
    def _to_tensor(x, dtype):
        return tf.convert_to_tensor(x, dtype=dtype)
      
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
      
  
  
    def _calculate_weighted_binary_crossentropy(target, output, from_logits=False):
        """Calculate weighted binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = _to_tensor(K.common.epsilon(), output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.log(output / (1 - output))
        return tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=pos_weight)


    def _weighted_binary_crossentropy(y_true, y_pred):
        return K.mean(_calculate_weighted_binary_crossentropy(y_true, y_pred), axis=-1)
    
    return _weighted_binary_crossentropy


AUTO = tf.data.experimental.AUTOTUNE
NUM_BATCH = 512
folder_name = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4' ]
#folder_name = ['fold1', 'fold2']
dataset_dir = 'gs://tree_patch_dataset/v7_before_cleansing/training/tfrec'
val_folder = 'fold0'
weight = [10,1]
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
        
        # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data
        "label":         tf.io.FixedLenFeature([], tf.string),  # one bytestring
        #"size":          tf.io.FixedLenFeature([3], tf.int64),  # two integers
        "filename": tf.io.FixedLenFeature([], tf.string)       # a certain number of floats
    }
    # decode the TFRecor
    example = tf.io.parse_single_example(example, features)
    #height = example['size'][0]
    #width  = example['size'][1]
    #channel = example['size'][2]
    # FixedLenFeature fields are now ready to use: exmple['size']
    # VarLenFeature fields require additional sparse_to_dense decoding
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image,dtype = tf.float32)
    
    #image = tf.python.decode_raw(example['image'], tf.uint8)
    #image = tf.reshape(image, [height,width,channel])
    class_num = example['class']
    
    
    label  = example['label']
    
    filename = example['filename']
    return image, class_num, label, filename

def preprocess(image, class_num, label, filename):
    
    
    '''if random.randint(1,2)==1:
      image = tf.image.random_crop(image, [300,300,3])
      #image = tf.math.divide(image,255)
      image = image/255
      image = image - np.array(MEAN_RGB)
      image = image / np.array(STDDEV_RGB)
      class_num = tf.cast(class_num, tf.float32)
      class_num = tf.expand_dims(class_num, 0)
          
      return image, class_num'''
    
    zoom = uniform(0.5,1)
    image = tf.image.central_crop(image, zoom)
    image = tf.image.resize(image, [420,420])
    image = tf.image.random_crop(image, [300,300,3])
    #image = tf.image.random_hue(image, 0.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
   
    #image = tf.math.divide(image,255)
    
    #image = image/255
    image = image - np.array(MEAN_RGB)
    image = image / np.array(STDDEV_RGB)
    image = tf.image.convert_image_dtype(image,dtype = tf.float32)
    class_num = tf.cast(class_num, tf.float32)

    
    class_num = tf.expand_dims(class_num, 0)
    #return image, class_num, label, filename
    return image, class_num

def make_dataset_helper(dataset):
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset = dataset.repeat()

    return dataset

def make_shuffle_batch(dataset):
    dataset = dataset.map(preprocess)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(NUM_BATCH, drop_remainder=False)
    
    return dataset


def make_dataset(dataset_dir, val_folder, folder_name, weight):
    
    vd_live = tf.data.Dataset.list_files(dataset_dir + '/' + val_folder + '/live/*.tfrec')
    vd_dead = tf.data.Dataset.list_files(dataset_dir + '/' + val_folder + '/dead/*.tfrec')
    
    dead_list =[]
    live_list = []
    for folder in folder_name:
        if folder == val_folder:
            continue
        
        train_live = tf.data.Dataset.list_files(dataset_dir + '/' + folder + '/live/*.tfrec')
        train_dead = tf.data.Dataset.list_files(dataset_dir + '/' + folder + '/dead/*.tfrec')
        #train_temp = tf.data.experimental.sample_from_datasets([train_live,train_dead],  tf.constant(weight, dtype = tf.float32))
        #traindataset_list.append(train_temp)
        dead_list.append(train_dead)
        live_list.append(train_live)
    
    deaddataset = None
    i=0
    for dset in dead_list:
        if i==0:
            deaddataset = dset
            i+=1
            continue
        deaddataset = deaddataset.concatenate(dset)
    
    livedataset = None
    j=0
    for dset in live_list:
        if j==0:
            livedataset = dset
            j+=1
            continue
        livedataset = livedataset.concatenate(dset)
    
    
    livedataset = make_dataset_helper(livedataset)
    deaddataset = make_dataset_helper(deaddataset)
    traindataset = tf.data.experimental.sample_from_datasets([livedataset,deaddataset],  tf.constant(weight, dtype = tf.float32))
    traindataset = make_shuffle_batch(traindataset)
    
    
    vd_livedataset = make_dataset_helper(vd_live)
    vd_deaddataset = make_dataset_helper(vd_dead)
    valdataset =  tf.data.experimental.sample_from_datasets([vd_livedataset,vd_deaddataset],  tf.constant(weight, dtype = tf.float32))
    valdataset = make_shuffle_batch(valdataset)
    
    return traindataset, valdataset

traindataset, valdataset = make_dataset(dataset_dir, val_folder, folder_name, weight)

def construct_model():
    # line 1: how do we keep all layers of this model ?
    model = EfficientNetB3(weights=None, include_top=False, pooling='avg')
    x = model.output
    x = KL.Dropout(0.1)(x)
    x = KL.Dense(1, kernel_initializer=dense_kernel_initializer)(x)
    new_output = KL.Activation('sigmoid')(x)
    new_model = Model(model.inputs, new_output)
    return new_model

resolver = tf.contrib.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.contrib.distribute.initialize_tpu_system(resolver)
strategy = tf.contrib.distribute.TPUStrategy(resolver)

with strategy.scope():
  model = construct_model()

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, ),
      loss=weighted_binary_crossentropy(10),
      #loss = 'binary_crossentropy',
      metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

filepath = 'model/'+'{epoch:02d}-{loss:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, save_weights_only=True, mode='min', save_freq=100)
callbacks_list = [checkpoint]

model.fit(
    traindataset,
    epochs = 100,
    steps_per_epoch = 100,
    validation_data = valdataset,
    validation_steps = 20,
    validation_freq = 10,
    callbacks = callbacks_list
)
