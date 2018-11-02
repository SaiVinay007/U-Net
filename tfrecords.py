import tensorflow as tf
import os
import numpy as np
import cv2
from random import shuffle
import glob
import sys

root = '/home/saivinay/Documents/U-Net/data/train/image'


'''
List images and their labels
'''

def get_filenames():

    filenames = os.listdir(root)
    images = []
    labels = []
    for i in filenames:
        image_file_path = os.path.join(root,i)
        images.append(image_file_path)
        labels.append(image_file_path.replace('image','label'))

    return images,labels

shuffle_data = True

# reading the addrs of images and labels 
train_images,train_labels = get_filenames()

# Shuffling the data
if shuffle_data:
    c = list(zip(train_images,train_labels))    # making a list of pairs of images and corresponding labels
    shuffle(c)                                  # shuffling the list 
    train_images, train_labels = zip(*c)        # unzipping a pair into two values







'''
First we need to load the image and labels
and convert it to the data type (float32 in this example) 
in which we want to save the data into a TFRecords file'''

def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def load_label(addr):
    lab = cv2.imread(addr)
    lab = cv2.resize(lab,(224,244),interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(lab,cv2.COLOR_BGR2RGB)
    lab = lab.astype(np.float32) 
    return lab


# Convert data to features

def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Address to save tfrecord file

train_filename = '/home/saivinay/Documents/U-Net/data/train.tfrecords'

# Writing into the tfrecord file

# open the tfrecord file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_images)):
    # print the number of images saved every 10 images

    if not i%10:
        print('Train data: {}/{}'.format(i,len(train_images)))
        sys.stdout.flush()
        '''
        When ever we execute print statements output will be written to buffer.
        And we will see the output on screen when buffer get flushed(cleared).
        By default buffer will be flushed when program exits.
        BUT WE CAN ALSO FLUSH THE BUFFER MANUALLY by using "sys.stdout.flush()" statement in the program
        '''
    
    # load image and label
    img = load_image(train_images[i])
    lab = load_label(train_labels[i])

    # create a feature
    feature = {'train/label': _bytes_feature(tf.compat.as_bytes(lab.tostring())),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring())) }

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()


'''


There are a couple of things to understand here.
One is the difference between buffered I/O and unbuffered I/O.
The concept is fairly simple - for buffered I/O, there is an internal buffer which is kept.
Only when that buffer is full (or some other event happens, such as it reaches a newline) is the output "flushed".
With unbuffered I/O, whenever a call is made to output something, it will do this, 1 character at a time.

Most I/O functions fall into the buffered category,
mainly for performance reasons: it's a lot faster to write chunks at a time 
(all I/O functions eventually get down to syscalls of some description, which are expensive.)

flush lets you manually choose when you want this internal buffer to be written -
a call to flush will write any characters in the buffer. 
Generally, this isn't needed, because the stream will handle this itself.
** However, there may be situations when you want to make sure something is output before you continue -
this is where you'd use a call to flush().

'''