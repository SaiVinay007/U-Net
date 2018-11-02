import argparse
import tensorflow as tf
import numpy as np
from Unet import Unet
from input_utils import segmentation_data


parser = argparse.ArgumentParser(description="Inputs to the code")

parser.add_argument("--input_record_file",type=str,help="path to TFRecord file with training examples")
parser.add_argument("--batch_size",type=int,default=16,help="Batch Size")
parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
parser.add_argument("--ckpt_savedir",type = str,default='./checkpoints/model_ckpt',help="path to save checkpoints")
parser.add_argument("--load_ckpt",type = str,default='./checkpoints',help="path to load checkpoints from")
parser.add_argument("--save_freq",type = int,default=100,help="save frequency")
parser.add_argument("--display_step",type = int,default=1,help="display frequency")
parser.add_argument("--summary_freq",type = int,default=100,help="summary writer frequency")
parser.add_argument("--no_epochs",type=int,default=10,help="number of epochs for training")


args = parser.parse_args()

TFRecord_file = args.input_record_file

# loading data for segmentation


next_element,init_op = segmentation_data('/home/saivinay/Documents/U-Net/data/train/image/',
                                        '/home/saivinay/Documents/U-Net/data/train/label')

# use next element as input to model

sess = tf.Session()
sess.run(init_op)


image,mask = sess.run(next_element)


model = Unet(image)


if __name__ = '__main__':

    