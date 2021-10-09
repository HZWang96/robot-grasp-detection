import argparse
import os
import sys

class opts(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # parser.add_argument('--learning_rate', type=float, default=0.001,
        #                     help='Initial learning rate.')

        # parser.add_argument('--data_dir', type=str, default='/root/imagenet-data',
        #                     help='Directory with training data.')

        # parser.add_argument('--num_epochs', type=int, default=None,
        #                     help='Number of epochs to run trainer.')
        
        # parser.add_argument('--batch_size', type=int, default=64,
        #                     help='Batch size.')
        
        # parser.add_argument('--log_dir', type=str, default='/tmp/tf', 
        #                     help='Tensorboard log_dir.')
        
        # parser.add_argument('--model_path', type=str, default='/tmp/tf/model.ckpt',
        #                     help='Variables for the model.')
        
        # parser.add_argument('--resnet_depth', type=int, default=50,
        #                     help='Number of residual blocks in the ResNet backbone')

        # Copy from GG-CNN Code
        # Network
        parser.add_argument('--network', type=str, default='ggcnn', help='Network Name in .models')

        # Dataset & Data & Training
        parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
        parser.add_argument('--dataset-path', type=str, help='Path to dataset')
        parser.add_argument('--use-depth', type=int, default=0, help='Use Depth image for training (1/0)')
        parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
        parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
        parser.add_argument('--ds-rotate', type=float, default=0.0,
                            help='Shift the start point of the dataset to use a different test/train split for cross validation.')
        parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

        parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
        parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
        parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
        parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')

        # Logging etc.
        parser.add_argument('--description', type=str, default='', help='Training description')
        parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
        parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
        parser.add_argument('--vis', action='store_true', help='Visualise the training process')

        # Image preprocessing.
        parser.add_argument('--image-size', type=int, default=224, help='Image size')
    
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        return opt