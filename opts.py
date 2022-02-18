import argparse
import os
import sys

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')

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
        self.parser.add_argument('--network', type=str, default='ResNet50', help='Network Name in .models')
        self.parser.add_argument('--which-gpu', type=int, default=1, help='Choose which gpu for training/validation')
        self.parser.add_argument('--warm-up', type=int, default=79, help='Warming up the learning rate')

        # Dataset & Data & Training
        self.parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
        self.parser.add_argument('--dataset-path', type=str, help='Path to dataset')
        self.parser.add_argument('--use-depth', type=int, default=0, help='Use Depth image for training (1/0)')
        self.parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
        self.parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
        self.parser.add_argument('--ds-rotate', type=float, default=0.0,
                            help='Shift the start point of the dataset to use a different test/train split for cross validation')
        self.parser.add_argument('--num-workers', type=int, default=4, help='Dataset workers')

        self.parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
        self.parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
        self.parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
        self.parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')

        # Logging etc.
        self.parser.add_argument('--description', type=str, default='', help='Training description')
        self.parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
        self.parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
        self.parser.add_argument('--vis', action='store_true', help='Visualise the training process')

        # # Image preprocessing.
        # parser.add_argument('--image-size', type=int, default=224, help='Image size')

        #eval_model
        # Network
        self.parser.add_argument('--trained-network', type=str, help='Path to saved network to evaluate')

        # Dataset & Data & Training
        self.parser.add_argument('--eval-use-depth', type=int, default=0, help='Use Depth image for evaluation (1/0)')
        self.parser.add_argument('--eval-use-rgb', type=int, default=1, help='Use RGB image for evaluation (0/1)')
        self.parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')

        self.parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
        self.parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
        self.parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
        self.parser.add_argument('--eval-vis', action='store_true', help='Visualise the network output')
    
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        return opt
    
    def init(self, args=''):
        opt = self.parse(args)
        return opt