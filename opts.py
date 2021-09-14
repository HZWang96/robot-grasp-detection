import argparse
import os
import sys

class opts(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='Initial learning rate.')

        parser.add_argument('--data_dir', type=str, default='/root/imagenet-data',
                            help='Directory with training data.')

        parser.add_argument('--num_epochs', type=int, default=None,
                            help='Number of epochs to run trainer.')
        
        parser.add_argument('--batch_size', type=int, default=64,
                            help='Batch size.')
        
        parser.add_argument('--log_dir', type=str, default='/tmp/tf', 
                            help='Tensorboard log_dir.')
        
        parser.add_argument('--model_path', type=str, default='/tmp/tf/model.ckpt',
                            help='Variables for the model.')
        
        parser.add_argument('--resnet_depth', type=int, default=50,
                            help='Number of residual blocks in the ResNet backbone')
    
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        return opt