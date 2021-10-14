import numpy as np
import torch
import torch.utils.data
import random
from data import grasp
from opts import opts

from data.grasp import GraspRectangle


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training ResNet-50 in a common format.
    """
    def __init__(self, output_size=224, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        grasp_labels = []
        bbs = self.get_gtbb(idx, rot, zoom_factor)  # <class 'dataset_processing.grasp.GraspRectangles'>
        bbs = bbs.to_array()
        for i in range(bbs.shape[0]):
            grasp_labels.append([GraspRectangle(bbs[i]).center[1], GraspRectangle(bbs[i]).center[0], GraspRectangle(bbs[i]).angle, GraspRectangle(bbs[i]).width, GraspRectangle(bbs[i]).length])
        # pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        
        # width_img = np.clip(width_img, 0.0, 150.0)/150.0

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0# if self.include_depth and self.include_rgb:
        #     x = self.numpy_to_torch(
        #         np.concatenate(
        #             (np.expand_dims(depth_img, 0),
        #              rgb_img),
        #             0
        #         )
        #     )
        # elif self.include_depth:
        #     x = self.numpy_to_torch(depth_img)
        # elif self.include_rgb:
        #     x = self.numpy_to_torch(rgb_img)

                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        # pos = self.numpy_to_torch(pos_img)
        # cos = self.numpy_to_torch(np.cos(2*ang_img))
        # sin = self.numpy_to_torch(np.sin(2*ang_img))
        # width = self.numpy_to_torch(width_img)

        # return x, (pos, cos, sin, width), idx, rot, zoom_factor
        return x, np.array(grasp_labels)

    def __len__(self):
        return len(self.grasp_files)


    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        rgb_img = list()
        grasp_labels = list()
        # labels = list()
        # difficulties = list()
        
        for b in batch:
            rgb_img.append(b[0])
            grasp_labels.append(b[1])
        #     # images.append(b[0])
        #     # boxes.append(b[1])
        #     # labels.append(b[2])
        #     # difficulties.append(b[3])

        rgb_img = torch.stack(rgb_img, dim=0)

        return rgb_img, grasp_labels                # tensor (N, 3, 300, 300), 3 lists of N tensors each
