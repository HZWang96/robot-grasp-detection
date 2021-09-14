import torch.utils.data as data
import os
import cv2
import json


class GraspDataset(data.Dataset):
    def __init__(self, img_dir, annotations_file):
        self.img_dir = img_dir
        f = open(annotations_file, 'r')
        self.img_labels = json.load(f)

    def __len__(self):
        return len(self.img_labels['images'])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels['images'][idx]['file_name'])
        print(idx, img_path)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        label = self.img_labels['annotations'][idx]['coordinates']
        return img, label