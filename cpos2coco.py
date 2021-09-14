import json
import os
from tqdm import tqdm
import glob

# Root path
root_path = '../datasets/cornell_grasping_dataset'

# Labels path
originLabelsDir = '../datasets/cornell_grasping_dataset/data-1'

# Converted file save path
saveDir = '../datasets/cornell_grasping_dataset/cgd_annos.txt'

# Picture path corresponding to the original label
originImagesDir = originLabelsDir

txtFileList = glob.glob(os.path.join(originLabelsDir, '*cpos.txt'))
# print(txtFileList)

# List storing the number of labels per image
labels_per_img = {}

label_count = 0
num_pics = 0
num_annos = 0

split = 'train'

def _process_bboxes(name):
    '''Create a list with the coordinates of the grasping rectangles. Every 
    element is either x or y of a vertex.'''
    with open(name, 'r') as f:
        bboxes = list(map(
              lambda coordinate: float(coordinate), f.read().strip().split()))
    return bboxes

dataset = {'images': [], 'annotations': []}

H, W = 480, 640  # width and height of images in CGD

# ------------ Part I: Converting all coordinate labels in  to COCO format and store them in json file ------------
with open(saveDir, 'w') as fw:
    for txtFile in tqdm(txtFileList, desc="Generating annotations json file... ", ascii=False, ncols=120):
        # print('\n\n'+txtFile.split('/')[-1])
        bboxes = _process_bboxes(txtFile)
        # Count the number of lables for each image
        label_count = len(bboxes)/2

        dataset['images'].append({'file_name': txtFile.split('/')[-1], 'id': num_pics, 'width': W, 'height': H})

        dataset['annotations'].append({'image_id': num_pics, 'coordinates': bboxes})

        num_pics += 1
        
        labels_per_img[txtFile.split('/')[-1]] = label_count
        num_annos += label_count
        print('{} images handled'.format(num_pics))

# print("Labels per image: {}".format(labels_per_img))
print("Done!")
print("Number of images:", num_pics)
print("Number of annotations:", num_annos)

#Save the resulting folder
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(split))
with open(json_name, 'w') as f:
    json.dump(dataset, f)

f = open(os.path.join(root_path, 'annotations/train.json'), 'r')
img_labels = json.load(f)
print(len(img_labels['images']))
