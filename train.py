''' 
Training a network on cornell grasping dataset for detecting grasping positions.
'''
import sys
import os.path
import glob
import torch
import torch.utils.data
import numpy as np
# from logger import Logger
import time
# import img_preproc
from bbox_utils import *
from models.model_utils import *
from opts import opts
from shapely.geometry import Polygon
from data.grasp_data import GraspDataset
# import tensorflow as tf


DATA_PATH = '../datasets/cornell_grasping_dataset/data-1'
ANN_PATH = '../datasets/cornell_grasping_dataset/annotations/train.json'


def train():
    opt = opts()
    # logger = Logger(opt)

    print('Creating model...')
    model = create_model()   # creates the graspnet model
    
    CGD_DATASET = GraspDataset(DATA_PATH, ANN_PATH)
    # images, bboxes = img_preproc.distorted_inputs([data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset=CGD_DATASET,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )


    for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch)
        # print(type(sample_batched[1][0]))
        # print(sample_batched[1][0:8])
        for i in range(0,len(sample_batched[1]),8):
            x, y, tan, h, w = bboxes_to_grasps(sample_batched[1][i:i+8])
            # print(x,y,tan,w,h)

        # observe one batch and stop.
        if i_batch == 0:
            break


    if torch.cuda.is_available():
        print('CUDA is available: {}'.format(torch.cuda.is_available()))
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model.training = True
    
#     x, y, tan, h, w = bboxes_to_grasps(bboxes)
#     x_hat, y_hat, tan_hat, h_hat, w_hat = torch.unbind(model(images), axis=1) # list

#     # tangent of 85 degree is 11 
#     tan_hat_confined = torch.minimum(11., torch.maximum(-11., tan_hat))
#     tan_confined = torch.minimum(11., torch.maximum(-11., tan))

#     # Loss function
#     gamma = tf.constant(10.)
#     loss = torch.sum(torch.pow(x_hat -x, 2) +torch.pow(y_hat -y, 2) + gamma*torch.pow(tan_hat_confined - tan_confined, 2) +torch.pow(h_hat -h, 2) +torch.pow(w_hat -w, 2))
#     train_op = tf.train.AdamOptimizer(epsilon=0.1).minimize(loss)
#     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     sess = tf.Session()
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#     #save/restore model
#     d={}
#     l = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2']
#     for i in l:
#         d[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]
    
#     dg={}
#     lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
#     for i in lg:
#         dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]

#     saver = tf.train.Saver(d)
#     saver_g = tf.train.Saver(dg)
#     #saver.restore(sess, "/root/grasp/grasp-detection/models/imagenet/m2/m2.ckpt")
#     saver_g.restore(sess, FLAGS.model_path)
#     try:
#         count = 0
#         step = 0
#         start_time = time.time()
#         while not coord.should_stop():
#             start_batch = time.time()
#             #train
#             if FLAGS.train_or_validation == 'train':
#                 _, loss_value, x_value, x_model, tan_value, tan_model, h_value, h_model, w_value, w_model = sess.run([train_op, loss, x, x_hat, tan, tan_hat, h, h_hat, w, w_hat])
#                 duration = time.time() - start_batch
#                 if step % 100 == 0:             
#                     print('Step %d | loss = %s\n | x = %s\n | x_hat = %s\n | tan = %s\n | tan_hat = %s\n | h = %s\n | h_hat = %s\n | w = %s\n | w_hat = %s\n | (%.3f sec/batch\n')%(step, loss_value, x_value[:3], x_model[:3], tan_value[:3], tan_model[:3], h_value[:3], h_model[:3], w_value[:3], w_model[:3], duration)
#                 if step % 1000 == 0:
#                     saver_g.save(sess, FLAGS.model_path)
#             else:
#                 bbox_hat = grasp_to_bbox(x_hat, y_hat, tan_hat, h_hat, w_hat)
#                 bbox_value, bbox_model, tan_value, tan_model = sess.run([bboxes, bbox_hat, tan, tan_hat])
#                 bbox_value = np.reshape(bbox_value, -1)
#                 bbox_value = [(bbox_value[0]*0.35,bbox_value[1]*0.47),(bbox_value[2]*0.35,bbox_value[3]*0.47),(bbox_value[4]*0.35,bbox_value[5]*0.47),(bbox_value[6]*0.35,bbox_value[7]*0.47)] 
#                 p1 = Polygon(bbox_value)
#                 p2 = Polygon(bbox_model)
#                 iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area) 
#                 angle_diff = np.abs(np.arctan(tan_model)*180/np.pi -np.arctan(tan_value)*180/np.pi)
#                 duration = time.time() -start_batch
#                 if angle_diff < 30. and iou >= 0.25:
#                     count+=1
#                     print('image: %d | duration = %.2f | count = %d | iou = %.2f | angle_difference = %.2f' %(step, duration, count, iou, angle_diff))
#             step +=1
#     except tf.errors.OutOfRangeError:
#         print('Done training for %d epochs, %d steps, %.1f min.' % (FLAGS.num_epochs, step, (time.time()-start_time)/60))
#     finally:
#         coord.request_stop()

#     coord.join(threads)
#     sess.close()

# def run_epoch(epoch, model, data_loader):
#     model.train()



    
if __name__ == '__main__':
    train()