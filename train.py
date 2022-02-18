''' 
Training a network on cornell grasping dataset for detecting grasping positions.
'''
import sys
import os.path
import glob
import torch
import torch.utils.data
import numpy as np
import time
from opts import opts
import cv2
import tensorboardX
import datetime
import os
import logging
from data import get_dataset
import torch.optim as optim
from torchsummary import summary
from models.common import post_process_output
from dataset_processing import evaluation
from models.ResNet50 import get_grasp_resnet
from models.AlexNet import get_grasp_alexnet
from visualisation.gridshow import gridshow

logging.basicConfig(level=logging.INFO)


def validate(epoch, net, device, val_data):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {'x_loss': 0,
                   'y_loss': 0,
                   'theta_loss': 0,
                   'length_loss': 0,
                   'width_loss': 0
        }
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        # while batch_idx < batches_per_epoch:
        for rgb_img, grasp_labels in val_data:
            batch_idx += 1
            # if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
            #     break

            val_img = rgb_img.to(device)
            gt = [torch.from_numpy(grasp_label).float().to(device) for grasp_label in grasp_labels]
            # print('gt:')
            # print(gt)
            # print(np.shape(gt[0]))

            lossd = net.compute_loss(val_img, gt)

            loss = lossd['loss']

            # logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()/ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()/ld
            
            # print(type(gt[0]))
            Val_pred = lossd['pred']
            # print('Val_pred:')
            # print(Val_pred)
            val_preds = list(Val_pred.values())
            # print('val_preds:')
            # print(val_preds)
            # val_preds = np.array(val_preds)
            val_pred_value = [val_pred.float().cpu() for val_pred in val_preds]
            val_pred_value[0] = val_pred_value[0] * 224
            val_pred_value[1] = val_pred_value[1] * 224
            val_pred_value[3] = val_pred_value[3] * 100
            val_pred_value[4] = val_pred_value[4] * 80

            for i in range(np.shape(gt[0])[0]):
                gt[0][i][0] = gt[0][i][0] * 224
                gt[0][i][1] = gt[0][i][1] * 224
                gt[0][i][3] = gt[0][i][3] * 100
                gt[0][i][4] = gt[0][i][4] * 80
            # val_pred_value = torch.stack(val_pred_value, dim=1)
            # print('val_pred_values:')
            # print(val_pred_value)
            # print(type(val_pred_value))
                    
            # for x, y, didx, rot, zoom_factor in val_data:
            #     batch_idx += 1
            #     if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
            #         break

                # xc = x.to(device)
                # yc = [yy.to(device) for yy in y]
                # lossd = net.compute_loss(xc, yc)

                # loss = lossd['loss']

                # results['loss'] += loss.item()/ld
                # for ln, l in lossd['losses'].items():
                #     if ln not in results['losses']:
                #         results['losses'][ln] = 0
                #     results['losses'][ln] += l.item()/ld
            # val_pred, gt = post_process_output(lossd['pred']['x'], lossd['pred']['y'],
            #                                                 lossd['pred']['theta'], lossd['pred']['length'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(val_pred_value, gt, no_grasps=1)

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

        # results['loss'] /= batch_idx
        # for l in results['losses']:
        #     results['losses'][l] /= batch_idx

    return results


# DATA_PATH = '../datasets/cornell_grasping_dataset/data-1'
# ANN_PATH = '../datasets/cornell_grasping_dataset/annotations/train.json'


# def train():
#     opt = opts()
#     # logger = Logger(opt)

#     print('Creating model...')
#     model = create_model()   # creates the graspnet model
    
#     CGD_DATASET = GraspDataset(DATA_PATH, ANN_PATH)
#     # images, bboxes = img_preproc.distorted_inputs([data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size)

#     train_loader = torch.utils.data.DataLoader(
#         dataset=CGD_DATASET,
#         batch_size=1,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=False
#     )


#     for i_batch, sample_batched in enumerate(train_loader):
#         print(i_batch)
#         # print(type(sample_batched[1][0]))
#         # print(sample_batched[1][0:8])
#         for i in range(0,len(sample_batched[1]),8):
#             x, y, tan, h, w = bboxes_to_grasps(sample_batched[1][i:i+8])
#             # print(x,y,tan,w,h)

#         # observe one batch and stop.
#         if i_batch == 0:
#             break


#     if torch.cuda.is_available():
#         print('CUDA is available: {}'.format(torch.cuda.is_available()))
#         model = torch.nn.DataParallel(model).cuda()
#     else:
#         model = torch.nn.DataParallel(model)

#     model.training = True


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    opt = opts().init()
    results = {
        'loss': 0,
        'losses': {'x_loss': 0,
                   'y_loss': 0,
                   'theta_loss': 0,
                   'length_loss': 0,
                   'width_loss': 0
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for rgb_img, y in train_data:                   #rgb_img: Input RGB images, y: Input ground truth grasp labels.
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            # print(rgb_img.shape)
            train_img = rgb_img.to(device)              #每次取batch_size张的图片和对应数量的bbx
            # print(train_img)

            # for grasp_label in grasp_labels:
            #     print('max value is:', np.max(grasp_label, axis=0))

            gt = [torch.from_numpy(yy).float().to(device) for yy in y]
            # print('gt is:', gt)

            # for i in range(opt.batch_size):
            lossd = net.compute_loss(train_img, gt)

            loss = lossd['loss']

            # if batch_idx % 100 == 0:
            logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():       # ln是dic里面每一个项目的名字，l是每个项目对应的值
                # print(ln)
                # print(l)
                if ln not in results['losses']:         
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()
                # results['losses'][ln] += l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, rgb_img.shape[0])
                for idx in range(n_img):
                    imgs.extend([rgb_img[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        rgb_img[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
                gridshow('Display', imgs,
                        [(train_img.min().item(), train_img.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                        [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


        # batch_idx += 1
        # # if batch_idx >= batches_per_epoch:
        # #     break
        # if batch_idx == 1:
        #     break
        # # break
            
            # xc = x.to# from logger import Logger(device)
            # yc = [yy.to(device) for yy in y]
            # for yy in y:
                # yy.to(device)
            # x_hat, y_hat, tan_hat, h_hat, w_hat = torch.unbind(net(rgb_img), axis=1) # list

            # tangent of 85 degree is 11 




            #有用的代码
            # tan_hat_confined = torch.minimum(11., torch.maximum(-11., tan_hat))
            # tan_confined = torch.minimum(11., torch.maximum(-11., tan))
            # # loss function
            # # lossd = net.compute_loss(xc, yc)
            # gamma = torch.tensor(10.)
            # lossd = torch.sum(torch.pow(x_hat -x, 2) +torch.pow(y_hat -y, 2) + gamma*torch.pow(tan_hat_confined - tan_confined, 2) +torch.pow(h_hat -h, 2) +torch.pow(w_hat -w, 2))

            # loss = lossd['loss']

            # if batch_idx % 100 == 0:# from .grasp import GraspRectangle
            #     logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            # results['loss'] += loss.item()
            # for ln, l in lossd['losses'].items():
            #     if ln not in results['losses']:
            #         results['losses'][ln] = 0
            #     results['losses'][ln] += l.item()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()






            # # Display the images
            # if vis:
            #     imgs = []
            #     n_img = min(4, x.shape[0])
            #     for idx in range(n_img):
            #         imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
            #             x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
            #     gridshow('Display', imgs,
            #              [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
            #              [cv2.COLORMAP_BONE] * 10 * n_img, 10)
            #     cv2.waitKey(2)


    #有用的代码
    # results['loss'] /= batch_idx
    # for l in results['losses']:
    #     results['losses'][l] /= batch_idx

    # return results


def run():
    opt = opts().init()

    # Vis window
    if opt.vis:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(opt.description.split()))

    save_folder = os.path.join(opt.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(os.path.join(opt.logdir, net_desc))

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(opt.dataset.title()))
    Dataset = get_dataset(opt.dataset)

    train_dataset = Dataset(opt.dataset_path, start=0.0, end=opt.split, ds_rotate=opt.ds_rotate,
                            random_rotate=False, random_zoom=False,
                            include_depth=opt.use_depth, include_rgb=opt.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn_train,
        num_workers=opt.num_workers
    )
    val_dataset = Dataset(opt.dataset_path, start=opt.split, end=1.0, ds_rotate=opt.ds_rotate,
                          random_rotate=False, random_zoom=False,
                          include_depth=opt.use_depth, include_rgb=opt.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=val_dataset.collate_fn_train,
        num_workers=opt.num_workers
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1*opt.use_depth + 3*opt.use_rgb             #  1*args.use_depth + 3*args.use_rgb
    # ggcnn = get_network(args.network)
    net = get_grasp_resnet()                       # choose ResNet model!
    # net = get_grasp_alexnet()                        # choose AlexNet model (Redmon version)!
    device = torch.device("cuda:"+str(opt.which_gpu) if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr, weight_decay=1e-4, momentum=0.9)
    logging.info('Done')

    # Print model architecture.
    summary(net, (input_channels, 224, 224))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 224, 224))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    for epoch in range(opt.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))

        # Warming up the learning rate
        if epoch > opt.warm_up:
            lr1 = opt.lr * 0.1
            print('Drop LR to:', lr1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr1
            print('LR now in optimizer is:', param_group['lr'])
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr
            print('LR now in optimizer is:', param_group['lr'])
            
        train_results = train(epoch, net, device, train_data, optimizer, opt.batches_per_epoch, vis=opt.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(epoch, net, device, val_data)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()



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



    
# if __name__ == '__main__':
#     train()
