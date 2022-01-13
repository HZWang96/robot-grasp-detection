import argparse
import logging
import numpy as np
import torch.utils.data

from models.common import post_process_output
from dataset_processing import evaluation, grasp
from data import get_dataset
from opts import opts

logging.basicConfig(level=logging.INFO)


# def parse_args():
#     parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

#     # Network
#     parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

#     # Dataset & Data & Training
#     parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
#     parser.add_argument('--dataset-path', type=str, help='Path to dataset')
#     parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
#     parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
#     parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
#     parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
#     parser.add_argument('--ds-rotate', type=float, default=0.0,
#                         help='Shift the start point of the dataset to use a different test/train split')
#     parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

#     parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
#     parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
#     parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
#     parser.add_argument('--vis', action='store_true', help='Visualise the network output')

#     args = parser.parse_args()

#     if args.jacquard_output and args.dataset != 'jacquard':
#         raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
#     if args.jacquard_output and args.augment:
#         raise ValueError('--jacquard-output can not be used with data augmentation.')

#     return args


if __name__ == '__main__':
    opt = opts().init()

    # Load Network
    net = torch.load(opt.trained_network)
    device = torch.device("cuda:"+str(opt.which_gpu) if torch.cuda.is_available() else "cpu")

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(opt.dataset.title()))
    Dataset = get_dataset(opt.dataset)
    test_dataset = Dataset(opt.dataset_path, start=opt.split, end=1.0, ds_rotate=opt.ds_rotate,
                           random_rotate=opt.augment, random_zoom=opt.augment,
                           include_depth=opt.eval_use_depth, include_rgb=opt.eval_use_rgb)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn_eval,  
        num_workers=opt.num_workers
    )
    logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    if opt.jacquard_output:
        jo_fn = opt.trained_network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    with torch.no_grad():
        # for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
        #     logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
        #     xc = x.to(device)
        #     yc = [yi.to(device) for yi in y]
        #     lossd = net.compute_loss(xc, yc)

        #     q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
        #                                                 lossd['pred']['sin'], lossd['pred']['width'])

        for idx, (rgb_img, grasp_labels, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            test_img = rgb_img.to(device)
            gt = [torch.from_numpy(grasp_label).float().to(device) for grasp_label in grasp_labels]
            lossd = net.compute_loss(test_img, gt)

            Test_pred = lossd['pred']
            test_preds = list(Test_pred.values())
            
            test_pred_value = [test_pred.float().cpu() for test_pred in test_preds]
            test_pred_value[0] = test_pred_value[0] * 224
            test_pred_value[1] = test_pred_value[1] * 224
            test_pred_value[3] = test_pred_value[3] * 100
            test_pred_value[4] = test_pred_value[4] * 80

            for i in range(np.shape(gt[0])[0]):
                gt[0][i][0] = gt[0][i][0] * 224
                gt[0][i][1] = gt[0][i][1] * 224
                gt[0][i][3] = gt[0][i][3] * 100
                gt[0][i][4] = gt[0][i][4] * 80

            if opt.iou_eval:
                # s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                #                                    no_grasps=opt.n_grasps,
                #                                    grasp_width=width_img,
                #                                    )
                s = evaluation.calculate_iou_match(test_pred_value, gt, no_grasps=opt.n_grasps)

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            if opt.jacquard_output:
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            if opt.eval_vis:
                # print('Image Number:', didx[0])
                evaluation.plot_output(test_data.dataset.get_rgb(didx[0], rot[0], zoom[0], normalise=False),
                                       test_data.dataset.get_depth(didx[0], rot[0], zoom[0]), test_pred_value, no_grasps=opt.n_grasps)

    if opt.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))

    if opt.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))