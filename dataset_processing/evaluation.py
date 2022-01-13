import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset_processing.grasp import GraspRectangles, detect_grasps, GraspRectangle, Grasps2GraspRectangles


def plot_output(rgb_img, depth_img, test_pred, no_grasps=1):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    # gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)
    gs = Grasps2GraspRectangles(test_pred)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    # for g in gs:
    #     g.plot(ax)
    gs.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    # for g in gs:
    #     g.plot(ax)
    gs.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')

    # ax = fig.add_subplot(2, 2, 3)
    # plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    # ax.set_title('Q')
    # ax.axis('off')
    # plt.colorbar(plot)

    # ax = fig.add_subplot(2, 2, 4)
    # plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    # ax.set_title('Angle')
    # ax.axis('off')
    # plt.colorbar(plot)
    
    plt.show()


def calculate_iou_match(val_pred, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)       #center, angle, 
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    # if not isinstance(ground_truth_bbs, GraspRectangles):
    #     gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)                                          #读入bbx的四个角的坐标
    # else:
    #     gt_bbs = ground_truth_bbs
    # gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)

    gs = Grasps2GraspRectangles(val_pred)
    # print('gs:')
    # print(gs)
    # print(type(gs))
    # print(type(val_pred))
    gt_bbs = [ground_truth_bb.cpu() for ground_truth_bb in ground_truth_bbs]
    # print(gt_bbs)
    # print(type(gt_bbs))
    # print(gt_bbs[0])
    # print(gt_bbs[0][0])

    # gt_bbs = np.array(gt_bbs)
    # gs = np.array(gs)

    # gt_bbs = torch.from_numpy(gt_bbs)
    # gs = torch.from_numpy(gs)

    if gs.max_iou(gt_bbs) > 0.25:
        return True
    else:
        return False