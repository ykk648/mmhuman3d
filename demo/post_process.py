from cv2box import CVFile, CVImage
import numpy as np
import cv2

pkl_p = 'vis_results/output_result/pare_jianghao_cam2_nosmooth.npz'
data = CVFile(pkl_p).data
# print(data['smpl'][()]['body_pose'].shape)
# print(data['smpl'][()]['global_orient'].shape)
# smpl_pose = []
# for i in range(len(data['smpl'][()]['body_pose'])):
#     smpl_pose_i = np.concatenate(
#         [data['smpl'][()]['body_pose'][i], data['smpl'][()]['global_orient'][i].reshape(-1, 3)], axis=0)
#     smpl_pose.append(smpl_pose_i)
# print(np.array(smpl_pose).shape)
# print(data['smpl'][()]['betas'].shape)
# print(data['pred_cams'][()].shape)
print(data['keypoints_3d'][()].shape)
print(data['bboxes_xyxy'][()].shape)
print(data['bboxes_xyxy'][()][0])
# print(data['keypoints_3d'][()][0])

bbox_height = data['bboxes_xyxy'][()][0][3] - data['bboxes_xyxy'][()][0][1]
bbox_width = data['bboxes_xyxy'][()][0][2] - data['bboxes_xyxy'][()][0][0]
center_x = data['bboxes_xyxy'][()][0][0] + bbox_width//2
center_y = data['bboxes_xyxy'][()][0][1] + bbox_height//2
kp3d = data['keypoints_3d'][()][0]
print(kp3d)
print(kp3d)

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
visualize_kp3d(kp3d=data['keypoints_3d'][()], data_source='h36m', output_path='some_video.mp4')


'''
(972, 23, 3)
(972, 3)
(972, 24, 3)
(972, 10)
(972, 3)
'''