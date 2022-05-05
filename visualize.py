from mmhuman3d.core.visualization import visualize_smpl_pose

body_model_config = dict(
    type='smpl', model_path='data/body_models/')

import numpy as np
import torch

file_p = ''
# CVFile(file_p).show(iter_times=4)
data = np.load(file_p, allow_pickle=True)
# data = CVFile(file_p).data
print(data.shape)

# print([data[:10][:72]])
# pose = data[0][:72]
# betas = data[0][72:]
# print(pose, betas)
# print(data[0])

visualize_smpl_pose(
    poses=torch.Tensor(data[:10,:72]),
    # poses=torch.tensor([data[1][:72]], dtype=torch.double),
    output_path='./smpl.mp4',
    resolution=(1024, 1024),
    body_model_config=body_model_config, )
