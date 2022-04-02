import torch
from mmhuman3d.apis import inference_image_based_model, init_model
# from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.models.architectures.mesh_estimator import (
    ImageBodyModelEstimator,
    # VideoBodyModelEstimator,
)


# from mmhuman3d.models.builder import build_body_model


def image_body_mesh_estimator():
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

    model_, _ = init_model(
        "configs/spin/resnet50_spin_pw3d.py",
        # "work_dir/spin_b128_hmr_eft-spinmosh-agora/latest.pth",
        "data/pretrained/spin_pretrain_spinself.pth",
        device='cuda')

    # model_.training = False
    input_ = torch.randn((1, 3, 224, 224)).cuda()
    # print(model_.forward(input_))
    # print('init done')
    # # print(model_)
    #
    onnx_model_name = "spin.onnx"
    input_names = ["input"]
    output_names = ["output"]
    opset_version = 12
    dynamic_axes = None
    # dynamic_axes = {'actual_input_1': [0, 2, 3], 'output1': [0, 1]}
    torch.onnx.export(model_, input_, onnx_model_name, verbose=True, opset_version=opset_version,
                      input_names=input_names,
                      output_names=output_names, dynamic_axes=dynamic_axes)
    # raise 'convert done !'


if __name__ == '__main__':
    image_body_mesh_estimator()
