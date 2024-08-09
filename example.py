import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import tf2onnx
import tensorflow as tf
import paddle
import paddle.nn as pnn
import subprocess
import mindspore
import mindspore.nn as mnn
import numpy as np
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
import sys
sys.path.append('/home/zhonghuihang/pythoncode/YOLO-v5')
def export_model_to_onnx(model, dummy_input, model_path, onnx_path, input_names=['input'], output_names=['output'], opset_version=11, framework='pytorch'):
    """
    导出模型为 ONNX 格式

    参数:
    model (torch.nn.Module 或 tf.keras.Model 或 paddle.nn.Layer 或 mindspore.nn.Cell): 模型实例
    dummy_input (torch.Tensor 或 None): PyTorch 模型的示例输入 (对于 TensorFlow 和 PaddlePaddle 为 None)
    model_path (str): 模型参数文件路径 (.pt, .h5, .pdparams, .ckpt)
    onnx_path (str): 导出 ONNX 模型文件路径 (.onnx)
    input_names (list of str): 输入节点名称
    output_names (list of str): 输出节点名称
    opset_version (int): ONNX opset 版本
    framework (str): 模型框架 ('pytorch', 'tensorflow', 'paddlepaddle', 'mindspore')
    """
    if framework == 'pytorch':
        # 导出模型
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=opset_version,
                          do_constant_folding=True, input_names=input_names, output_names=output_names,
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

 
    # 加载和检查导出的 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型导出成功，路径为:", onnx_path)
    print(onnx.helper.printable_graph(onnx_model.graph))

if __name__ == '__main__':
    # 创建与期望输入相同形状的虚拟数据
    weights_path = '/home/zhonghuihang/pythoncode/onnx/yolov5_models/yolov5s.pt'  # 权重文件路径
    img_size = [640, 640]  # 图像尺寸
    batch_size = 1  # 批处理大小
    onnx_file_name = weights_path.replace('.pt', '.onnx')  # 准备模型的ONNX文件名
    dummy_input = torch.zeros((batch_size, 3, *img_size))
    model = torch.load(weights_path)
    # Export to onnx
    model.model[-1].export = True  # set Detect() layer export=True
    _ = model(dummy_input)  # dry run

    # 使用提供的函数导出模型到 ONNX
    export_model_to_onnx(model, dummy_input, weights_path, onnx_file_name, input_names=['images'], output_names=['output'], framework='pytorch')

    print('ONNX export completed. Model saved to:', onnx_file_name)