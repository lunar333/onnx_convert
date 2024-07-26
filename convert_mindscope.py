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
        # 加载模型参数
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 切换到评估模式

        # 导出模型
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=opset_version,
                          do_constant_folding=True, input_names=input_names, output_names=output_names,
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    elif framework == 'tensorflow':
        # 加载 TensorFlow 模型
        model = tf.keras.models.load_model(model_path)

        # 定义输入签名
        spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),)

        # 导出模型
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset_version, output_path=onnx_path)

    elif framework == 'paddlepaddle':
        # 使用命令行工具 paddle2onnx 进行转换
        subprocess.run([
            "paddle2onnx",
            "--model_dir", model_path,
            "--model_filename", "model.pdmodel",
            "--params_filename", "model.pdiparams",
            "--save_file", onnx_path,
            "--opset_version", str(opset_version)
        ], check=True)

    elif framework == 'mindspore':
        # 加载 MindSpore 模型
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        load_param_into_net(model, load_checkpoint(model_path))
        input_tensor = Tensor(dummy_input)

        # 导出模型
        export(model, input_tensor, file_name=onnx_path.replace('.onnx', ''), file_format='ONNX')

    else:
        raise ValueError("Unsupported framework. Please choose 'pytorch', 'tensorflow', 'paddlepaddle', or 'mindspore'.")

    # 加载和检查导出的 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型导出成功，路径为:", onnx_path)
    print(onnx.helper.printable_graph(onnx_model.graph))

# MindSpore 模型定义
class SimpleMindSporeModel(mnn.Cell):
    def __init__(self):
        super(SimpleMindSporeModel, self).__init__()
        self.conv1 = mnn.Conv2d(1, 10, 5)
        self.pool = mnn.MaxPool2d(2, 2)
        self.flatten = mnn.Flatten()
        self.fc1 = mnn.Dense(10 * 12 * 12, 50)
        self.fc2 = mnn.Dense(50, 10)

    def construct(self, x):
        x = self.pool(mindspore.ops.ReLU()(self.conv1(x)))
        x = self.flatten(x)
        x = mindspore.ops.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化 MindSpore 模型
mindspore_model = SimpleMindSporeModel()

# 创建一个示例输入
dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)

# 确保输入形状正确
x = Tensor(dummy_input)
x = mindspore_model.conv1(x)
x = mindspore_model.pool(mindspore.ops.ReLU()(x))
flattened_size = x.shape[1] * x.shape[2] * x.shape[3]
mindspore_model.fc1 = mnn.Dense(flattened_size, 50)

# 保存模型参数
mindspore_checkpoint_path = "/home/zhonghuihang/pythoncode/onnx/mindspore_model.ckpt"
mindspore.save_checkpoint(mindspore_model, mindspore_checkpoint_path)

# 模型文件路径和 ONNX 文件路径
mindspore_onnx_path = "mindspore_model.onnx"  # ONNX 模型输出路径

# 导出 MindSpore 模型为 ONNX 格式
export_model_to_onnx(mindspore_model, dummy_input, mindspore_checkpoint_path, mindspore_onnx_path, framework='mindspore')