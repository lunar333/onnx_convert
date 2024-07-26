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

def export_model_to_onnx(model, dummy_input, model_path, onnx_path, input_names=['input'], output_names=['output'], opset_version=11, framework='pytorch'):
    """
    导出模型为 ONNX 格式

    参数:
    model (torch.nn.Module 或 tf.keras.Model 或 paddle.nn.Layer): 模型实例
    dummy_input (torch.Tensor 或 None): PyTorch 模型的示例输入 (对于 TensorFlow 和 PaddlePaddle 为 None)
    model_path (str): 模型参数文件路径 (.pt, .h5, .pdparams)
    onnx_path (str): 导出 ONNX 模型文件路径 (.onnx)
    input_names (list of str): 输入节点名称
    output_names (list of str): 输出节点名称
    opset_version (int): ONNX opset 版本
    framework (str): 模型框架 ('pytorch', 'tensorflow', 'paddlepaddle')
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

    else:
        raise ValueError("Unsupported framework. Please choose 'pytorch', 'tensorflow', or 'paddlepaddle'.")

    # 加载和检查导出的 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型导出成功，路径为:", onnx_path)
    print(onnx.helper.printable_graph(onnx_model.graph))

# 确保保存模型的目录存在
output_dir = "/home/zhonghuihang/pythoncode/onnx/paddle"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# PaddlePaddle 模型定义
class SimplePaddleModel(pnn.Layer):
    def __init__(self):
        super(SimplePaddleModel, self).__init__()
        self.conv1 = pnn.Conv2D(1, 10, 5)
        self.pool = pnn.MaxPool2D(2, 2)
        self.fc1 = pnn.Linear(10 * 12 * 12, 50)
        self.fc2 = pnn.Linear(50, 10)

    @paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, 1, 28, 28], dtype='float32', name="input")])
    def forward(self, x):
        x = self.pool(paddle.nn.functional.relu(self.conv1(x)))
        x = paddle.flatten(x, start_axis=1)
        x = paddle.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
paddle_model = SimplePaddleModel()

# 保存模型
paddle.jit.save(paddle_model, os.path.join(output_dir, "model"), input_spec=[paddle.static.InputSpec(shape=[None, 1, 28, 28], dtype='float32', name="input")])

# 模型文件路径和 ONNX 文件路径
paddle_model_path = "/home/zhonghuihang/pythoncode/onnx/paddle"  # PaddlePaddle 模型路径
paddle_onnx_path = "paddle_model.onnx"  # ONNX 模型输出路径

# 导出 PaddlePaddle 模型为 ONNX 格式
export_model_to_onnx(None, None, paddle_model_path, paddle_onnx_path, framework='paddlepaddle')