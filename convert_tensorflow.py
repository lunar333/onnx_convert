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

    # 加载和检查导出的 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型导出成功，路径为:", onnx_path)
    print(onnx.helper.printable_graph(onnx_model.graph))

# 创建一个简单的 TensorFlow 模型
tensorflow_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译并保存 TensorFlow 模型
tensorflow_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tensorflow_model.save("tensorflow_model.h5")

# 模型文件路径和 ONNX 文件路径
tensorflow_model_path = "tensorflow_model.h5"  # TensorFlow 模型路径
tensorflow_onnx_path = "tensorflow_model.onnx"  # ONNX 模型输出路径

# 导出 TensorFlow 模型为 ONNX 格式
export_model_to_onnx(None, None, tensorflow_model_path, tensorflow_onnx_path, framework='tensorflow')