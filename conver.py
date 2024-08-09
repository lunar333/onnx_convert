import torch
import onnx
import json
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_model_class(model_code_path):
    # 动态加载模型类定义
    model_dir = os.path.dirname(model_code_path)
    model_file = os.path.basename(model_code_path)
    model_name = os.path.splitext(model_file)[0]

    if model_dir not in sys.path:
        sys.path.append(model_dir)

    # 使用 __import__ 动态导入模块
    module = __import__(model_name)
    globals().update(vars(module))

def export_model_to_onnx(config_path):
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_path = config['model_path']
    onnx_path = config['onnx_path']
    input_size = config['input_size']
    batch_size = config['batch_size']
    input_names = config['input_names']
    output_names = config['output_names']
    model_code_path = config['model_code_path']
    
    # 加载模型类定义
    load_model_class(model_code_path)
    
    # 设定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建虚拟输入
    dummy_input = torch.zeros((batch_size, 3, *input_size)).to(device)

    # 加载模型
    model = torch.load(model_path)
    
    model.to(device)
    model.eval()

    # 导出模型
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=input_names, output_names=output_names,
                      dynamic_axes={input_names[0]: {0: 'batch_size'}, output_names[0]: {0: 'batch_size'}})

    print(f'ONNX export completed. Model saved to: {onnx_path}')

if __name__ == '__main__':
    config_file_path = '/home/zhonghuihang/pythoncode/onnx/config.json'  # JSON 配置文件路径
    export_model_to_onnx(config_file_path)