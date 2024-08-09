import onnxruntime as ort
import numpy as np

def load_model(onnx_model_path):
    session = ort.InferenceSession(onnx_model_path)
    return session

def prepare_input(input_shape):
    return np.random.rand(*input_shape).astype(np.float32)

def infer(session, input_data):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_data})
    return outputs

if __name__ == '__main__':
    onnx_model_path = '/home/zhonghuihang/pythoncode/onnx/yolov5_models/yolov5s.onnx'
    input_shape = (1, 3, 640, 640)
    session = load_model(onnx_model_path)
    input_data = prepare_input(input_shape)
    outputs = infer(session, input_data)

    # 添加类型检查和简化打印
    print("Output type:", type(outputs))
    print("Output shape and dtype if numpy array:", [(out.shape, out.dtype) for out in outputs if isinstance(out, np.ndarray)])
    try:
        print("Output content:", outputs)
    except Exception as e:
        print("Error printing the outputs:", str(e))