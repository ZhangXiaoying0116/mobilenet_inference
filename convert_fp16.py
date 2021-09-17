'''
Author liguo.wang@enflame-tech.com
Convert the given float32 onnx model to mixed precision float16 onnx
'''
import argparse
import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import load_model, save_model

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Onnx Model FP32 to FP16',
                                     add_help=add_help)
    parser.add_argument('--model',
                        default='inception_v3',
                        help='model')
    parser.add_argument('--opset',
                        default=13,
                        type=int,
                        help='opset')
    return parser

def main(args):
    model_name_fp32 = "{}-op{}-fp32.onnx".format(args.model,str(args.opset))
    model_name_fp16 = "{}-op{}-fp16.onnx".format(args.model,str(args.opset))
    onnx_model = load_model(model_name_fp32)
    onnx.checker.check_model(onnx_model)
    print("FP32 ==> Passed")
    new_onnx_model = convert_float_to_float16(onnx_model)
    onnx.checker.check_model(new_onnx_model)
    print("FP16 ==> Passed")
    save_model(new_onnx_model, model_name_fp16)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
