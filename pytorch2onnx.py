import argparse
import torch
import torchvision

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='PyTorch Classification Export Onnx Model',
                                     add_help=add_help)
    parser.add_argument('--model',
                        default='inception_v3',
                        help='model')
    parser.add_argument('--opset',
                        default=13,
                        type=int,
                        help='opset')
    parser.add_argument('--batchsize',
                        default=1,
                        type=int,
                        help='batchsize')
    parser.add_argument("--dynamic-bs",
                        dest="dynamic_bs",
                        help="Set dynamic bs",
                        action="store_true",)

    return parser

def main(args):
    if args.model == 'inception_v3':
        dummy_input = torch.randn(args.batchsize, 3, 299, 299, device='cpu')
    else:
        dummy_input = torch.randn(args.batchsize, 3, 224, 224, device='cpu')

    model = torchvision.models.__dict__[args.model](pretrained=True).cpu()
    input_names = ["input"]
    output_names = ["output"]
    onnx_file_name = "{}-op{}-fp32.onnx".format(args.model,str(args.opset))
    if args.dynamic_bs:
        torch.onnx.export(model,
                        dummy_input,
                        onnx_file_name,
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names,
                        opset_version=args.opset,
                        dynamic_axes={'input':{0:'bs'}, 'output':{0:'bs'}}
                        )
    else:
        torch.onnx.export(model,
                        dummy_input,
                        onnx_file_name,
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names,
                        opset_version=args.opset,
                        )

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)