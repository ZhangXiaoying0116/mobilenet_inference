'''
Author liguo.wang@enflame-tech.cn
Classification model fp32/fp16 inference
'''
import os
import time
import numbers
import argparse
import torch
from PIL import Image
import numpy as np
import onnxruntime as ort

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Onnx Model Fp32/Fp16 inference',
                                     add_help=add_help)
    parser.add_argument('--input_height',
                        default=224,
                        type=int,
                        help='model input image height')
    parser.add_argument('--input_width',
                        default=224,
                        type=int,
                        help='model input image width')
    parser.add_argument('--preprocess',
                        default='default',
                        type=str,
                        help='preprocessing method: default, inception, caffe')
    parser.add_argument('--offset',
                        default=0,
                        type=int,
                        help='label offset: set to 1 if your model classify 1001 categories')
    parser.add_argument('--model',
                        default='inception_v2.onnx',
                        help='onnx path')
    parser.add_argument("--test-fp16",
                        dest="test_fp16",
                        help="Test fp16 model",
                        action="store_true",)
    return parser

def img_resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size, interpolation)

def img_crop(img, top, left, height, width):
    return img.crop((left, top, left + width, top + height))

def img_center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img_crop(img, crop_top, crop_left, crop_height, crop_width)

def img_crop_fraction(img, frac):
    image_width, image_height = img.size
    crop_height = int(image_height * frac)
    crop_width = int(image_width * frac)
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img_crop(img, crop_top, crop_left, crop_height, crop_width)

def preprocess(img_file, args):
    image = Image.open(img_file).convert("RGB")
    input_size = (args.input_width, args.input_height)
    max_size = max(args.input_width, args.input_height)
    if args.preprocess == 'unused':
        # original inception use this, but less accuracy
        image = img_crop_fraction(image, 0.875)
        image = img_resize(image, input_size)
    elif args.preprocess == 'caffe':
        image = img_resize(image, max_size)
        image = img_center_crop(image, input_size)
    else:
        image = img_resize(image, 256 if max_size <= 256 else 342)
        image = img_center_crop(image, input_size)

    if args.preprocess == 'default':
        image_data = np.array(image, dtype='float32').transpose(2, 0, 1)
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_image_data = np.zeros(image_data.shape).astype('float32')
        for i in range(image_data.shape[0]):
            norm_image_data[i, :, :] = (image_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
        norm_image_data = norm_image_data.reshape(1, 3, args.input_height, args.input_width).astype('float32')
    elif args.preprocess == 'inception':
        image_data = np.array(image,dtype='float32')
        norm_image_data = (image_data / 255 - 0.5) * 2
        norm_image_data = norm_image_data.reshape(1, args.input_height, args.input_width, 3).astype('float32')
    elif args.preprocess == 'caffe':
        image_data = np.array(image, dtype='float32').transpose(2, 0, 1)
        mean_vec = np.array([123.68, 116.779, 103.939])
        norm_image_data = np.zeros(image_data.shape).astype('float32')
        for i in range(image_data.shape[0]):
            norm_image_data[i, :, :] = image_data[i, :, :] - mean_vec[i]
        norm_image_data = norm_image_data[(2, 1, 0), :, :] # to bgr
        norm_image_data = norm_image_data.reshape(1, 3, args.input_height, args.input_width).astype('float32')
    return norm_image_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    #return np.array(result).reshape(-1).tolist()
    return softmax(np.array(result)).tolist()

def main(args):
    acc = 0
    acc5 = 0
    image_count = 0
    start = time.time()
    onnx_model = args.model

    session = ort.InferenceSession(onnx_model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print('Input Name:', input_name)
    print('Output Name:', output_name)

    with open("./val.txt", "r") as file_path:
        for idx in range(0, 50000):
            image_label = file_path.readline()
            img_file, label = image_label.split(' ', -1)
            img_file = os.path.join('data/evaluate/', img_file)
            label = np.int32(int(label)) + args.offset
            image_count = image_count + 1
            input_data = preprocess(img_file, args)
            if args.test_fp16:
                input_data = input_data.astype('float16')
            raw_result = session.run([], {input_name: input_data})
            res = postprocess(raw_result)
            res = torch.Tensor(res)
            _, indices = res.topk(5, largest=True, sorted=True)
            pred = np.argmax(res.tolist())
            acc += 1 if pred == label else 0
            if label in indices.tolist():
                acc5 += 1
            print('%d/50000------>label:%s-pred:%s'%(idx, label, pred))
        acc = acc / image_count
        acc5 = acc5 / image_count
        print("acc1 = " + str(acc))
        print("acc5 = " + str(acc5))
    end_time = time.time() - start
    print('fps = %f'% (image_count / end_time))

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
