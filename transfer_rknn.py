# 此文件为onnx———>rknn的转换文件
import cv2
import numpy as np
from rknn.api import RKNN
ONNX_MODEL = './model/fusionmodel_480_80.onnx'
RK3588_RKNN_MODEL = './model/fusionmodel_480_80.rknn'
DATASET = './dataset.txt'
QUANTIZE_ON = False


if __name__ == '__main__':
    # Create RKNN object 
    rknn = RKNN(verbose=True)
    # pre-process config
    print("--->Config rknn para")
    rknn.config(mean_values=[[0],[0]],std_values=[[1],[1]],target_platform='rk3588')
    print('rknn config done')

    # Load ONNX model  
    print("--->Loading onnx model")
    # inputs
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('Load ONNX model done')

    # Build model
    print("--->Build model")
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('Build model done')

    # Export RKNN model
    print("--->Export rknn model")
    ret = rknn.export_rknn(RK3588_RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('Export rknn model done')

    