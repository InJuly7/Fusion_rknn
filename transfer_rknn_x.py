# 此文件为onnx———>rknn的转换文件

import cv2
import numpy as np
from rknn.api import RKNN
ONNX_MODEL = './model/fusionmodel_288_384.onnx'
RK3588_RKNN_MODEL = './model/fusionmodel_288_384_x_1.rknn'
DATASET = './dataset.txt'
QUANTIZE_ON = False

def img_process(img,vis_flag):
    # visible images RGB channel
    if vis_flag:
        # img_vis  H,W,C    channel:BGR
        img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_vis = img_vis.astype(np.float32)/255.0
        return img_vis
    # infrared images single channel
    else: 
        # img_ir H,W,C channle = 1
        # 转化为维度为1的图像 三个通道的值相同
        ir_img = img[:,:,0:1]
        ir_img = ir_img.astype(np.float32)/255.0
        # ir_img = np.transpose(ir_img,(2,0,1))
        return ir_img

#vis_img
def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """
    R = rgb_image[:,:,0:1]
    G = rgb_image[:,:,1:2]
    B = rgb_image[:,:,2:3]

    Y = 0.299*R + 0.587*G + 0.114*B
    Cr = (R-Y)*0.713 + 0.5
    Cb = (B-Y)*0.564 + 0.5

    Y = np.clip(Y,0.0,1.0)
    Cr = np.clip(Cr,0.0,1.0)
    Cb = np.clip(Cb,0.0,1.0)
    
    return Y,Cb,Cr


def YCrCb2RGB(Y,Cr,Cb):
    ycrcb = np.concatenate([Y,Cr,Cb],axis=2)
    H,W,C = ycrcb.shape
    im_flat = np.reshape(ycrcb,(-1,3)) # [H*W,3]
    mat = np.array([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]) # [3,3]
    bias = np.array([0.0 / 255, -0.5, -0.5]) # [3]
    temp = np.matmul(im_flat+bias, mat)
    temp = np.reshape(temp,(H,W,C))
    out = np.transpose(temp,(2,0,1))
    out = np.clip(out,0.0,1.0)
    return out


def numpy2img(img,is_norm):
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    if is_norm:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # C,H,W -> H,W,C
    img = np.transpose(img, (1, 2, 0))  * 255.0
    return img.astype(np.uint8)


if __name__ == '__main__':

    # Create RKNN object 
    rknn = RKNN(verbose=True)
    # pre-process config
    # print("--->Config rknn para")
    # rknn.config(mean_values=[[0]], std_values=[[255]],target_platform="RK3588")
    # print('rknn config done')

    # Load ONNX model  
    # print("--->Loading onnx model")
    # ret = rknn.load_onnx(model=ONNX_MODEL)
    # if ret != 0:
    #     print('Load model failed!')
    #     exit(ret)
    # print('Load ONNX model done')

    # Build model
    # print('--> hybrid_quantization_step1')
    # ret = rknn.hybrid_quantization_step1(dataset=DATASET, proposal=False)
    # if ret != 0:
    #     print('hybrid_quantization_step1 failed!')
    #     exit(ret)
    # print('done')

    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./fusionmodel_288_384.model',
                                        data_input='./fusionmodel_288_384.data',
                                        model_quantization_cfg='./fusionmodel_288_384.quantization.cfg')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print("--->Export rknn model")
    ret = rknn.export_rknn(RK3588_RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('Export rknn model done')

    # # Init runtime environment
    # print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    # # ret = rknn.init_runtime(target='rk3588',device_id='c18d04671f6d3f16')
    # if ret != 0:
    #     print('Init runtime environment failed!')
    #     exit(ret)
    # print('Init runtime environment done')
    
    # print("--->Build model")
    # ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    # if ret != 0:
    #     print('Build model failed!')
    #     exit(ret)
    # print('Build model done')

    rknn.release()

    
    
    


























    # img_vis = cv2.imread("vis_00858N.png")
    # img_ir = cv2.imread("ir_00858N.png")
    # img_vis = cv2.imread('vis_s_040.jpg')
    # img_ir = cv2.imread('inf_s_040.jpg')
    # vis_img = img_process(img_vis,vis_flag=True)
    # ir_img  = img_process(img_ir,vis_flag=False)
    # 指定计算前向传播过程特定层
    # layerNames = ['output']
    # img_vis = img_vis/np.float32(255)
    # img_ir = img_ir/np.float32(255)
    
    # H,W
    # vis_Y = cv2.cvtColor(vis_img,cv2.COLOR_RGB2GRAY)
    # vis_Y, vis_Cb, vis_Cr = RGB2YCrCb(vis_img)
    # H,W,Cvis_Y = np.expand_dims(vis_Y,axis=2)
    # vis_Y = np.expand_dims(vis_Y,axis=2)
    # N,C,H,W
    # fused_img = rknn.inference(inputs=[vis_Y,ir_img])
    # fused_img = fused_img[0]
    # fused_img = np.squeeze(fused_img,axis=0)
    # fused_img = np.transpose(fused_img,(1,2,0))
    # fused_img = YCrCb2RGB(fused_img, vis_Cr, vis_Cb)
    # fused_img = numpy2img(fused_img,is_norm=True)
    # img_cv2 = cv2.cvtColor(fused_img,cv2.COLOR_RGB2BGR)
    # cv2.imshow("Fused_Image",fused_img)
    # cv2.waitKey(0)

    # # N,C,H,W
    # fused_img = fused_img[0]
    # f
    # fused_img = np.transpose(fused_img,(1,2,0))
    # fused_img = YCrCb2RGB(fused_img, vis_Cr, vis_Cb)
    # # C,H,W
    # fused_img = numpy2img(fused_img,is_norm=True)
    # img_cv2 = cv2.cvtColor(fused_img,cv2.COLOR_RGB2BGR)
    


    



