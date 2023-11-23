import cv2
import numpy as np

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
        img_ir = img[:,:,0:1]
        img_ir = img_ir.astype(np.float32)/255.0
        return img_ir

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
    
    return Y,Cr,Cb


def pre_process_func(img_vis,img_ir):
    img_vis = img_process(img_vis,vis_flag=True)
    img_ir  = img_process(img_ir,vis_flag=False)
    # 指定计算前向传播过程特定层
    # layerNames = ['output']
    vis_Y, vis_Cr, vis_Cb = RGB2YCrCb(img_vis)

    return vis_Y,vis_Cr,vis_Cb,img_ir
