import cv2
import numpy as np
import os

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


def post_process_func(fused_img,vis_Cr,vis_Cb):
    
    # N,C,H,W
    fused_img = np.squeeze(fused_img,axis=0)
    fused_img = np.transpose(fused_img,(1,2,0))
    fused_img = YCrCb2RGB(fused_img, vis_Cr, vis_Cb)
    # C,H,W
    fused_img = numpy2img(fused_img,is_norm=True)
    fused_img = cv2.cvtColor(fused_img,cv2.COLOR_RGB2BGR)
    return fused_img

