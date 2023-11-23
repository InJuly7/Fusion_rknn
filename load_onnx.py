import cv2
import numpy as np
import os

def img_process(img_path,vis_flag):
    # visible images RGB channel
    if vis_flag:
        # img_vis  H,W,C    channel:BGR
        img_vis = cv2.imread(img_path)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        img_vis = np.array(img_vis)
        img_vis = img_vis.astype(np.float32)/255.0
        return img_vis
    # infrared images single channel
    else: 
        # img_ir H,W,C channle = 1
        img_ir = cv2.imread(img_path)
        # 转化为维度为1的图像 三个通道的值相同
        img_ir = img_ir[:,:,0:1]
        img_ir= np.array(img_ir)
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
    
    return Y,Cb,Cr


   
def YCbCr2RGB(Y,Cr,Cb):
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


if __name__ == "__main__":

    ir_img_path = "ir_00858N.png"
    vis_img_path = "vis_00858N.png"
    # ir_img_path = "./SeAFusion_main/test_imgs/ir/17.png"
    # vis_img_path = "./SeAFusion_main/test_imgs/vi/17.png"
    img_name = "00858N.png"
    onnx_path = "./model/fusionmodel_final.onnx"
    assert os.path.exists(onnx_path), "文件不存在"

    vis_img = img_process(vis_img_path,vis_flag=True)
    ir_img  = img_process(ir_img_path,vis_flag=False)
    # 指定计算前向传播过程特定层
    layerNames = ['output']
    vis_Y, vis_Cb, vis_Cr = RGB2YCrCb(vis_img)
    # H,W,C--> N,C,H,W
    blob_vis_Y = cv2.dnn.blobFromImage(vis_Y,scalefactor=1.0,swapRB=False,crop=False)
    blob_img_ir = cv2.dnn.blobFromImage(ir_img,scalefactor=1.0,swapRB=False,crop=False)
    FusionNet = cv2.dnn.readNetFromONNX(onnx_path)
    FusionNet.setInput(blob_vis_Y, 'input_vis')
    FusionNet.setInput(blob_img_ir, 'input_inf')
    # N,C,H,W
    fused_img = FusionNet.forward(layerNames)
    fused_img = fused_img[0]
    fused_img = np.squeeze(fused_img,axis=0)
    fused_img = np.transpose(fused_img,(1,2,0))
    fused_img = YCbCr2RGB(fused_img, vis_Cr, vis_Cb)
    # C,H,W
    fused_img = numpy2img(fused_img,is_norm=True)
    img_cv2 = cv2.cvtColor(fused_img,cv2.COLOR_RGB2BGR)
    cv2.imshow(img_name,img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


