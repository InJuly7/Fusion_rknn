import cv2
import os
import numpy as np
import time
from post_process import myFunc 
from rknnlite.api import RKNNLite

RKNN_MODEL = "./model/fusionmodel_288_384.rknn"

if __name__ == '__main__':

    img_vis = cv2.imread('vis_s_040.jpg')
    img_ir = cv2.imread('inf_s_040.jpg')

    rknn_lite = RKNNLite()
    print("--->Load RKNN Model")
    ret = rknn_lite.load_rknn(RKNN_MODEL)


    
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    print("Load RKNN Model done!")

    print("--->Init runtime environmet")
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print("Init runtime environmet failed")
        exit(ret)
    print("Init runtime environmet done!")

    fused_image = myFunc(rknn_lite,img_vis,img_ir)
    cv2.imshow("Fused_Image",fused_image)
    cv2.waitKey(0)
    rknn_lite.release()