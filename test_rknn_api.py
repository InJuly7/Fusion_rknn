import cv2
import numpy as np
from rknn.api import RKNN
from pre_process import pre_process_func
from post_process import post_process_func

ONNX_MODEL = "./model/fusionmodel_160_640.onnx"
# ONNX_MODEL = './model/fusionmodel_480_80.onnx'
RK3588_RKNN_MODEL = './model/fusionmodel_160_640.rknn'
DATASET = './dataset.txt'
QUANTIZE_ON = False
VERBOSE_FILE = "./fusionmodel.log"
IMG_VIS_PATH = "./vis_00858N.png"
IMG_IR_PATH = "./ir_00858N.png"

if __name__ == '__main__':
    # Create RKNN object 
    rknn = RKNN(verbose=False,verbose_file=VERBOSE_FILE)
    # pre-process config
    print("--->Config rknn model")
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

# Init runtime environment NHWC
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588',device_id='8550ae0ed69723b3',perf_debug=True,eval_mem=True,core_mask=1)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('Init runtime environment done')

# pre_process
    img_vis = cv2.imread(IMG_VIS_PATH)
    img_ir = cv2.imread(IMG_IR_PATH)
    vis_Y,vis_Cr,vis_Cb,img_ir = pre_process_func(img_vis,img_ir)

# Inferenece
    # HWC-->NHWC
    vis_Y = np.expand_dims(vis_Y,axis=0)
    img_ir = np.expand_dims(img_ir,axis=0)
    
    # 480_80
    # fused_img_list = []
    # for i in range(9):
    #     fused_img = rknn.inference(inputs=[vis_Y[:,:,70*i:70*i+80,:],img_ir[:,:,70*i:70*i+80]],data_format="nhwc")
    #     if i == 8:
    #         fused_img_list.append(fused_img[0])
    #     else:
    #         fused_img_list.append(fused_img[0][:,:,:,:-10])
    
    # 160_640
    fused_img_list = []
    for i in range(3):
        fused_img = rknn.inference(inputs=[vis_Y[:,150*i:150*i+160,:,:],img_ir[:,150*i:150*i+160,:,:]],data_format="nhwc")
        fused_img_list.append(fused_img[0][:,:,0:150,:])
    
    perf_detail = rknn.eval_perf(is_print=True)
    mermory_detail = rknn.eval_memory(is_print=True)

    # 480_80
    # fused_img = np.concatenate(fused_img_list,axis=3)
    # fused_img = post_process_func(fused_img,vis_Cr,vis_Cb)
    # cv2.imwrite("fused_img_480_80.png",fused_img)
    
    # 160_640
    fused_img = np.concatenate(fused_img_list,axis=2)
    fused_image = post_process_func(fused_img,vis_Cr[0:450],vis_Cb[0:450])
    cv2.imwrite("fused_img_160_640.png",fused_image)
    
    # cv2.imshow("Fused_Image",fused_image)
    # cv2.waitKey(0)
    rknn.release()
