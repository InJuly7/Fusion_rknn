import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite
import os
# decice tree for rk356x/rk3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                elif 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

INPUT_SIZE = 224
RK3588_RKNN_MODEL = 'fusionmodel78.rknn'

def show_top5(result):
    output = result[0].reshape(-1)
    # softmax
    output = np.exp(output)/sum(np.exp(output))
    output_sorted = sorted(output, reverse=True)
    top5_str = 'resnet18\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


if __name__ == '__main__':

    host_name = get_host()
    if host_name == 'RK3566_RK3568':
        rknn_model = RK3566_RK3568_RKNN_MODEL
    elif host_name == 'RK3562':
        rknn_model = RK3562_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    # run on RK356x/RK3588 with Debian OS, do not need specify target.
    if host_name == 'RK3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    #cv2.nameWindows("Fusion")
    # Inference
    print('--> Running model')
    for i in range(1,20):
        #print(filename) #just for test
        #img is used to store the image data
        img = cv2.imread("gray_images/IR" + str(i)+".png",0)
        img = cv2.resize(img, (640, 480))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=img/np.float32(255)
        
        img2 = cv2.imread("gray_images/VIS" + str(i)+".png",0)
        img2 = cv2.resize(img2, (640, 480))
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2=img2/np.float32(255)
        
        #img2=np.transpose(img2,(2,0,1))
        output = rknn_lite.inference(inputs=[img,img2])
        output = np.float32(output)*np.float32(255)
        output = np.uint8(output)
        #cv2.imwrite("result/"+str(i)+".png",output[0][0][0])
        cv2.imshow("Fusion",output[0][0][0])
        cv2.waitKey(2)
    cv2.destroyAllWindows()
    rknn_lite.release()
