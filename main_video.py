import cv2
import time
import os
from rknnpool import rknnPoolExecutor
from func import myFunc 

cap_vis = cv2.VideoCapture("./vis_s_001.mp4")
cap_ir = cv2.VideoCapture("./inf_s_001.mp4")

modelPath = "./model/fusionmodel_final.rknn"
TPEs = 3

pool = rknnPoolExecutor(rknnModel=modelPath,TPEs=TPEs,
                        func=myFunc)

if (cap_vis.isOpened()&cap_ir.isOpened()):
    for i in range(TPEs + 1):
        ret_1, frame_vis = cap_vis.read()
        ret_2, frame_ir = cap_ir.read()

        if (ret_1&ret_2) == 0:
            cap_ir.release()
            cap_vis.release()
            del pool
            print("视频读取失败")
            exit(-1)
        # 读取成功 将帧放入pool中
        pool.put([frame_vis,frame_ir])

# 从视频流中读取帧并显示在窗口中,并显示每30帧的平均帧率 
# frames:记录循环次数;读取了多少帧
# loopTime: 计算每30帧的平均帧率的起始时间
frames, loopTime, initTime = 0, time.time(), time.time()
# 只要视频流cap是打开状态一直执行
while (cap_ir.isOpened()&cap_vis.isOpened()):
    # 每次循环 读取一帧
    frames += 1
    ret_vis, frame_vis = cap_vis.read()
    ret_ir, frame_ir = cap_ir.read()

    if (ret_ir&ret_vis) == 0:
        break
    pool.put([frame_vis,frame_ir])
    frame_fused, flag = pool.get()
    # 获取帧失败
    if flag == False:
        break
    # 显示帧在 'test' 窗口中
    cv2.imshow('test', frame_fused)
    # 按下Q键 退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
        loopTime = time.time()

print("总平均帧率\t", frames / (time.time() - initTime))
# 释放cap和rknn线程池
cap_ir.release()
cap_vis.release()
cv2.destroyAllWindows()
pool.release()
