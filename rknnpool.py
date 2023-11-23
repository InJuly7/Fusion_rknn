from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed

RKNN_MODEL = "./model/fusionmodel_final.rknn"
# 创建RKNNLite对象 指定在哪个核运行
def initRKNN(rknnModel=RKNN_MODEL,id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    print("Load RKNN rknnModel done!")

    # 设置该RKNNLite对象 在哪个核运行
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    if ret != 0:
        print("Init runtime environmet failed")
        exit(ret)
    print("Init runtime environmet done!")
    return rknn_lite

# 设置线程数 为每个线程指定在哪个NPU核执行
# return:RKNNLite对象列表
def initRKNNs(rknnModel=RKNN_MODEL,TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i%3))
    return rknn_list


class rknnPoolExecutor():
    def __init__(self,TPEs,func,rknnModel=RKNN_MODEL):
        self.TPEs = TPEs
        self.queue = Queue()
        # 创建 TPEs个RKNN
        self.rknnPool = initRKNNs(rknnModel,TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0
    
    # num 计算传入多少帧
    def put(self, frame):
        # 传入后处理函数,确定该帧在哪个子线程运行,以及当前帧
        self.queue.put(self.pool.submit(
            self.func, self.rknnPool[self.num % self.TPEs], frame))
        # 记录帧数 尽可能的均匀分配给三个核
        self.num += 1
    
    # 从线程中获取数据
    def get(self):
        # 任务队列为空
        if self.queue.empty():
            return None,False
        data_from_thread = self.queue.get()
        return data_from_thread.result(),True
    
    def release(self):
        self.pool.shutdown()
        # 将创建的每个RKNNLite资源释放掉 rknnPool :RKNNLite对象列表
        for rknn_lite in self.rknnPool:
            rknn_lite.release()