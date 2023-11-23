import torch
import os
import onnx
from SeAFusion_main.FusionNet import FusionNet
# Visible light sensor 可见光传感器
# Infrared light sensor 红外光传感器
PTH_MODEL = "./SeAFusion_main/model/Fusion/fusionmodel_final.pth"
ONNX_MODEL = "./model/fusionmodel_480_80.onnx"
assert os.path.exists(PTH_MODEL), ("文件不存在")

model = FusionNet(output=1)
state_dict = torch.load(PTH_MODEL)
# 模型结构 vis_conv vis_rgbd1 vis_rgbd2 
# inf_conv inf_rgbd1 inf_rgbd2
# decode {4,3,2,1}
# for key in state_dict.keys():
#     print(key)
model.load_state_dict(state_dict)
model.eval()

# rand_input_vis_inf = torch.randn(1,2,480,640)
# input_names = ["input_vis_inf"]
# dynamic_axes = {"input_vis_inf": {2:'high',3:'width'}}



rand_input_vis = torch.randn(1,1,480,80)
rand_input_inf = torch.randn(1,1,480,80)
input_names = ['input_vis', 'input_inf']
dynamic_axes = {'input_vis': {0:'batch',2:'high',3:'width'}, 
                'input_inf': {0:'batch',2:'high',3:'width'}}
output_names = ['output']

torch.onnx.export(model,(rand_input_vis,rand_input_inf)
                  ,ONNX_MODEL, export_params=True,
                  opset_version=12,verbose=False,input_names=input_names,output_names=output_names
                  )
