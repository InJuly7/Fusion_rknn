# coding:utf-8
import os
import argparse
from utils import *
import torch
from torch.utils.data import DataLoader
from datasets import Fusion_dataset
from FusionNet import FusionNet
from tqdm import tqdm

# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
def main(ir_dir='./test_imgs/ir', vi_dir='./test_imgs/vi', save_dir='./SeAFusion', 
         fusion_model_path='./model/Fusion/fusionmodel_final.pth'):
    # Load model 
    fusionmodel = FusionNet(output=1)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # fusionmodel.load_state_dict(torch.load(fusion_model_path))
    fusion_model_weights = torch.load(fusion_model_path)
    fusionmodel.load_state_dict(fusion_model_weights)
    # for key in fusion_model_weights.keys():
    #     print(key)
    # layer_list = list(fusionmodel.children())
    # for child in layer_list:
    #     print(child)
    fusionmodel = fusionmodel.to(device)
    print('fusionmodel load done!')
    test_dataset = Fusion_dataset('val', ir_path=ir_dir, vi_path=vi_dir)
    # test_loader iterable
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # 数据加载到CUDA固定内存，提高数据加载速度
        pin_memory=True,
        # 不丢弃最后一个批次
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    # print("test_loader.n_iter: %d"%test_loader.n_iter)
    test_bar = tqdm(test_loader)

    # Pre-process
    # 拿到img_vis img_ir 
    with torch.no_grad():
        # it : loop_index 
        # for it, (img_vis, img_ir, name) in enumerate(test_loader):
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            # print(f"{it} : {img_vis.shape},{img_ir.shape}")
            img_ir = img_ir.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            # vi_Y.shape (1,1,450,620) img_ir.shape (1,1,450,620)
            fused_img = fusionmodel(vi_Y, img_ir)
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            # 多批次循环操作
            for k in range(len(name)):
                img_name = name[0]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(fused_img[k, ::], save_path)
                # 0代表format()的索引 将信息嵌入到进度条中
                test_bar.set_description('Fusion {0} Sucessfully!'.format(img_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./SeAFusion_main/model/Fusion/fusionmodel_final.pth')
    ## dataset
    parser.add_argument('--ir_dir', '-ir_dir', type=str, default='./SeAFusion_main/test_imgs/ir')
    parser.add_argument('--vi_dir', '-vi_dir', type=str, default='./SeAFusion_main/test_imgs/vi')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='./SeAFusion_main/SeAFusion')
    # python test.py -B 1 与 python test.py --batch_size 1 是等价的 默认是1
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    # 使用parse_args()方法解析命令行参数时,
    # 返回一个argparse.Namespace的实例,充当一个简单的容器,
    # 用于存储通过 命令行参数解析器 解析的命令行参数的值
    args = parser.parse_args()
    # print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    print(parser.description)
    print('| testing %s on GPU #%d with pytorch' % ('SeAFusion', args.gpu))
    main(ir_dir=args.ir_dir, vi_dir=args.vi_dir, save_dir=args.save_dir, fusion_model_path=args.model_path)
