import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import os
import torch
torch.cuda.empty_cache()#清理GPU缓存
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('')#改进配置
    # model.load('yolov8n.pt') # loading pretrain weights  加载预训练权重
    model.train(data=r'',
                cache=False,
                imgsz=640,
                epochs=200,#训练轮数
                batch=32,#通道数，一般取32
                close_mosaic=0,#close_mosaic=0表示全程启用Mosaic增强
                workers=0,
                optimizer='SGD',
                # device='0,1',
                # patience=20, # set 0 to close earlystop.早停机制
                # resume=True, # 断点续训
                # amp=False, # close amp
                # fraction=0.2,#数据集使用比例
                project='runs/train',
                name='x',
                )