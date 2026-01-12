import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import os
import torch
torch.cuda.empty_cache()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('')
    # model.load('yolov8n.pt') # loading pretrain weights 
    model.train(data=r'',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=0,
                optimizer='SGD',
                # device='0,1',
                # patience=20,
                # resume=True, 
                # amp=False, # close amp
                project='runs/train',
                name='x',
                )
