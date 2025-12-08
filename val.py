import warnings
warnings.filterwarnings('ignore')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r'')
    model.val(
        data=r'',
              split='test',
              imgsz=640,
              batch=16,
              iou=0.5,
              project='runs/val',
              name='x',
              )
