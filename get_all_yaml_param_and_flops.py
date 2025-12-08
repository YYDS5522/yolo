import warnings
warnings.filterwarnings('ignore')
import torch  # 深度学习框架，用于模型加载和计算
import glob  # 用于查找符合特定模式的文件路径
import tqdm  # 用于显示循环进度条
from ultralytics import YOLO  #YOLO模型
from ultralytics.utils.torch_utils import model_info  
if __name__ == '__main__':
    flops_dict = {}
    model_size, model_type = 'n', ''
    if model_type == '':
        yaml_base_path = ''  # 配置文件目录
    elif model_type == '':
        yaml_base_path = ''  # 配置文件目录
    for yaml_path in tqdm.tqdm(glob.glob(f'{yaml_base_path}/*.yaml')):
        yaml_path = yaml_path.replace(f'{yaml_base_path}/{model_type}', f'{yaml_base_path}/{model_type}{model_size}')
        # 跳过包含DCN（可变形卷积）的模型配置
        if 'DCN' in yaml_path:
            continue

        try:
            # 加载YOLO模型（从配置文件构建）
            model = YOLO(yaml_path)
            # 融合模型中的卷积和BN层，提升推理速度（不改变精度）
            model.fuse()
            # 获取模型信息：层数、参数量、梯度数、计算量（GFLOPs）
            n_l, n_p, n_g, flops = model_info(model.model)
            # 存储计算量和参数量到字典
            flops_dict[yaml_path] = [flops, n_p]
        except:
            # 若模型加载或计算失败，跳过该配置文件
            continue

    # 按模型计算量（GFLOPs）从小到大排序
    sorted_items = sorted(flops_dict.items(), key=lambda x: x[1][0])
    # 打印排序后的结果：配置文件路径、计算量、参数量
    for key, value in sorted_items:
        print(f"{key}: {value[0]:.2f} GFLOPs {value[1]:,} Params")