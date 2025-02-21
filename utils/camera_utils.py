#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    """
    加载并预处理相机数据，生成标准化相机对象
    主要功能：
    - 根据输入参数计算目标分辨率
    - 调整图像尺寸并转换格式
    - 分离RGB图像和Alpha蒙版（如果存在）
    参数说明：
    args: 命令行参数对象，包含resolution/data_device等配置
    id: 相机唯一标识符
    cam_info: 包含原始相机数据的对象（colmap格式）
    resolution_scale: 分辨率缩放因子（用于多尺度训练）
    返回：
    Camera对象：包含处理后的图像数据、相机参数和设备信息
    """
    """
    典型处理流程示例：
    输入：4000x3000像素图像，args.resolution=2，resolution_scale=1.0
    计算：4000/(1.0 * 2)=2000，3000/(1.0 * 2)=1500 → 分辨率2000x1500
    输出：图像缩放到2000x1500，转换为Tensor格式
    """
    # 获取原始图像尺寸
    orig_w, orig_h = cam_info.image.size  # 原始宽高（PIL Image格式）

    # 分辨率计算逻辑 ----------------------------------------------------------
    if args.resolution in [1, 2, 4, 8]:  # 离散缩放模式
        # 计算新分辨率：原始尺寸/(缩放系数*分辨率等级)
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # 连续缩放模式
        if args.resolution == -1:  # 自动缩放模式
            if orig_w > 1600:  # 大尺寸图像处理
                global WARNED  # 全局警告标记（防止重复提示）
                if not WARNED:
                    print("[ INFO ] 检测到超大输入图像（宽度>1600像素），自动缩放到1600宽度。"
                          "如需保留原尺寸，请显式指定--resolution为1")
                    WARNED = True
                global_down = orig_w / 1600  # 计算降采样比例
            else:
                global_down = 1  # 保持原尺寸
        else:  # 指定目标宽度模式
            global_down = orig_w / args.resolution  # 计算降采样比例
        
        # 综合缩放因子计算（全局缩放*分辨率缩放）
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))  # 最终分辨率

    # 图像预处理 --------------------------------------------------------------
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)  # 调整尺寸并转为Tensor [C,H,W]

    # 通道分离
    gt_image = resized_image_rgb[:3, ...]   # 提取RGB通道 [3,H,W]
    loaded_mask = None  # 初始化Alpha蒙版
    if resized_image_rgb.shape[1] == 4:     # 检查是否存在Alpha通道
        loaded_mask = resized_image_rgb[3:4, ...]  # 提取Alpha通道 [1,H,W]

    # 构建相机对象 ------------------------------------------------------------
    return Camera(
        colmap_id=cam_info.uid,    # COLMAP原始ID
        R=cam_info.R,              # 旋转矩阵 [3,3]
        T=cam_info.T,              # 平移向量 [3,1]
        FoVx=cam_info.FovX,        # 水平视场角（弧度）
        FoVy=cam_info.FovY,        # 垂直视场角（弧度） 
        image=gt_image,            # 处理后的RGB图像 [3,H,W]  可由此参数得original_image
        gt_alpha_mask=loaded_mask, # Alpha蒙版（可选）[1,H,W]
        image_name=cam_info.image_name,  # 图像文件名
        uid=id,                    # 自定义唯一ID
        data_device=args.data_device  # 指定存储设备（CPU/GPU）
    )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    """将原始相机信息转换为相机对象列表"""
    camera_list = []  # 初始化空列表存储相机对象
    
    # 遍历所有相机配置信息
    for id, c in enumerate(cam_infos):  # 带索引遍历保证相机ID唯一性
        # 加载并配置单个相机参数
        camera_list.append(loadCam(args, id, c, resolution_scale))  # 调用加载函数生成相机实例
    
    return camera_list  # 返回完整相机对象集合

def camera_to_JSON(id, camera : Camera):
    """将相机数据转换为JSON可序列化格式"""
    # 构建相机到世界坐标系的变换矩阵
    Rt = np.zeros((4, 4))                   # 初始化4x4齐次坐标矩阵
    Rt[:3, :3] = camera.R.transpose()       # 旋转矩阵转置后存入前三行三列
    Rt[:3, 3] = camera.T                    # 平移向量存入前三行第四列
    Rt[3, 3] = 1.0                          # 齐次坐标补位

    # 计算世界到相机坐标系的变换矩阵
    W2C = np.linalg.inv(Rt)                 # 求逆矩阵得到世界坐标系到相机坐标系变换
    pos = W2C[:3, 3]                        # 提取相机在世界空间中的位置坐标
    rot = W2C[:3, :3]                       # 提取相机的旋转矩阵分量
    
    # 准备可序列化的旋转数据
    serializable_array_2d = [x.tolist() for x in rot]  # 将numpy数组转为嵌套列表

    # 构建标准化相机参数字典
    camera_entry = {
        'id' : id,                          # 相机唯一标识符
        'img_name' : camera.image_name,     # 关联图像文件名
        'width' : camera.width,             # 图像水平分辨率
        'height' : camera.height,           # 图像垂直分辨率
        'position': pos.tolist(),           # 三维位置坐标列表化
        'rotation': serializable_array_2d,  # 3x3旋转矩阵二维列表
        'fy' : fov2focal(camera.FovY, camera.height),  # 垂直焦距计算
        'fx' : fov2focal(camera.FovX, camera.width)    # 水平焦距计算
    }
    return camera_entry  # 返回符合JSON格式标准的相机参数
