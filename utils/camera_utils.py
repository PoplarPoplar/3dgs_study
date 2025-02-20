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
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

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
