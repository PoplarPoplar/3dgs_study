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
#graphics_utils.py 文件
#主要涉及点云、相机视图变换、投影矩阵以及视场角与焦距的转换等图形学基础操作。
#它为处理三维几何变换和相机投影提供了必要的数学工具，
#适用于计算机图形学、三维重建、计算机视觉等领域。
import torch
import math
import numpy as np
from typing import NamedTuple
#基础点云对象，主要用于存储和操作点云数据
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    """
    构建基础的世界坐标系到相机坐标系的变换矩阵
    参数：
    R: 旋转矩阵 (3x3)，表示相机姿态（通常来自COLMAP的R，即世界->相机的旋转）
    t: 平移向量 (3,)，表示相机位置（通常来自COLMAP的T，即世界坐标系下的相机原点）
    返回：
    4x4齐次变换矩阵，格式为 [R^T | t; 0 0 0 1]
    """
    Rt = np.zeros((4, 4))          # 初始化4x4矩阵
    Rt[:3, :3] = R.transpose()     # 旋转矩阵转置（相机->世界 转为 世界->相机）
    Rt[:3, 3] = t                  # 设置平移分量
    Rt[3, 3] = 1.0                 # 齐次坐标归一化
    return np.float32(Rt)          # 确保矩阵为float32类型

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    构建带场景缩放和平移调整的世界坐标系到相机坐标系的变换矩阵
    新增功能：
    - 应用场景缩放（scale）
    - 附加平移调整（translate）
    典型应用：用于场景中心化和尺度归一化
    """
    """
    数学原理说明：
    1. 基础变换：
    world_view = [R^T | -R^T @ T] （经典视图矩阵形式）
    
    2. 带缩放平移的变换：
    - 先计算相机在世界中的位置：C = R^T @ (-T)
    - 应用场景变换：C' = (C + translate) * scale
    - 新视图矩阵：world_view' = [R^T | -R^T @ (T - translate/scale)] * diag(1/scale, 1/scale, 1/scale, 1)
    
    典型应用场景：
    - 当场景坐标范围过大时，通过scale缩小数值范围提升计算稳定性
    - 通过translate将场景中心移到原点附近
    """
    # 初始变换矩阵构建（同getWorld2View）
    Rt = np.zeros((4, 4))          # 初始化4x4矩阵
    Rt[:3, :3] = R.transpose()     # 旋转分量
    Rt[:3, 3] = t                  # 平移分量
    Rt[3, 3] = 1.0                 # 齐次坐标
    
    # 计算相机到世界的变换
    C2W = np.linalg.inv(Rt)        # 求逆得到相机->世界的变换
    
    # 调整相机中心位置
    cam_center = C2W[:3, 3]        # 提取相机在世界坐标中的位置
    cam_center = (cam_center + translate) * scale  # 应用场景平移和缩放
    C2W[:3, 3] = cam_center        # 更新相机位置
    
    # 重新计算世界到相机的变换
    Rt = np.linalg.inv(C2W)        # 再次求逆得到调整后的世界->相机变换
    return np.float32(Rt)          # 返回float32类型矩阵
    #代码对比分析：
    #特性	       getWorld2View	    getWorld2View2
    #功能	       基础世界->相机变换	 带场景缩放/平移的变换
    #输入参数	   R, t	                增加translate和scale
    #矩阵计算	   直接构造	             通过逆矩阵调整相机位置后重新计算
    #应用场景	   原始COLMAP数据	     需要场景归一化/中心化的处理
    #计算复杂度	   O(1)	                O(n^3) 因涉及矩阵求逆
    #数值稳定性	   直接使用原始数据	      通过scale提升大场景稳定性

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    计算透视投影矩阵（兼容不同视场角的非对称视锥体）
    参数：
    znear: 近裁剪平面距离（>0）
    zfar: 远裁剪平面距离（>znear）
    fovX: 水平方向视场角（弧度）
    fovY: 垂直方向视场角（弧度）
    返回：
    4x4透视投影矩阵（torch.Tensor），将相机空间坐标映射到裁剪空间
    """
    
    # 计算视场角半角正切值
    tanHalfFovY = math.tan(fovY / 2)  # 垂直方向半角正切
    tanHalfFovX = math.tan(fovX / 2)  # 水平方向半角正切

    # 计算近裁剪平面边界
    top = tanHalfFovY * znear    # 上边界 y = +top
    bottom = -top                # 下边界 y = -top（对称视锥体）
    right = tanHalfFovX * znear  # 右边界 x = +right 
    left = -right                # 左边界 x = -right（对称视锥体）

    # 初始化投影矩阵
    P = torch.zeros(4, 4)        # 创建4x4零矩阵

    z_sign = 1.0  # 控制深度方向（1=右手坐标系，-1=左手坐标系）

    # 填充矩阵元素（按列优先顺序）
    P[0, 0] = 2.0 * znear / (right - left)  # x轴缩放因子
    P[1, 1] = 2.0 * znear / (top - bottom)  # y轴缩放因子
    P[0, 2] = (right + left) / (right - left)  # x轴平移项（保持对称时为0）
    P[1, 2] = (top + bottom) / (top - bottom)  # y轴平移项（保持对称时为0）
    P[3, 2] = z_sign  # 透视除法项（通常用于齐次坐标的w分量）
    P[2, 2] = z_sign * zfar / (zfar - znear)  # 深度压缩因子
    P[2, 3] = -(zfar * znear) / (zfar - znear)  # 深度平移项

    return P  # 返回投影矩阵

"""
数学原理详解：
标准透视投影矩阵形式：
     [ 2n/(r-l)      0      (r+l)/(r-l)       0     ]
     [    0       2n/(t-b)  (t+b)/(t-b)       0     ]
     [    0          0        -f/(f-n)   -fn/(f-n) ]
     [    0          0          -1           0     ]

本实现特点：
1. 简化对称视锥体计算（r=-l, t=-b）
2. 使用z_sign控制深度方向（默认右手坐标系）
3. 保留第四行[0,0,z_sign,0]实现透视除法

坐标系约定：
- 相机空间：右手坐标系，视线方向为+z
- 裁剪空间：x,y ∈ [-1,1]，z ∈ [0,1]（当z_sign=1时）
"""

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))