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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    """
    相机参数容器类（继承自nn.Module）
    职责：
    - 封装相机内外参数和图像数据
    - 计算视图/投影变换矩阵
    - 管理数据存储设备（CPU/GPU）
    特性：
    - 支持COLMAP格式参数输入
    - 自动处理图像归一化和蒙版应用
    - 提供完整的投影变换链计算
    """

    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        """
        初始化相机对象
        参数说明：
        colmap_id: COLMAP数据库中的相机ID
        R: 世界坐标系到相机坐标系的旋转矩阵 [3,3]
        T: 世界坐标系到相机坐标系的平移向量 [3,1]
        FoVx/FoVy: 水平/垂直视场角（弧度）
        image: 原始图像张量 [C,H,W]
        gt_alpha_mask: Alpha蒙版张量 [1,H,W]（可选）
        image_name: 图像文件名
        uid: 自定义唯一标识符
        trans: 附加平移变换 [3,]（用于场景调整）
        scale: 场景缩放因子
        data_device: 数据存储设备（默认cuda）
        """
        
        """
        关键矩阵关系说明：
        world_view_transform = T * R * S * trans  # 包含场景缩放和平移
        projection_matrix = 透视投影矩阵
        full_proj_transform = world_view * projection  # 完整变换
        camera_center = (R^T*(-T)) + trans  # 相机在世界坐标系的位置
        """
        super(Camera, self).__init__()

        # 基础参数存储
        self.uid = uid                  # 唯一标识符
        self.colmap_id = colmap_id      # COLMAP原始ID
        self.R = R                      # 旋转矩阵 [3,3]
        self.T = T                      # 平移向量 [3,1]
        self.FoVx = FoVx                # 水平视场角（弧度）
        self.FoVy = FoVy                # 垂直视场角（弧度）
        self.image_name = image_name    # 图像文件名

        # 设备配置
        try:
            self.data_device = torch.device(data_device)  # 尝试用户指定设备
        except Exception as e:
            print(f"[Warning] 设备{data_device}配置失败，回退到默认cuda")
            self.data_device = torch.device("cuda")  # 失败时使用cuda

        # 图像数据处理
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)  # 归一化并转移设备
        self.image_width = self.original_image.shape[2]   # 图像宽度（像素）
        self.image_height = self.original_image.shape[1]  # 图像高度（像素）
        #这里使用 Alpha mask的目的：

        #背景分离：通过mask滤除图像背景（如绿幕抠图）
        #关注区域聚焦：只保留目标物体的像素参与计算
        #抗边缘伪影：平滑处理物体边缘的混合像素
        # Alpha mask处理
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)  # 应用mask
        else:
            # 无mask时使用全白mask
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        #Alpha Mask：
        #取值范围 [0,1]，表示透明度
        #1 = 完全不透明（完全显示），0 = 完全透明（完全隐藏）
        #常用于PNG图像的透明通道

        # 投影参数
        self.zfar = 100.0    # 远裁剪平面
        self.znear = 0.01    # 近裁剪平面

        # 场景变换参数
        self.trans = trans   # 附加平移 [x,y,z]
        self.scale = scale   # 场景缩放因子

        # 视图变换矩阵计算
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()  # 世界->相机 [4,4]
        
        # 投影矩阵计算
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, 
            zfar=self.zfar, 
            fovX=self.FoVx, 
            fovY=self.FoVy
        ).transpose(0,1).cuda()  # 投影矩阵 [4,4]
        
        # 组合变换矩阵
        self.full_proj_transform = (  # 世界->裁剪空间 [4,4]
            self.world_view_transform.unsqueeze(0)  # 增加批次维度
            .bmm(self.projection_matrix.unsqueeze(0))  # 矩阵相乘
            .squeeze(0)  # 移除批次维度
        )
        
        # 相机中心计算（世界坐标系）
        self.camera_center = self.world_view_transform.inverse()[3, :3]  # 从逆矩阵提取位置

class MiniCam:
    """
    轻量级相机参数容器类（用于快速渲染或临时相机配置）
    特点：
    - 仅包含核心渲染参数，无图像数据
    - 直接接受预计算变换矩阵，避免重复计算
    典型应用场景：
    - 实时视口预览
    - 多视角批量渲染
    - 相机路径插值
    """

    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        """
        初始化微型相机
        参数说明：
        width: 图像宽度（像素）
        height: 图像高度（像素）
        fovy: 垂直方向视场角（弧度）
        fovx: 水平方向视场角（弧度）
        znear: 近裁剪平面距离（>0）
        zfar: 远裁剪平面距离（>znear）
        world_view_transform: 世界到相机视图的变换矩阵 [4x4 torch.Tensor]
        full_proj_transform: 组合投影变换矩阵 [4x4 torch.Tensor]
        """
        # 基础参数
        self.image_width = width        # 输出图像宽度
        self.image_height = height      # 输出图像高度
        self.FoVy = fovy                # 垂直视场角（来自原始相机）
        self.FoVx = fovx                # 水平视场角（可能根据宽高比调整）
        
        # 投影参数
        self.znear = znear              # 近裁剪平面（防止数值不稳定）
        self.zfar = zfar                # 远裁剪平面（场景尺度相关）
        
        # 变换矩阵
        self.world_view_transform = world_view_transform  # 世界->相机变换 [4x4]
        self.full_proj_transform = full_proj_transform    # 世界->裁剪空间变换 [4x4]
        
        # 相机中心计算
        view_inv = torch.inverse(self.world_view_transform)  # 计算逆矩阵（相机->世界）
        self.camera_center = view_inv[3][:3]  # 提取位置分量 [x,y,z]

    """
    与完整Camera类的差异对比：
    +---------------------+-------------------------------+-------------------------------+
    | 特性                | MiniCam                       | Camera                        |
    +---------------------+-------------------------------+-------------------------------+
    | 图像数据存储         | ❌ 无                         | ✅ 包含图像和蒙版              |
    | 矩阵预计算          | ✅ 直接输入                   | ❌ 内部计算                   |
    | 设备管理            | ❌ 无                         | ✅ 支持CPU/GPU                |
    | 场景缩放/平移       | ❌ 无                         | ✅ 支持                       |
    | 使用场景            | 快速渲染/临时相机             | 完整场景重建                  |
    +---------------------+-------------------------------+-------------------------------+
    
    矩阵关系示意图：
    world_view_transform ────┐
                             ├─> full_proj_transform = world_view * projection
    projection_matrix ───────┘
    """

