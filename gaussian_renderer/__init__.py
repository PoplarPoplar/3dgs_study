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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

import math

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    使用3D高斯模型渲染场景到2D图像空间
    主要流程：初始化屏幕空间坐标 -> 配置光栅化参数 -> 准备几何属性 -> 处理颜色计算 -> 执行光栅化 -> 返回结果
    参数说明：
    viewpoint_camera: 包含相机参数的相机对象
    pc: 包含3D高斯属性的模型对象
    pipe: 渲染管线配置参数
    bg_color: 背景颜色张量（必须位于GPU）
    scaling_modifier: 缩放因子调节参数
    override_color: 可选的覆盖颜色值
    """

    # 初始化屏幕空间坐标张量（保留梯度用于后续优化）
    # 创建可梯度追踪的屏幕空间坐标，用于后续的2D均值梯度计算
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()  # 确保梯度保留（兼容性处理）
    except:
        pass

    # 配置光栅化参数
    # 计算相机视场角的tan值用于投影变换
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # X轴方向视场角
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # Y轴方向视场角

    # 构建光栅化设置对象
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 输出图像高度
        image_width=int(viewpoint_camera.image_width),    # 输出图像宽度
        tanfovx=tanfovx,                    # X轴投影参数
        tanfovy=tanfovy,                    # Y轴投影参数
        bg=bg_color,                        # 背景颜色
        scale_modifier=scaling_modifier,    # 缩放调节因子
        viewmatrix=viewpoint_camera.world_view_transform,  # 世界-视图变换矩阵
        projmatrix=viewpoint_camera.full_proj_transform,   # 完整投影变换矩阵
        sh_degree=pc.active_sh_degree,       # 使用的球谐函数阶数
        campos=viewpoint_camera.camera_center,  # 相机位置
        prefiltered=False,                   # 是否预过滤
        debug=pipe.debug                     # 调试模式开关
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # 初始化光栅化器

    # 准备几何属性
    means3D = pc.get_xyz        # 获取3D高斯中心坐标
    means2D = screenspace_points  # 屏幕空间坐标（带梯度）
    opacity = pc.get_opacity    # 获取不透明度值

    # 处理协方差矩阵计算方式
    # 根据配置选择协方差矩阵的预处理方式
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)  # Python端预计算3D协方差
    else:
        scales = pc.get_scaling     # 获取缩放参数
        rotations = pc.get_rotation  # 获取旋转参数

    # 处理颜色计算逻辑
    # 根据配置选择颜色计算路径：覆盖颜色/SH函数计算/光栅化器计算
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:  # Python端球谐函数转换
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)  # 重构SH系数维度
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  # 计算视线方向
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)  # 归一化方向向量
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)  # 球谐函数转RGB
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # 颜色值截断处理
        else:
            shs = pc.get_features  # 直接获取SH系数供光栅化器使用
    else:
        colors_precomp = override_color  # 使用覆盖颜色

    # 执行光栅化过程
    # 将3D高斯投影到2D图像空间，生成渲染图像和可见性信息
    rendered_image, radii = rasterizer(
        means3D = means3D,          # 3D位置
        means2D = means2D,          # 屏幕空间坐标
        shs = shs,                  # 球谐系数
        colors_precomp = colors_precomp,  # 预计算颜色
        opacities = opacity,        # 不透明度
        scales = scales,            # 缩放参数
        rotations = rotations,      # 旋转参数
        cov3D_precomp = cov3D_precomp)  # 预计算协方差

    # 返回结果字典
    return {"render": rendered_image,          # 最终渲染图像
            "viewspace_points": screenspace_points,  # 屏幕空间坐标（带梯度）
            "visibility_filter" : radii > 0,  # 可见性过滤器（半径>0为可见）
            "radii": radii}                   # 各高斯在屏幕空间的半径