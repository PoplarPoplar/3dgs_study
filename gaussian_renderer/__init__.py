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

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """执行3D高斯渲染流程，生成视角图像及辅助信息"""
    # 初始化屏幕空间坐标跟踪（用于反向传播梯度计算）
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0  # +0防止原地操作错误
    try:  # 兼容不同版本梯度保留机制
        screenspace_points.retain_grad()  # 保持梯度用于后续优化
    except: pass

    # 配置视锥体参数
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 水平视场角正切值
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 垂直视场角正切值

    # 构建光栅化参数集
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 输出图像高度
        image_width=int(viewpoint_camera.image_width),    # 输出图像宽度
        tanfovx=tanfovx,          # 用于计算像素覆盖范围
        tanfovy=tanfovy,          # 控制透视投影变形
        bg=bg_color,              # 背景填充颜色
        scale_modifier=scaling_modifier,  # 高斯尺度调节系数
        viewmatrix=viewpoint_camera.world_view_transform,  # 世界到相机坐标变换
        projmatrix=viewpoint_camera.full_proj_transform,   # 组合投影矩阵
        sh_degree=pc.active_sh_degree,     # 当前使用的球谐阶数
        campos=viewpoint_camera.camera_center,  # 相机中心位置
        prefiltered=False,         # 是否启用预过滤
        debug=pipe.debug           # 调试模式开关
    )

    # 初始化光栅化器
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # 创建CUDA光栅化实例

    # 准备高斯属性数据
    means3D = pc.get_xyz          # 3D均值(中心位置)
    means2D = screenspace_points  # 2D投影坐标(需计算梯度)
    opacity = pc.get_opacity      # 不透明度参数

    # 协方差矩阵计算路径选择
    scales, rotations, cov3D_precomp = None, None, None
    if pipe.compute_cov3D_python:  # Python端预计算3D协方差
        cov3D_precomp = pc.get_covariance(scaling_modifier)  # 考虑缩放修正的协方差
    else:  # 由光栅化器实时计算
        scales = pc.get_scaling   # 各向异性缩放参数
        rotations = pc.get_rotation  # 旋转四元数

    # 颜色计算路径选择
    shs, colors_precomp = None, None
    if override_color is None:    # 未指定覆盖颜色时
        if pipe.convert_SHs_python:  # Python端球谐计算
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)  # 重组SH系数
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  # 视线方向向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)  # 单位化方向向量
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)  # SH到RGB转换
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # 颜色值截断处理
        else:  # 由光栅化器计算
            shs = pc.get_features  # 原始球谐系数
    else:  # 使用覆盖颜色
        colors_precomp = override_color  # 外部指定颜色

    # 执行光栅化渲染
    rendered_image, radii = rasterizer(
        means3D = means3D,        # 高斯中心3D坐标
        means2D = means2D,        # 屏幕空间投影坐标
        shs = shs,                # 球谐系数
        colors_precomp = colors_precomp,  # 预计算颜色
        opacities = opacity,      # 不透明度值
        scales = scales,          # 缩放参数
        rotations = rotations,    # 旋转参数
        cov3D_precomp = cov3D_precomp)  # 预计算协方差

    # 组装返回结果
    return {
        "render": rendered_image,            # 渲染图像RGB数据
        "viewspace_points": screenspace_points,  # 带梯度的屏幕坐标
        "visibility_filter" : radii > 0,     # 可见高斯标识掩码
        "radii": radii}                      # 高斯投影半径(用于后续优化)