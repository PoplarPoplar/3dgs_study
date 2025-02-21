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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    """
    计算L1损失（平均绝对误差）
    公式：L = mean(|预测值 - 真实值|)
    特点：对异常值鲁棒，促进稀疏解
    适用场景：图像重建、深度估计等需要保留边缘的任务
    """
    return torch.abs((network_output - gt)).mean()  # 绝对值差 → 全局平均

def l2_loss(network_output, gt):
    """
    计算L2损失（均方误差）
    公式：L = mean((预测值 - 真实值)^2)
    特点：对异常值敏感，收敛速度快
    适用场景：光流估计、回归问题等平滑输出任务
    """
    return ((network_output - gt) ** 2).mean()  # 平方差 → 全局平均

def gaussian(window_size, sigma):
    """
    生成一维高斯核
    公式：G(x) = exp(-(x-center)^2/(2σ²))，中心对称
    参数：
    window_size: 滤波器长度（应为奇数）
    sigma: 标准差（控制核的宽度）
    """
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()  # 归一化保证核元素和为1

def create_window(window_size, channel):
    """
    创建多通道可分离高斯窗口
    步骤：
    1. 生成一维高斯核
    2. 外积得到二维高斯核
    3. 扩展为多通道卷积核
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # 列向量 [w,1]
    _2D_window = _1D_window.mm(_1D_window.t()).float()     # 矩阵乘法 [w,w]
    window = _2D_window.unsqueeze(0).unsqueeze(0)          # 增加维度 [1,1,w,w]
    return window.expand(channel, 1, window_size, window_size).contiguous()  # [C,1,w,w]

def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算结构相似性指数（SSIM）
    流程：
    1. 创建高斯卷积核
    2. 处理设备兼容性
    3. 调用核心计算函数
    注意：输入图像应归一化到[0,1]
    """
    channel = img1.size(-3)  # 获取通道数（支持NCHW和HWC格式）
    window = create_window(window_size, channel)
    
    # 确保窗口与输入数据在同一设备
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)  # 匹配数据类型
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    SSIM核心计算（基于滑动窗口的局部统计量）
    公式分解：
    - 亮度对比：2μ1μ2/(μ1²+μ2²)
    - 结构对比：2σ12/(σ1²+σ2²)
    - 加入稳定性常数C1,C2防止除零
    """
    # 计算局部均值（高斯模糊）
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)  # [B,C,H,W]
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    # 计算均值平方和互相关
    mu1_sq = mu1.pow(2)    # 各元素平方 [B,C,H,W]
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2    # 元素级乘积

    # 计算方差和协方差（利用E[X²] - E[X]^2）
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # 稳定性常数（基于论文设置）
    C1 = 0.01**2  # 亮度分量常数
    C2 = 0.03**2  # 对比度分量常数

    # 组合SSIM公式
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 空间维度平均（默认全局平均）
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)  # 保留批次维度

"""
SSIM计算流程详解：
1. 高斯模糊：使用可分离高斯滤波器进行两次卷积
2. 统计量计算：
   - μ: 局部均值（亮度）
   - σ²: 局部方差（对比度）
   - σ12: 局部协方差（结构相似性）
3. 分量组合：将亮度、对比度、结构分量相乘
4. 归一化处理：通过常数项避免不稳定情况

参数选择建议：
- 窗口大小：11x11（适合512x512图像）
- σ：1.5（经验值，控制高斯核衰减速度）
- C1/C2：根据图像像素范围调整（假设输入已归一化）

典型输出范围：
- 完美匹配：1.0
- 完全不相关：接近0（可能为负值）
- 一般情况：0.6-0.95

与MSE的视觉对比：
[清晰图像] SSIM=0.92 | MSE=0.001
[模糊图像] SSIM=0.76 | MSE=0.005
[噪声图像] SSIM=0.58 | MSE=0.008
"""

