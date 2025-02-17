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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):# inverse_sigmoid 函数的作用是将一个介于 (0, 1) 范围内的值（通常是 sigmoid 函数的输出）转换回对应的实数值。
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
#该函数的主要目的是根据训练步数（step）动态调整学习率，从初始值 lr_init 衰减到最终值 lr_final。
# 如果设置了延迟步骤 lr_delay_steps，则在训练的开始阶段，学习率会先延迟衰减，直到步数超过 lr_delay_steps 后，才开始使用正常的衰减方式。
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    生成一个连续的学习率衰减函数，适用于对数线性衰减。
    参数:
    - lr_init: 初始学习率
    - lr_final: 最终学习率
    - lr_delay_steps: 延迟步骤数，如果大于0则学习率会在前几步内保持较高，直到步数超过 lr_delay_steps 后才开始衰减
    - lr_delay_mult: 延迟衰减时的学习率乘法因子，默认值为 1.0
    - max_steps: 最大训练步骤数，决定衰减结束时的学习率
    返回值: 一个函数 `helper(step)`，根据训练的当前步数调整学习率
    """

    def helper(step):
        # 如果步数小于 0 或者初始化和最终学习率都为 0，则返回 0.0，表示该参数已禁用
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0

        # 如果设置了延迟衰减，则使用一个平滑的余弦函数来计算延迟的学习率
        if lr_delay_steps > 0:
            # 使用逆余弦衰减，使得训练开始时学习率较高，然后平滑过渡到目标学习率
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            # 如果没有设置延迟衰减，则学习率乘以 1（即无延迟）
            delay_rate = 1.0
        
        # 计算当前步数 t 的归一化值，t 从 0 到 1
        t = np.clip(step / max_steps, 0, 1)
        
        # 使用对数线性插值计算学习率：从 lr_init 到 lr_final
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        
        # 返回调整后的学习率：乘以延迟因子和对数线性插值计算的学习率
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
