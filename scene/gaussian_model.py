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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:
    #设置和初始化类中用于激活和逆激活的各种函数。这些函数将在其他方法中使用，以确保属性在访问和更新时能够正确地进行转换和处理。
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation): # 根据给定的缩放参数、缩放修饰符和旋转参数来构建协方差矩阵。
            L = build_scaling_rotation(scaling_modifier * scaling, rotation) # 构建缩放旋转矩阵
            actual_covariance = L @ L.transpose(1, 2) # 计算实际的协方差矩阵, 即 L @ L.T, 这里的 @ 表示矩阵乘法。
#L.transpose(1, 2) 表示对矩阵 L 进行转置操作。在这里，transpose(1, 2) 用于交换矩阵的第二维和第三维，这在三维矩阵中通常是指交换行和列。
            symm = strip_symmetric(actual_covariance) #确保 actual_covariance 是对称的，并将其赋值给 symm。
            return symm
        #缩放参数的激活函数和逆激活函数 对应的逆函数用于将原始参数转换为优化空间中的可学习参数
        self.scaling_activation = torch.exp #指数函数，保证缩放参数始终为正
        self.scaling_inverse_activation = torch.log #对数函数
        # 构建协方差矩阵
        self.covariance_activation = build_covariance_from_scaling_rotation
        # 不透明度参数的激活函数和逆激活函数
        self.opacity_activation = torch.sigmoid #不透明度参数在正向传播时会被Sigmoid函数压缩到[0,1]区间
        self.inverse_opacity_activation = inverse_sigmoid #不透明度参数在反向传播时会被反向Sigmoid函数进行变换

        self.rotation_activation = torch.nn.functional.normalize #旋转参数在正向传播时会被归一化，确保旋转四元数保持单位长度
        # torch.nn.functional.normalize 通常用于归一化向量而不是旋转矩阵，如果要确保旋转矩阵的有效性，可能还需要额外的处理步骤来保证其正交性和行列式为1
    # 初始化
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0  #球谐阶数 原始是0
        self.max_sh_degree = sh_degree   #最大球谐阶数
        # 存储不同信息的张量（tensor）
        self._xyz = torch.empty(0) #空间位置
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)  #椭球的形状尺度
        self._rotation = torch.empty(0) #椭球的旋转
        self._opacity = torch.empty(0)  #不透明度
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None  #初始化优化器为 None。
        self.percent_dense = 0  #初始化百分比密度为0。
        self.spatial_lr_scale = 0 #初始化空间学习速率缩放为0。
        self.setup_functions() #调用 setup_functions 方法设置各种激活和变换函数
    # 返回模型参数
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),#state_dict()是一个包含优化器当前状态的Python字典，它保存了优化器的所有信息
            self.spatial_lr_scale,
        )
    # 将 model_args 中的各种属性赋值给对象的相应属性，调用 training_setup 方法来设置训练相关的配置，并加载优化器的状态字典以恢复训练状态。
    def restore(self, model_args, training_args):  #! 用于加载检查点
        # 从 model_args 中解包出一系列模型参数和状态信息
        (self.active_sh_degree,  # 活跃的SH度数
        self._xyz,  # XYZ坐标
        self._features_dc,  # 特征（深度图/纹理图）
        self._features_rest,  # 其余特征
        self._scaling,  # 缩放因子
        self._rotation,  # 旋转因子
        self._opacity,  # 不透明度
        self.max_radii2D,  # 最大二维半径
        xyz_gradient_accum,  # XYZ梯度累积值
        denom,  # 分母，用于累积的归一化
        opt_dict,  # 优化器的状态字典
        self.spatial_lr_scale) = model_args  # 空间学习率的缩放因子

        # 调用 training_setup 来设置训练过程中的参数
        self.training_setup(training_args) #设置训练相关的配置，例如学习率。和前面解包赋值初始化不冲突。

        # 恢复梯度累积值和分母
        self.xyz_gradient_accum = xyz_gradient_accum  # 恢复之前保存的梯度累积值
        self.denom = denom  # 恢复之前保存的分母

        # 加载优化器的状态字典，恢复优化器的状态
        self.optimizer.load_state_dict(opt_dict)  # 恢复优化器的状态，以便从中断处继续训练

    
    # @property 装饰器用于将类的方法转换为属性（即可以通过 obj.method 的方式访问，而不需要使用 obj.method() 的形式）
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    # 计算协方差矩阵
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    # 更新球谐函数阶数
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    # 从点云数据创建模型
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale #设置空间学习速率缩放
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()  #将点云数据转换为张量（tensor），并移动到 GPU 上,#!注意点云数据
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) #将点云颜色数据转换为球谐函数表示，并移动到 GPU 上
        
        # 随机选择点的数量，这里选择原始数量的一半
        #num_points = fused_point_cloud.shape[0]
        #new_num_points = num_points // 2
        #indices = torch.randperm(num_points)[:new_num_points]
        #fused_point_cloud = fused_point_cloud[indices]
        #fused_color = fused_color[indices]
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # 初始化一个全零张量，用于存储特征
        features[:, :3, 0 ] = fused_color # 将颜色数据填充到特征张量的前3个通道的第0个位置
        features[:, 3:, 1:] = 0.0 # 将特征张量的第3个通道及其后的所有位置初始化为零，这些位置通常用于存储额外的特征信息。
        # 打印初始点数量，需要对点云数量进行下采样，可以在此进行操作
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # distCUDA2计算点云中每个点的最小距离，torch.clamp_min(..., 0.0000001) 将距离裁剪到最小值 0.0000001，避免距离为零或负值。
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # [..., None].repeat(1, 3) 对得到的值进行扩展，以便与后续计算中的张量形状匹配。
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) #torch.sqrt(dist2) 计算距离的平方根。torch.log(...) 对计算出的距离取对数
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")# 初始化旋转矩阵为单位四元数，行数为点云数量，列数为4，rots张量每一行是一个四元数。
        rots[:, 0] = 1 # 单位四元数默认设置w=1，其余分量为0

        # 初始化不透明度参数：先将0.1通过逆sigmoid映射到优化空间，使sigmoid(opacity)初始值为0.1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))#torch.ones创建一个填充了标量值 1 的张量
        
        # 将各属性包装为可训练参数并注册到模型
        # nn.Parameter类用于将张量标记为模型参数，这样在训练过程中，这些参数可以通过反向传播来更新。
        # requires_grad_(True)表示需要为这些参数计算梯度，从而在训练过程中进行优化。
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # 3D坐标作为可优化参数，#!fused_point_cloud已经传给了_xyz
        # 特征张量维度转换：[N,3,SH_coeffs] -> [N,SH_coeffs,3]，适配后续球谐函数计算(下面两行)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))  # 零阶球谐系数（颜色基础分量）
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))  # 高阶球谐系数（细节分量）
        self._scaling = nn.Parameter(scales.requires_grad_(True))  # 高斯椭球的缩放参数
        self._rotation = nn.Parameter(rots.requires_grad_(True))   # 高斯椭球的旋转参数（四元数）
        self._opacity = nn.Parameter(opacities.requires_grad_(True))  # 不透明度参数（优化空间表示）
        
        # 初始化每个高斯在2D投影的最大半径为0（后续训练中更新）
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args): # 设置训练相关的配置。例如学习率,arguments\__init__.py中OptimizationParams类。crear_from_pcd执行早于该函数，在scene实例化时会被调用
        # 从 training_args 获取稠密化百分比，并初始化相关变量
        self.percent_dense = training_args.percent_dense  # 设置稠密化比例，用于控制训练时稠密化的程度，模型会保留多少比例的高斯球（或点）来保持模型的细节和复杂性
        
        # 初始化梯度积累变量，用于存储梯度累积 #?这里的self.get_xyz.shape[0]的数量，和create_from_pcd中的点云数量是一样的，且crear_from_pcd执行早于该函数
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # XYZ梯度累积
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # 分母，累积值
        #print("Number of self.get_xyz.shape[0] : ", self.get_xyz.shape[0])
        # 为训练过程定义不同的优化器参数和学习率
        l = [ # 包含多个字典的列表，每个字典表示一个优化器的参数组，
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]#'params': 这个键对应的是一个包含模型参数的列表。优化器将会对这些参数进行优化。
        #'lr': 每个参数组的学习率设置。这里不同参数组的学习率有不同的初始值，这取决于 training_args 中的配置。
        #'name': 每个参数组的名称，主要用于调试和记录，帮助识别和跟踪不同的参数组。

        # 使用Adam优化器来优化上述参数，学习率设置为0（后续通过调度器调整）
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        # 创建学习率调度器，根据训练参数调整位置学习率
        self.xyz_scheduler_args = get_expon_lr_func( #得到了一个函数
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,  # 初始位置学习率
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,  # 最终位置学习率
            lr_delay_mult=training_args.position_lr_delay_mult,  # 学习率延迟因子
            max_steps=training_args.position_lr_max_steps  # 最大训练步骤数
        )#学习率调度器的作用是帮助模型在训练过程中逐步降低学习率，避免训练初期的过快收敛或训练后期的振荡，从而使模型更稳定地收敛。

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:# 遍历优化器中的所有参数组,即上面 l
            if param_group["name"] == "xyz":## 找到名称为 "xyz" 的参数组
                lr = self.xyz_scheduler_args(iteration) # 调用学习率调度器，得到当前位置学习率
                param_group['lr'] = lr # 更新该参数组的学习率
                return lr
            
    #construct_list_of_attributes 函数的作用是构建一个包含所有模型属性的列表。这些属性包括：
    #位置（x, y, z）、法线（nx, ny, nz）、零阶球谐系数（DC分量）、高阶球谐系数、不透明度缩放参数、旋转参数
    #这个列表通常用于保存模型数据到文件（如PLY文件）时，定义文件中每个点的属性字段。
    def construct_list_of_attributes(self):
        # 初始化一个列表，包含基本的几何属性：位置 (x, y, z) 和法线 (nx, ny, nz)
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # 添加零阶球谐系数（DC分量）的特征名称
        # self._features_dc.shape[1] 是特征的数量（通常为3，对应RGB颜色）
        # self._features_dc.shape[2] 是球谐系数的数量（通常为1，因为DC分量只有一个）
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))  # 例如：f_dc_0, f_dc_1, f_dc_2
        # 添加高阶球谐系数的特征名称
        # self._features_rest.shape[1] 是特征的数量（通常为3，对应RGB颜色）
        # self._features_rest.shape[2] 是球谐系数的数量（通常为 (max_sh_degree + 1)^2 - 1，因为DC分量已经单独处理）
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))  # 例如：f_rest_0, f_rest_1, ...
        l.append('opacity')# 添加不透明度属性
        # 添加缩放参数的属性名称
        for i in range(self._scaling.shape[1]):# self._scaling.shape[1] 是缩放参数的数量（通常为3，对应x, y, z三个方向的缩放）
            l.append('scale_{}'.format(i))  # 例如：scale_0, scale_1, scale_2
        # 添加旋转参数的属性名称
        for i in range(self._rotation.shape[1]):# self._rotation.shape[1] 是旋转参数的数量（通常为4，对应四元数的四个分量）
            l.append('rot_{}'.format(i))  # 例如：rot_0, rot_1, rot_2, rot_3
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))  # 确保保存路径的目录存在

        xyz = self._xyz.detach().cpu().numpy()  # 获取 3D 坐标 (xyz)，从计算图中分离，移动到 CPU，并转换为 numpy 数组
        normals = np.zeros_like(xyz)  # 初始化法线为与 xyz 相同形状的全零数组
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()  # 将 _features_dc 转换为 numpy 数组，并做适当的转置和扁平化处理
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()  # 同样处理 _features_rest 数据
        opacities = self._opacity.detach().cpu().numpy()  # 将透明度数据转换为 numpy 数组
        scale = self._scaling.detach().cpu().numpy()  # 将缩放数据转换为 numpy 数组
        rotation = self._rotation.detach().cpu().numpy()  # 将旋转数据转换为 numpy 数组
        # 使用了上面定义的 construct_list_of_attributes 函数，构造了一个包含所有模型属性的列表
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]  # 定义每个属性的数据类型，都是 'f4' 类型（32 位浮点数）

        elements = np.empty(xyz.shape[0], dtype=dtype_full)  # 创建一个空的结构化 numpy 数组用于存储顶点数据
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)  # 将所有属性数据按列拼接
        # 不同行是不同点的数据。每一行中的数据顺序是一致的，例如第一行中的所有属性都是对应第一个点的，第二行中的所有属性都是对应第二个点的
        elements[:] = list(map(tuple, attributes))  # 将拼接后的属性数据赋值给元素数组
        el = PlyElement.describe(elements, 'vertex')  # 创建 PlyElement 对象，描述这些数据为 'vertex'（顶点数据）
        PlyData([el]).write(path)  # 将 PlyData（顶点数据）写入指定路径的 .ply 文件

    #对当前透明度值进行一些约束（确保最小值不小于 0.01），然后使用反 sigmoid 函数进行变换，接着将新的透明度值传入优化器
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))  # 确保透明度值不小于0.01并进行反sigmoid变换
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")  # 将更新后的透明度值传递给优化器
        self._opacity = optimizable_tensors["opacity"]  # 更新对象的 _opacity 属性，保存优化后的透明度值

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    #替换优化器中的某个参数并重新初始化与该参数相关的优化器状态（如动量和平方动量），使得新的张量能够顺利参与训练。
    # 这个函数通常用于动态更新模型的某些参数，并确保优化器能够正确处理这些新参数。
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}  # 初始化字典，用于存储替换后的优化器参数
        for group in self.optimizer.param_groups:  # 遍历优化器中的所有参数组
            if group["name"] == name:  # 如果找到名字匹配的参数组
                stored_state = self.optimizer.state.get(group['params'][0], None)  # 获取当前参数的优化器状态
                stored_state["exp_avg"] = torch.zeros_like(tensor)  # 初始化动量（exp_avg）为零张量
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)  # 初始化平方动量（exp_avg_sq）为零张量

                del self.optimizer.state[group['params'][0]]  # 删除旧的优化器状态
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))  # 用新的张量替换旧的参数，确保其可训练
                self.optimizer.state[group['params'][0]] = stored_state  # 将新的参数状态添加回优化器

                optimizable_tensors[group["name"]] = group["params"][0]  # 将更新后的参数添加到字典中
        return optimizable_tensors  # 返回包含替换后的所有优化器参数的字典

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
