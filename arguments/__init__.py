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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    #接收一个ArgumentParser实例（parser）、一个组名（name）和一个布尔值（fill_none）。fill_none用于控制是否将参数的默认值设置为None。
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):#parser: ArgumentParser, name : str用于指定参数的类型
        group = parser.add_argument_group(name)#语法：调用ArgumentParser的add_argument_group方法。功能：在parser中创建新的参数组，组名为name。
        for key, value in vars(self).items():#语法：调用vars()方法，将self转换为字典，items()方法返回键值对列表。功能：将self的属性和值转换为键值对。
            #检查属性名是否以_开头。如果是，则将shorthand标志设置为True，并去掉属性名的第一个字符。
            shorthand = False
            if key.startswith("_"):
                shorthand = True # 表面有简写
                key = key[1:] # 去掉下划线
            t = type(value) # 获取属性值的类型
            value = value if not fill_none else None # 三元表达式 如果fill_none为True，则将默认值设置为None
            # 根据属性值的类型，调用ArgumentParser的add_argument方法，添加参数到参数组中。
            if shorthand:#需要缩写
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)#为该参数添加短选项（如-k）和长选项（如--key）。
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")# action="store_true"表示该参数是一个布尔值，如果在命令行中出现该参数，则将其值设置为True
                else:
                    group.add_argument("--" + key, default=value, type=t)# type参数用于指定参数值的类型。

    def extract(self, args):#接收一个Namespace对象（args）并返回一个GroupParams对象。
        # 遍历args的属性，如果属性名在self中，则将属性值设置为GroupParams的属性。
        group = GroupParams()#创建一个用于存储提取参数的对象。
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])#设置GroupParams的属性。arg[0] 是参数名称（字符串形式）。arg[1] 是参数值。
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0 # 用于指定球谐函数的最大阶数，默认为3，scene\gaussian_model.py中最大阶数由此限制
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda" #指定源图像数据的放置位置，默认为 cuda，如果对大型/高分辨率数据集进行训练，建议使用 cpu，这将减少 VRAM 消耗，但训练速度会稍慢。
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)# 调用父类的extract方法，将参数提取到GroupParams对象中。
        g.source_path = os.path.abspath(g.source_path)# 将source_path属性设置为绝对路径。
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_00
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
