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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        #load_iteration 和 loaded_iter 主要用于加载检查点
        if load_iteration:# TODO: 这里的 load_iteration 什么时候会被执行？
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud")) #在point_cloud文件夹中查找最大的迭代次数
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        # 检查是否存在 Colmap 格式的 sparse 文件夹
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)# 调用 Colmap 场景加载函数，加载 Colmap 格式数据
        # 检查是否存在 Blender 格式的 transforms_train.json 文件
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")  # 提示找到 Blender 格式数据
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval) # 调用 Blender 场景加载函数，加载 Blender 格式数据
        else:
            assert False, "Could not recognize scene type!"

        # 如果没有加载已有的模型（即 self.loaded_iter 为 None），则执行以下操作
        if not self.loaded_iter: 
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file: # 打开场景信息中的点云文件，复制到结果路径下的 input.ply 文件中
                dest_file.write(src_file.read())# 将源文件的内容写入目标文件，完成点云文件的复制
            json_cams = []# 初始化一个空列表，用于存储相机信息的 JSON 格式数据
            camlist = [] # 初始化一个空列表，用于存储所有的相机对象
            if scene_info.test_cameras:# 如果场景信息中存在测试相机，则将它们添加到 camlist 列表中
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras: # 如果场景信息中存在训练相机，则将它们添加到 camlist 列表中
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):# 遍历 camlist 中的每个相机对象，并将其转换为 JSON 格式
                json_cams.append(camera_to_JSON(id, cam)) # 使用 camera_to_JSON 函数将相机信息转换为 JSON 格式，并添加到 json_cams 列表中
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:# 打开模型路径下的 cameras.json 文件，准备写入相机信息
                json.dump(json_cams, file) # 将 json_cams 列表中的相机信息以 JSON 格式写入 cameras.json 文件

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"] #相机的范围也由scene_info返回

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:#下面的load_ply函数加载的是高斯点云ply
            self.gaussians.load_ply(os.path.join(self.model_path,#TODO注意看这里，加载是什么情况,什么时候会被执行
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else: #scene_info由前面 sceneLoadTypeCallbacks返回，包含点云，也有CameraInfo类对象，cameras_extent也由scene_info返回
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
#1. load_iteration 参数
#load_iteration 是一个传递给 Scene 类构造函数的参数，用于指定要加载的模型训练迭代次数。它的作用是根据指定的迭代次数加载已经训练好的模型。具体来说：
#
#load_iteration 为 None：表示不加载任何已经训练好的模型，而是从头开始初始化模型。
#
#load_iteration 为 -1：表示加载最新的（即最大迭代次数的）模型。代码会通过 searchForMaxIteration 函数在 point_cloud 文件夹中查找最大的迭代次数，并加载对应的模型。
#
#load_iteration 为一个具体的整数值：表示加载指定迭代次数的模型。
#
#2. loaded_iter 属性
#loaded_iter 是 Scene 类的一个属性，用于存储实际加载的模型迭代次数。它的值根据 load_iteration 参数的不同而有所不同：
#
#如果 load_iteration 为 None，则 loaded_iter 也为 None，表示没有加载任何已经训练好的模型。
#
#如果 load_iteration 为 -1，则 loaded_iter 会被设置为 point_cloud 文件夹中最大的迭代次数。
#
#如果 load_iteration 为一个具体的整数值，则 loaded_iter 会被设置为该值。