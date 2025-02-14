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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
#该函数的作用是根据相机的位置信息（即旋转矩阵 R 和平移向量 T）
#计算一个包围所有相机的球体的中心和半径，确定了相机的范围。
#这些信息通常用于神经辐射场（NeRF）模型中的场景预处理（如归一化），
#以确保场景的视角和缩放适合模型的训练。
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)  # 将所有相机的中心位置横向拼接
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)  # 计算所有相机位置的平均值（即场景的中心）
        center = avg_cam_center  # 得到场景的中心
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)  # 计算每个相机与中心的距离
        diagonal = np.max(dist)  # 找到最大距离，这个最大值即为包围所有相机的最小球体的对角线长度
        return center.flatten(), diagonal  # 返回中心点和对角线长度（即球体的直径）

    cam_centers = []  # 初始化一个空列表，用于存储所有相机的世界坐标系中的中心位置

    for cam in cam_info:
        # 获取相机的世界到视图转换矩阵（W2C）
        W2C = getWorld2View2(cam.R, cam.T)
        # 将其转换为视图到世界的转换矩阵（C2W），并获取相机中心的位置（位于矩阵的最后一列）
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])  # 提取相机中心的3D位置，添加到 cam_centers 列表中

    # 计算所有相机的场景中心和最大对角线长度
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1  # 将球体的半径设置为对角线长度的1.1倍，确保包含所有相机

    translate = -center  # 将场景中心平移到原点，返回平移向量

    # 返回场景的平移向量和半径，供后续神经辐射场的归一化使用
    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []  # 初始化一个空列表，用于存储所有相机的相关信息

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]  # 获取当前相机的外部参数（旋转、平移）
        intr = cam_intrinsics[extr.camera_id]  # 获取相机的内部参数
        height = intr.height  # 相机图像的高度
        width = intr.width  # 相机图像的宽度

        uid = intr.id  # 获取相机的 ID
        R = np.transpose(qvec2rotmat(extr.qvec))  # 将四元数转换为旋转矩阵，并转置
        T = np.array(extr.tvec)  # 将平移向量转换为 NumPy 数组

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)  # 计算垂直视场角
            FovX = focal2fov(focal_length_x, width)  # 计算水平视场角
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)  # 计算垂直视场角
            FovX = focal2fov(focal_length_x, width)  # 计算水平视场角
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))  # 计算图像文件的路径
        image_name = os.path.basename(image_path).split(".")[0]  # 获取图像文件名（去除扩展名）
        image = Image.open(image_path)  # 打开图像文件

        # 创建一个 CameraInfo 对象并添加到 cam_infos 列表
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)  # 将当前相机的所有信息添加到列表 cam_infos 中

    sys.stdout.write('\n')  # 输出换行符，表示进度显示完成
    return cam_infos  # 返回包含所有相机信息的列表 cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)  # 读取 .ply 文件数据
    vertices = plydata['vertex']  # 获取点云数据中的顶点部分（包含位置、颜色和法线）
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T  # 将点云的 x, y, z 坐标合并为一个 NumPy 数组，形状为 (num_points, 3)
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0  # 将点云的颜色信息（红、绿、蓝）合并为一个 NumPy 数组，并归一化到 [0, 1] 范围
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T  # 将点云的法线信息（nx, ny, nz）合并为一个 NumPy 数组，形状为 (num_points, 3)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)  # 返回包含位置、颜色和法线信息的 BasicPointCloud 对象。主要用于存储和操作点云数据

def storePly(path, xyz, rgb):
    #定义一个结构化数据类型 dtype，用于表示每个点的信息。结构包含：
    #x, y, z：三维坐标，数据类型为 f4（4字节浮点数）。
    #nx, ny, nz：法线分量，数据类型为 f4。
    #red, green, blue：RGB颜色值，数据类型为 u1（无符号1字节整数，范围 0-255）。
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)  # 创建法线数组，默认值为零向量，大小与 xyz 相同

    elements = np.empty(xyz.shape[0], dtype=dtype)  # 创建一个空的结构化 NumPy 数组，用于存储点云数据
    attributes = np.concatenate((xyz, normals, rgb), axis=1)  # 将 xyz、法线和 rgb 连接成一个数组
    elements[:] = list(map(tuple, attributes))  # 将连接后的数据转换为元组，并存入 elements 中

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')  # 描述顶点元素
    ply_data = PlyData([vertex_element])  # 创建 PlyData 对象
    ply_data.write(path)  # 将数据写入指定路径的 .ply 文件


# 定义函数：读取COLMAP重建的场景信息（相机参数、点云等），返回SceneInfo对象
# 参数说明：
# - path: COLMAP重建结果的根目录路径
# - images: 自定义图像文件夹名称（默认为None，使用"images"）
# - eval: 是否进行验证集划分（布尔值）
# - llffhold: 验证集采样间隔（默认每8个取1个作为测试集）
def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        # 尝试从二进制文件读取相机外参和内参
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)  # 读取相机外参（位姿）
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)  # 读取相机内参（焦距、畸变等）
    except:
        # 二进制文件读取失败时，改用文本文件读取
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)  # 从文本文件读取外参
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)   # 从文本文件读取内参

    # 确定图像文件夹路径：若images参数为None，默认使用"images"，否则用传入的名称
    reading_dir = "images" if images == None else images
    # 读取所有相机信息（包括图像路径、位姿、内参等），结果未排序
    cam_infos_unsorted = readColmapCameras( #! 结果里有CameraInfo类对象，即本类的实例
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir)
    )
    # 按图像名称对相机信息排序，确保顺序一致，排序的原理是对每个相机信息对象中的 image_name 属性（即文件名）进行一一匹配，并根据文件名的字典顺序来排序。
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    
    # 划分训练集和测试集
    if eval:
        # 每隔llffhold个样本取1个作为测试集，其余为训练集
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0] #使用了列表推导式和enumerate函数,最后将enumerate加入列表中
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        # 不划分测试集，全部用于训练
        train_cam_infos = cam_infos
        test_cam_infos = []

    # 计算NeRF所需的归一化参数（场景中心、缩放系数等）
    nerf_normalization = getNerfppNorm(train_cam_infos) #确定了相机的范围。

    # 处理3D点云数据
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")#colmap的点云数据格式
    txt_path = os.path.join(path, "sparse/0/points3D.txt")#二进制不行用txt
    if not os.path.exists(ply_path):
        # 如果不存在.ply文件，从.bin或.txt转换并保存为.ply
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")#只有第一次会这样
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)  # 尝试读取二进制点云 #  _表示一个不打算使用的值。
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)  # 失败则读取文本格式
        storePly(ply_path, xyz, rgb)  # 将点云保存为.ply格式
    try:
        pcd = fetchPly(ply_path)  # 加载.ply点云数据
    except:
        pcd = None  # 加载失败则设为None

    # 封装场景信息到SceneInfo对象
    scene_info = SceneInfo(
        point_cloud=pcd,               # 3D点云数据
        train_cameras=train_cam_infos,  # 训练集相机参数
        test_cameras=test_cam_infos,    # 测试集相机参数
        nerf_normalization=nerf_normalization,  # 归一化参数，返回场景中心和相机的范围.
        ply_path=ply_path               # 点云文件路径
    )
    return scene_info  # 返回场景信息对象

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}