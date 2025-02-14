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

import numpy as np
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """从二进制文件中读取并解包下一个字节。
    
    :param fid: 文件对象，代表打开的二进制文件的文件描述符。
    :param num_bytes: 读取的字节总数，应为2，4，8的组合，例如2, 6, 16, 30等。
    :param format_char_sequence: 格式字符的列表，支持的字符包括 {c, e, f, d, h, H, i, I, l, L, q, Q}。
    :param endian_character: 字节顺序字符，可选值包括 {@, =, <, >, !}，默认为小端（<）。
    :return: 读取并解包后的值的元组。
    """
    data = fid.read(num_bytes)  # 从文件中读取指定数量的字节
    return struct.unpack(endian_character + format_char_sequence, data)  # 解包并返回结果


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    
    # 打开二进制文件进行读取
    with open(path_to_model_file, "rb") as fid:
        # 读取点的数量，格式为无符号长整型 (Q)，占8个字节
        num_points = read_next_bytes(fid, 8, "Q")[0]

        # 初始化数组以存储每个点的坐标、颜色和误差
        xyzs = np.empty((num_points, 3))  # 用于存储每个点的坐标 (x, y, z)
        rgbs = np.empty((num_points, 3))  # 用于存储每个点的颜色 (r, g, b)
        errors = np.empty((num_points, 1))  # 用于存储每个点的误差

        # 遍历每个点，读取其相关数据
        for p_id in range(num_points):
            # 读取每个点的属性：ID、坐标(x, y, z)、颜色(r, g, b)、误差
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            # 获取坐标数据
            xyz = np.array(binary_point_line_properties[1:4])  # 提取坐标 (x, y, z)
            # 获取颜色数据
            rgb = np.array(binary_point_line_properties[4:7])  # 提取颜色 (r, g, b)
            # 获取误差数据
            error = np.array(binary_point_line_properties[7])  # 提取误差值
            # 读取轨迹长度（跟踪点的数量）
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            # 读取轨迹中的每个点的 ID，长度为 `track_length`
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            # 将当前点的数据存储到对应的数组中
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error

    # 返回所有点的坐标、颜色和误差
    return xyzs, rgbs, errors


def read_intrinsics_text(path):
    """
    从文本文件中读取相机内参数据并返回一个字典。每个字典条目表示一个相机，包含相机的 ID、内参模型、图像宽度、高度及内参参数。
    
    来源：https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    # 创建一个空字典，用于存储所有相机的信息
    cameras = {}

    # 以读取模式打开指定路径的文本文件
    with open(path, "r") as fid:
        while True:
            # 逐行读取文件
            line = fid.readline()
            # 如果读取到文件末尾，跳出循环
            if not line:
                break
            # 去除行尾的空白字符（如换行符）
            line = line.strip()

            # 如果该行有效（非空且不以 '#' 开头），则继续处理
            if len(line) > 0 and line[0] != "#":
                # 将当前行按照空格分隔为多个元素
                elems = line.split()

                # 解析相机的各个内参字段
                camera_id = int(elems[0])  # 相机 ID
                model = elems[1]           # 相机模型（目前假设为 'PINHOLE'）
                
                # 确保相机模型为 'PINHOLE'，若是其他模型则会触发断言错误
                assert model == "PINHOLE", "While the loader supports other types, the rest of the code assumes PINHOLE"
                
                width = int(elems[2])      # 图像宽度
                height = int(elems[3])     # 图像高度
                # 剩余的元素为相机的内参参数，转换为浮动类型并转为 NumPy 数组
                params = np.array(tuple(map(float, elems[4:])))

                # 将相机数据封装到 Camera 对象中并存储到字典中
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)

    # 返回包含所有相机信息的字典
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """
    读取包含图像外部参数（旋转向量、平移向量、图像名称等）的二进制文件，并将数据存储为一个字典。
    每个字典项代表一张图像，包含其 ID、外部参数、相机 ID、2D 特征点和对应的 3D 点 ID。

    参考：src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    # 创建一个空字典，用于存储所有图像的信息
    images = {}

    # 打开二进制文件
    with open(path_to_model_file, "rb") as fid:
        # 读取文件中记录的图像数量
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]

        # 遍历每一张图像
        for _ in range(num_reg_images):
            # 读取每张图像的64字节属性
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            
            # 解析图像的 ID、旋转向量、平移向量、相机 ID
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])  # 四元数旋转向量
            tvec = np.array(binary_image_properties[5:8])  # 平移向量
            camera_id = binary_image_properties[8]

            # 读取图像名称（以 '\x00' 为结束符）
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # 查找 ASCII 0 作为结束符
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            # 读取当前图像的 2D 特征点数量
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            
            # 读取 2D 特征点的坐标和对应的 3D 点 ID
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D, format_char_sequence="ddq"*num_points2D)

            # 将读取的 2D 坐标（x, y）与 3D 点 ID 分离
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

            # 将图像的所有信息存储到字典中
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)

    # 返回包含所有图像信息的字典
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
