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
import traceback
import socket
import json
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    '''初始化网络监听套接字并配置非阻塞模式'''
    global host, port, listener        # 声明全局变量以修改网络配置参数
    host = wish_host                   # 存储用户指定的主机地址
    port = wish_port                   # 存储用户指定的端口号
    listener.bind((host, port))        # 将套接字绑定到指定地址端口组合
    listener.listen()                  # 启用TCP连接监听队列
    listener.settimeout(0)             # 设置非阻塞模式(立即返回IO操作结果)

def try_connect():
    '''尝试连接到服务器'''
    global conn, addr, listener        # 声明全局变量以修改网络连接参数
    try:
        conn, addr = listener.accept()  # 接受TCP连接请求
        print(f"\nConnected by {addr}")
        conn.settimeout(None)           # 设置非阻塞模式(等待服务器响应)
    except Exception as inst:
        pass

def read():
    '''从服务器接收数据'''
    global conn
    messageLength = conn.recv(4)       # 接收数据长度信息
    messageLength = int.from_bytes(messageLength, 'little')  # 解析数据长度信息
    message = conn.recv(messageLength)  # 接收数据内容
    return json.loads(message.decode("utf-8"))  # 解析JSON数据并返回

def send(message_bytes, verify):
    '''向服务器发送数据'''
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)    # 发送数据内容
    conn.sendall(len(verify).to_bytes(4, 'little'))  # 发送数据长度信息
    conn.sendall(bytes(verify, 'ascii'))  # 发送验证信息

def receive():
    '''从服务器接收渲染参数'''
    message = read()  # 接收渲染参数信息
    width = message["resolution_x"] # 解析渲染参数
    height = message["resolution_y"] # 解析渲染参数

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None