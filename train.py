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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset) # 准备输出文件夹和设置日志记录器
    gaussians = GaussianModel(dataset.sh_degree)  # 初始化高斯模型，输入的参数用来指定球谐函数的最大阶数
    scene = Scene(dataset, gaussians)  # 初始化场景 #! gaussians是一个可变对象，在训练过程中会被修改，则scene中的gaussians也会被修改，因此检查点没有直接作用与scene
    gaussians.training_setup(opt) # 设置训练参数，例如学习率，和前面初始化不冲突，且crear_from_pcd执行早于该函数,在scene实例化时会被调用
    
    # 检查点加载逻辑
    if checkpoint: # 如果有检查点，加载检查点
        (model_params, first_iter) = torch.load(checkpoint)  # 加载检查点，得到模型参数和迭代次数
        gaussians.restore(model_params, opt) # 恢复模型参数和优化器状态，即OptimizationParams，优化参数

    # 背景颜色初始化
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] # 设置背景颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 创建背景张量并送入GPU

    # CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing = True)  # 创建CUDA事件记录迭代开始时间
    iter_end = torch.cuda.Event(enable_timing = True)    # 创建CUDA事件记录迭代结束时间

    # 训练状态初始化
    viewpoint_stack = None       # 相机视角栈初始化
    ema_loss_for_log = 0.0       # 指数移动平均损失值（用于平滑日志显示）
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")  # 创建进度条对象
    first_iter += 1  # 调整起始迭代次数
    
    # 主训练循环
    for iteration in range(first_iter, opt.iterations + 1):        
        # 网络GUI连接处理  主要是SIBR_remoteGaussian_app用来实时查看训练的
        if network_gui.conn == None:                     # 检查网络GUI连接状态
            network_gui.try_connect()                    # 尝试建立GUI连接
        while network_gui.conn != None:                  # 处理持续连接状态
            try:
                net_image_bytes = None
                # 接收GUI参数：自定义相机、训练控制标志等
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:                   # 处理自定义相机渲染
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)  # 发送渲染结果到GUI
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()  # 记录迭代开始时间  然后才能使用elapsed_time()方法计算时间间隔

        # 学习率更新
        gaussians.update_learning_rate(iteration)  # 根据当前迭代次数调整学习率

        # 球谐函数等级提升
        if iteration % 1000 == 0:                # 每1000次迭代提升球谐函数等级
            gaussians.oneupSHdegree()            # 增加球谐函数的阶数（增强表达能力）

        # 相机视角采样
        if not viewpoint_stack:                  # 当视角栈为空时重新填充
            viewpoint_stack = scene.getTrainCameras().copy()  # 从场景中获取训练相机列表 Camera 类的实例
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))  # 随机选择一个训练视角

        # 渲染设置
        if (iteration - 1) == debug_from:        # 调试模式开关
            pipe.debug = True

        # 背景选择逻辑
        bg = torch.rand((3), device="cuda") if opt.random_background else background  # 随机背景或固定背景

        # 执行渲染操作 #TODO 这里包含CUDA代码，以后再阅读
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)  # 渲染当前视角得到图像和中间数据 这块很多使用CUDA和C  pipe为PipelineParams类的实例
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # 损失计算与反向传播
        gt_image = viewpoint_cam.original_image.cuda()  # 获取真实图像 original_image是Camera类的属性
        Ll1 = l1_loss(image, gt_image)                  # 计算L1损失
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))  # 组合L1和SSIM损失 lambda_dssim是DSSIM损失函数的权重，影响损失函数的平衡
        loss.backward()                                 # 反向传播计算梯度

        iter_end.record()  # 记录迭代结束时间

        # 无梯度计算上下文  在这个包裹的所有代码中，禁用梯度跟踪。
        with torch.no_grad():
            # 进度条更新逻辑
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log  # 计算指数移动平均损失
            if iteration % 10 == 0:                                        # 每10次迭代更新进度条
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:                                # 训练结束时关闭进度条
                progress_bar.close()

            # 训练结果记录与保存  iter_start.elapsed_time(iter_end)用于计算从 iter_start 开始到 iter_end 之间的时间间隔，是每一轮的时间。
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))  # 生成训练报告
            if (iteration in saving_iterations):                           # 到达指定保存迭代时保存模型
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 高斯模型致密化处理
            if iteration < opt.densify_until_iter:                         # 在指定迭代前执行致密化
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])  # 更新最大二维半径
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)  # 收集致密化统计信息,作为后续的条件判断

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  # 满足条件时执行分裂/修剪
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # 半径阈值，用于控制分裂/修剪的粒度
                    #0.005是一个固定值，scene.cameras_extent是相机的范围 size_threshold是一个阈值，用于控制分裂/修剪的粒度
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)  # 执行致密化和修剪操作 
                    #一方面是周期性重置，另一方面是当使用白色背景且达到致密化起始迭代时，额外触发一次不透明度重置
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):  # 重置不透明度。
                    gaussians.reset_opacity()

            # 优化器参数更新
            if iteration < opt.iterations:                                 # 在最终迭代前执行参数更新
                gaussians.optimizer.step()                                 # 执行优化器参数更新
                gaussians.optimizer.zero_grad(set_to_none = True)          # 清空梯度

            # 检查点保存
            if (iteration in checkpoint_iterations):                       # 到达检查点保存迭代时
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")  # 保存模型状态和迭代次数
# 准备输出文件夹和设置日志记录器
def prepare_output_and_logger(args):    
    if not args.model_path: #model_path参数为空时，生成一个随机的uuid作为输出文件夹名
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = None #!SummaryWriter(args.model_path) #现在关掉了tensorboard的记录功能,有需要可以打开
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
#将l1_loss和render作为参数传递，体现了以下设计原则：
#模块化：函数职责单一，仅关注报告生成，不耦合具体实现。
#可扩展性：通过参数注入支持未来变化，降低维护成本。
#清晰性：显式依赖关系提升代码可读性和可维护性。
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    """
    训练过程监控与报告生成函数
    主要功能：
    - 记录训练指标到TensorBoard
    - 定期在测试集和训练子集上评估模型性能
    - 可视化渲染结果
    - 跟踪场景参数变化
    参数说明：
    tb_writer: TensorBoard写入器对象
    iteration: 当前迭代次数
    Ll1: L1损失值
    loss: 总损失值
    l1_loss: L1损失计算函数
    elapsed: 单次迭代耗时
    testing_iterations: 需要进行测试的迭代点列表
    scene: 包含3D场景数据的Scene对象
    renderFunc: 渲染函数接口
    renderArgs: 渲染函数参数
    """
    
    # ==================== 训练指标记录 ====================
    if tb_writer:
        # 记录L1损失到TensorBoard
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)  # 当前迭代的L1损失
        # 记录总损失到TensorBoard
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)  # 综合损失值
        # 记录迭代耗时
        tb_writer.add_scalar('iter_time', elapsed, iteration)  # 单次迭代时间（秒）

    # ==================== 定期性能评估 ====================
    if iteration in testing_iterations:  # 在预设的关键迭代点进行评估
        torch.cuda.empty_cache()  # 清空显存碎片
        
        # 配置评估数据集（测试集+训练子集）
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()},  # 全量测试集
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] 
                                         for idx in range(5, 30, 5)]}  # 从训练集抽样5个视角
        )

        # 遍历评估配置（测试集和训练子集）
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0  # 初始化L1损失累计值
                psnr_test = 0.0  # 初始化PSNR累计值
                
                # 遍历当前配置的所有相机视角
                for idx, viewpoint in enumerate(config['cameras']):
                    # 渲染当前视角图像
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)  # 限制输出范围[0,1]
                    # 获取对应的真实图像
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)  # 确保数据在GPU
                    
                    # 可视化前5个视角的渲染结果
                    if tb_writer and (idx < 5):
                        # 记录渲染图像到TensorBoard
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                                           image[None], global_step=iteration)  # 添加批次维度
                        # 仅在首次测试迭代记录真实图像
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                               gt_image[None], global_step=iteration)
                    
                    # 累计评估指标
                    l1_test += l1_loss(image, gt_image).mean().double()  # 计算L1损失
                    psnr_test += psnr(image, gt_image).mean().double()  # 计算PSNR
                
                # 计算平均指标
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # 打印评估结果
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                # 记录评估指标到TensorBoard
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # ==================== 场景参数监控 ====================
        if tb_writer:
            # 记录高斯模型不透明度的分布直方图
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            # 记录场景总点数
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        # 再次清空显存
        torch.cuda.empty_cache()
"""
代码结构解析：
1. 训练指标跟踪：
   - 实时记录L1/总损失和迭代时间
   - 数据流向：内存 → TensorBoard

2. 定期评估机制：
   - 测试集全评估 vs 训练集抽样评估
   - 评估流程：渲染 → 计算指标 → 可视化 → 记录
   - 关键设计：只在特定迭代评估以平衡性能

3. 可视化策略：
   - 每个评估点记录最多5张渲染图
   - 仅在首次评估时保存真实图像（避免重复）

4. 资源管理：
   - 评估前后主动清理显存
   - 使用clamp防止渲染值溢出

典型TensorBoard面板元素：
+---------------------+---------------------------+
| 标签                | 含义                      |
+---------------------+---------------------------+
| train_loss_patches  | 训练集损失变化曲线         |
| test/loss_viewpoint | 测试集评估指标            |
| scene/opacity       | 点云不透明度分布          |
| iter_time           | 迭代时间监控              |
+---------------------+---------------------------+

评估指标说明：
- L1 Loss: 衡量像素级绝对误差（值越小越好）
- PSNR: 峰值信噪比（值越大越好，>30dB为佳）
"""

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")#创建一个ArgumentParser对象，用于解析命令行参数。
    lp = ModelParams(parser)#初始化与模型相关的参数
    op = OptimizationParams(parser)#初始化与优化相关的参数。
    pp = PipelineParams(parser)#初始化与数据处理管道相关的参数。
    #一些具体的命令行参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False) # 默认为False，如果设置为True，则会在下面启用自动梯度检测异常功能
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000,30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000,30_000])
    parser.add_argument("--quiet", action="store_true")# action="store_true"表示该参数是一个布尔值，如果在命令行中出现该参数，则将其值设置为True
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[]) # 保存检查点的迭代次数，默认为空
    parser.add_argument("--start_checkpoint", type=str, default = None) # 开始训练的检查点，默认为空
    
    #parser.set_defaults(test_iterations=-1)
    #parser.set_defaults(densification_interval=200)
    args = parser.parse_args(sys.argv[1:])#从命令行参数中解析出用户输入的参数，并将这些参数存储在变量args中。若没有输入则用默认值
    args.save_iterations.append(args.iterations)# 保存模型的迭代次数 ，args.iterations是训练的总迭代次数，默认是30000，保存7000，30000和所有
    
    print("Optimizing " + args.model_path)#只有这个输出没有时间戳

    # Initialize system state (RNG)
    safe_state(args.quiet) # 通过重定向标准输出来为每个打印的行添加时间戳，
    # 当提供 --quiet 参数时，代码将减少输出信息，可能用于日志记录或调试，避免过多的控制台输出干扰。

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port) # 初始化一个 GUI 服务器，以便在训练过程中与外部 GUI 客户端进行通信，如果不需要 GUI 交互，可以注释掉
    torch.autograd.set_detect_anomaly(args.detect_anomaly)#主要用于设置 PyTorch 的自动梯度检测异常功能。根据参数值决定是否开启。
    # 该功能可以帮助我们定位代码中的梯度计算错误，并帮助我们修复这些错误。
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
