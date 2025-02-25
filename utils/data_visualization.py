import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

def difference_fp32_16(fp32_path:str, fp16_path:str, save_name:str, save_path:str):
    fp32_tensor = torch.load(fp32_path)
    fp16_tensor = torch.load(fp16_path)
    diff = torch.abs(fp32_tensor - fp16_tensor).detach().cpu().numpy()

    x = np.arange(diff.shape[0])
    y = np.arange(diff.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # 创建matplotlib的3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, diff.T, cmap='coolwarm', edgecolor='none')
    ax.set_title(f'{save_name}', pad=20, fontsize=12)
    ax.set_xlabel('Out_Channel')
    ax.set_ylabel('In_Channel')
    ax.set_zlabel('Value')
    ax.set_zlim(diff.min(), diff.min()+0.5*(diff.max()-diff.min()))
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # 添加颜色条
    # fig.text(0.5, 0.8, f'{diff.mean()/fp32_tensor.mean()}', fontsize=12, color='red')

    # 设置视角
    ax.view_init(elev=22, azim=30)
    plt.tight_layout()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(f'{save_path}/{save_name}.png')
    # 清理图形
    plt.close(fig)    



def create_tensor_img(tensor_path:str, save_name:str, save_path:str):
    tensor = torch.load(tensor_path)
    tensor_np = tensor.abs().detach().cpu().numpy()
    x = np.arange(tensor_np.shape[0])
    y = np.arange(tensor_np.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # 创建matplotlib的3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, tensor_np.T, cmap='coolwarm', edgecolor='none')
    ax.set_title(f'{save_name}', pad=20, fontsize=12)
    ax.set_xlabel('Out_Channel')
    ax.set_ylabel('In_Channel')
    ax.set_zlabel('Value')
    ax.set_zlim(tensor_np.min(), tensor_np.min()+0.5*(tensor_np.max()-tensor_np.min()))
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # 添加颜色条
    # fig.text(0.5, 0.8, f'{diff.mean()/fp32_tensor.mean()}', fontsize=12, color='red')

    # 设置视角
    ax.view_init(elev=22, azim=30)
    plt.tight_layout()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(f'{save_path}/{save_name}.png')
    # 清理图形
    plt.close(fig)


def difference_ratio(fp32_path:str, fp16_path:str, save_name:str, save_path:str):
    fp32_tensor = torch.load(fp32_path)
    fp16_tensor = torch.load(fp16_path)
    diff = torch.abs(fp32_tensor - fp16_tensor)

    denominator = torch.where(fp16_tensor.abs() > 0, fp16_tensor.abs(), torch.tensor(1e-8))
    percentage_difference = (diff / denominator) * 100
    percentage_difference = torch.clamp(percentage_difference, min=0, max=200)

    percentage_difference = torch.where(percentage_difference > 100, 0, percentage_difference)

    percentage_difference = percentage_difference.detach().cpu().numpy()


    x = np.arange(percentage_difference.shape[0])
    y = np.arange(percentage_difference.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # 创建matplotlib的3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, percentage_difference.T, cmap='coolwarm', edgecolor='none')
    ax.set_title(f'{save_name}', pad=20, fontsize=12)
    ax.set_xlabel('Out_Channel')
    ax.set_ylabel('In_Channel')
    ax.set_zlabel('ratio(%)')
    ax.set_zlim(percentage_difference.min(), percentage_difference.max())
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # 添加颜色条
    # fig.text(0.5, 0.8, f'{diff.mean()/fp32_tensor.mean()}', fontsize=12, color='red')

    # 设置视角
    ax.view_init(elev=22, azim=30)
    plt.tight_layout()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(f'{save_path}/{save_name}.png')
    # 清理图形
    plt.close(fig)  