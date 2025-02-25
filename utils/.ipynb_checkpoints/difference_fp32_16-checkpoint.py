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
    ax.set_zlim(diff.min(), diff.max())
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # 添加颜色条
    # ax.text(500, 120, diff.max()*1.1, text, fontsize=12, color='red')

    # 设置视角
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(f'{save_path}/{save_name}.png')
    # 清理图形
    plt.close(fig)    
