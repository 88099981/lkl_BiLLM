import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from utils.autosearch import structural_searching
from utils.mask import generate_structural_mask
from binary import high_order_residual
'''
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
'''
def structural_guassian_distribution(tmp, H=None, metric="magnitude", up_lim=30):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    elif metric == "lkl_hessian":
        mask_forsearch = torch.ones_like(tmp, dtype=torch.bool)
        # Q_forsearch_1 = high_order_residual(tmp, mask_forsearch, order=1)
        Q_forsearch_2 = high_order_residual(tmp, mask_forsearch, order=2)
        target_weights_lkl = (tmp - Q_forsearch_2)**2 / (torch.diag(H).reshape((1, -1))) 
        target_weights_billm = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    else:
        raise NotImplementedError
    
    optimal_split, mask3, chosen_columns_lkl = structural_searching(target_weights_lkl, up_lim)
    optimal_split, mask3, chosen_columns_billm = structural_searching(target_weights_billm, up_lim)
    mask1, mask2 = generate_structural_mask(target_weights_lkl, mask3, optimal_split)


    # 可视化部分

    # 创建3D surface plots
    def create_3d_surface(tensor, name, text=None):
        tensor_np = tensor.abs().detach().cpu().numpy()
        x = np.arange(tensor_np.shape[0])
        y = np.arange(tensor_np.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # 创建matplotlib的3D surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, tensor_np.T, cmap='coolwarm', edgecolor='none')
        ax.set_title(name, pad=20, fontsize=12)
        ax.set_xlabel('Token')
        ax.set_ylabel('Dimension')
        ax.set_zlabel('Value')
        ax.set_zlim(tensor_np.min(), tensor_np.max())
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # 添加颜色条
        ax.text(500, 120, tensor_np.max()*1.1, text, fontsize=12, color='red')

        # 设置视角
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        
        # 创建返回字典
        result = {
            f"{name}": wandb.Image(fig)
        }
        
        plt.savefig('test_3d.png')
        # 清理图形
        plt.close(fig)
        
        return result
    
    log_dict = {}
    log_dict.update(create_3d_surface(tmp, f"tmp_{metric}"))
    log_dict.update(create_3d_surface(target_weights_billm, f"target_weights_billm", text=chosen_columns_billm))
    log_dict.update(create_3d_surface(target_weights_lkl, f"target_weights_lkl", text=chosen_columns_lkl))
    

    wandb.log(log_dict)

    print(optimal_split, mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    return mask1, mask2, mask3
