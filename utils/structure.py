import torch
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
        Q_forsearch = high_order_residual(tmp, mask_forsearch, order=2) # 可能还要还要修改为残差和普通的差值
        target_weights = (tmp - Q_forsearch)**2 / (torch.diag(H).reshape((1, -1))) 
    else:
        raise NotImplementedError

    optimal_split, mask3 = structural_searching(target_weights, up_lim)
    mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    return mask1, mask2, mask3
