import torch


def sampling_without_replacement(logp, k):
    def gumbel_like(u):
        return -torch.log(-torch.log(torch.rand_like(u) + 1e-7) + 1e-7)

    scores = logp + gumbel_like(logp)
    return scores.topk(k, dim=-1)[1]


"""
sample rays from image
"""


def sample_rays(mask, num_samples):
    B, H, W = mask.shape
    mask_unfold = mask.reshape(-1)
    indices = torch.rand_like(mask_unfold).topk(num_samples)[1]
    sampled_masks = (torch.zeros_like(
        mask_unfold).scatter_(-1, indices, 1).reshape(B, H, W) > 0)
    return sampled_masks
