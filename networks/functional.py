import torch


def l2norm_dense(features, eps=1e-6) :
    # N x C x H x W
    return features / (torch.norm(features, p=2, dim=1, keepdim=True) + eps).expand_as(features)