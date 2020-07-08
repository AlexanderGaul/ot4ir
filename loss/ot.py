from utils.distance import pairwise_distances

import torch
import torch.nn as nn


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for iter in range(iters):
        # if iter % 20 == 0: print(iter)
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(M, mu, nu, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    M_ = M.unsqueeze(0)
    b, m, n = M_.shape
    one = M_.new_tensor(1)

    # trash
    # ms, ns = (m * one).to(M), (n * one).to(M)
    # norm = - (ms + ns).log()
    log_mu = mu.log().unsqueeze(0)
    log_nu = nu.log().unsqueeze(0)
    # log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    # log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    # log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(M_, log_mu, log_nu, iters)
    Z = Z  # multiply probabilities by M+N
    return Z.squeeze(0)


def ot_loss(query, target, label, margin, eps) :

    query_features, query_attention = query
    target_features, target_attention = target

    query_features = query_features.transpose(1,0)
    target_features = target_features.tranpose(1,0)

    ## select top k attention values ##
    # q_sel = torch.topk( query_att, 10 )
    # t_sel = torch.topk( attention[1:, :].reshape(-1, H*W), 10, dim=1 )
    #
    # qf_sel = query_features[q_sel]
    # tf_sel = target_features.reshape(N-1, -1, C).gather( 1, t_sel.unsqueeze(-1).repeat(C) )
    #
    # qat_sel = query_att[q_sel]
    # tat_sel = target_att.reshape(N-1, -1).gather( 1, t_sel )

    M = pairwise_distances(query_features, target_features)

    P = log_optimal_transport(M, query_attention, target_attention, 500).exp()

    # TODO label
    return (M * P).sum()


def otmatch_loss(query, target, label, margin, eps):
    """
        query, target tuples of features and scores flattened
    Returns:

    """
    query_features, query_attention = query
    target_features, target_attention = target
    query_features = query_features.transpose(1,0)
    target_features = target_features.transpose(1,0)


    ## select top k attention values ##
    # q_sel = torch.topk( query_att, 10 )
    # t_sel = torch.topk( attention[1:, :].reshape(-1, H*W), 10, dim=1 )
    #
    # qf_sel = query_features[q_sel]
    # tf_sel = target_features.reshape(N-1, -1, C).gather( 1, t_sel.unsqueeze(-1).repeat(C) )
    #
    # qat_sel = query_att[q_sel]
    # tat_sel = target_att.reshape(N-1, -1).gather( 1, t_sel )
    M = pairwise_distances(query_features, target_features)

    P = log_optimal_transport(M, query_attention, target_attention, 500).exp()

    distance = (query_attention.unsqueeze(1) * query_features - torch.mm(P, target_features)).norm(dim=1)
    # lbl = label[1:].unsqueeze(1).repeat(1, H * W).flatten().unsqueeze(1)
    return (0.5 * label * torch.pow(distance, 2) +
            0.5 * (1 - label) * torch.pow(torch.clamp(margin - distance, min=0),
                                                                             2)).sum()


class OTMatchContrastiveLoss(nn.Module):
    """CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(OTMatchContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, first, second, label):
        return otmatch_loss(first, second, label, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


# end OTContrastiveLoss

class OTContrastiveLoss(nn.Module):
    def __init__(self, margin=0.7, eps=1e-6):
        super(OTContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, first, second, label):
        return ot_loss(first, second, label, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

