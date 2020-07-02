import torch


def pairwise_distances(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return dist


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


def ot_loss(features, attention, label, margin, eps):
    """
    Args:
        features: N x C x W x H
        attention: N x W x H

    Returns:

    """
    N, C, H, W = features.shape

    query_features = features[0, :].reshape((-1, C))  # : C x (W*H)
    target_features = features[1:, :].reshape((-1, C, W * H)).permute(1, 0, 2).reshape((-1, C))

    query_att = attention[0, :].flatten()
    target_att = attention[1:, :].flatten().reshape((N - 1) * (H * W))
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

    P = log_optimal_transport(M, query_att, target_att, 500).exp()

    distance = (query_att.unsqueeze(1) * query_features - torch.mm(P, target_features)).norm(dim=1)
    lbl = label[1:].unsqueeze(1).repeat(1, H * W).flatten().unsqueeze(1)
    return (0.5 * lbl * torch.pow(distance, 2) + 0.5 * (1 - lbl) * torch.pow(torch.clamp(margin - distance, min=0),
                                                                             2)).sum()


class OTContrastiveLoss(nn.Module):
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
        super(OTContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, x, att, label):
        return ot_loss(x, att, label, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'
# end OTContrastiveLoss
