import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from cirtorch.datasets.genericdataset import ImagesFromList

from networks.functional import l2norm_dense

class AttRetrievalNet(nn.Module) :
    def __init__(self, resnet='34', freeze=-2, scale=4):
        super(AttRetrievalNet, self).__init__()
        
        if resnet == '50' :
            backbone = models.resnet50(pretrained = True)
            self.num_channels = 2048
        elif resnet == '101' :
            backbone = models.resnet101(pretrained = True)
            self.num_channels = 2048
        else :
            backbone = models.resnet34(pretrained = True)
            self.num_channels = 512

        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # TODO ad transofrm parameters
        self.mean = []
        self.std = []

        self.scale=scale

        if not freeze == 0 :
            for param in self.features[:freeze].parameters() :
                param.requires_grad = False
            

        self.att = FeatureAttention(self.num_channels)

    def forward(self, input):
        x = self.features(input)
        if not self.scale == 1 :
            x_re = F.max_pool2d(x, self.scale, self.scale)
        else :
            x_re = x
        N, C, H, W = x_re.shape
        att = F.softmax( self.att(x_re).reshape(N, 1, -1), dim=2 ).reshape(N, 1, H, W)
        x_re = l2norm_dense(x_re) # NORMALIZE LENGTH
        scaled = x_re * att
        agg = scaled.sum((2, 3))

        return agg, x_re, att


class FeatureAttention(nn.Module):
    """
    TODO find out convolution modules
    TODO use activation
    TODO use batchnorm
    """

    def __init__(self, feature_dim):
        super(FeatureAttention, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=feature_dim, out_channels=512, kernel_size=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)

    def forward(self, features):
        x = self.conv1(features)
        x = self.relu(x)
        att = self.conv2(x)
        return att

def extract_vectors(net, images, image_size, transform) :
    net.cuda()
    net.eval()

    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=None, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    with torch.no_grad():
        vecs = torch.zeros(net.num_channels, len(images))
        for i, input in enumerate(loader):
            input = input.cuda()

            vecs[:, i],  _, _ = net(input)

    return vecs
