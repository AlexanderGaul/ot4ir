import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class Backbone(nn.Module) :
    def __init__(self) :
        super(Backbone, self).__init__()

        model = models.resnet34(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])

        self.num_channels = 512

    def forward(self, batch) :
        output = self.model(batch)
        return output


class MaxAttNet(nn.Module) :
    def __init__(self, use_location=True, use_attention=True, use_unique=True) :
        super(MaxAttNet, self).__init__()

        self.backbone = Backbone()

        self.locations = EncodeLocations()

        self.attention = Attention()

        self.use_unique = use_unique
        self.use_attention = use_attention
        self.use_location = use_location

    def forward(self, batch) :
        features = self.backbone(batch)
        N, C, H, W = features.shape

        # grid
        if self.use_location :
            h = torch.FloatTensor(range(H))
            w = torch.FloatTensor(range(W))
            loc_x, loc_y = torch.meshgrid(h, w)
            loc = torch.cat((loc_x.reshape(-1,1),  loc_y.reshape(-1,1)), dim=1)
            features = features + self.locations(loc).repeat(N, C, 1, 1)

        if self.use_attention :
            features = self.attention(features.flatten(-2)).reshape(N, C, H, W)

        if self.use_unique :
            scores_all, indices_flat = max_locations(features)

            scores = []
            locations = []
            features_sel = []

            for i in range(N) :
                indices_unique, inverse = torch.unique(indices_flat[i, :], return_inverse=True, dim=0)
                # indices to 2d
                score = torch.zeros(len(indices_unique))
                score.index_add_(0, inverse, scores_all[i, :])
                scores.append(score)

                indices_2d = torch.cat((indices_unique.unsqueeze(1) / W, indices_unique.unsqueeze(1) % W), dim=1)
                locations.append(indices_2d)

                # features should be C x num_selected
                features_sel.append(features[i, :].flatten(-2)[:, indices_unique])
        else :
            scores, locations, features_sel = max_features(features)

        return scores, locations, features_sel, features



class EncodeLocations(nn.Module) :
    def __init__(self, num_channels) :
        super(EncodeLocations, self).__init__()

        self.encoder = torch.nn.Linear(2, num_channels)
    def forward(self, batch_locations) :
        return self.encoder(batch_locations)

class Attention(nn.Module) :
    def __init__(self, num_channels):
        super(Attention, self).__init__()

        self.Q = torch.nn.Conv2d(num_channels, num_channels, 1)
        self.K = torch.nn.Conv2d(num_channels, num_channels, 1)
        self.V = torch.nn.Conv2d(num_channels, num_channels, 1)

    def forward(self, batch_flat):
        N, C, HW = batch_flat.shape
        Q = self.Q(batch_flat)
        K = self.K(batch_flat)
        V = self.V(batch_flat)

        att = F.softmax(torch.bmm(Q.transpose(1,2), K), dim=2)
        att_features = torch.mm(att, batch_flat.transpose(1,2))
        return att_features.transpose(1,2)



def max_locations(features) :
    N, C, H, W = features.shape

    scores, indices = F.max_pool2d_with_indices(features, (H,W))
    indices = indices.reshape(N, C)
    scores = scores.reshape(N, C)

    return scores, indices

def unique_locations(locations) :
    torch.unique(locations, return_inverse=True)

def select_features(features, indices_flat) :
    N, C, H, W = features.shape
    indices_flat = (indices_flat[:, :, 0]*W + indices_flat[:, :, 1]).squeeze(-1)
    features_sel = features.flatten(-2).gather(2, indices_flat.unsqueeze(1).repeat(1, C, 1))


def max_features(features) :
    N, C, H, W = features.shape
    scores, indices = max_locations(features)

    # N x C x C_sel
    m_features = features.flatten(-2).gather(2, indices.unsqueeze(1).repeat(1, C, 1))


    indices_2d = torch.cat((indices.unsqueeze(2)/W, indices.unsqueeze(2)%W), dim=2)

    return scores, indices_2d, m_features





